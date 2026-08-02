"""
Inference-time explanations: Grad-CAM over the radiograph, occlusion over the
report text.

Both mirror the notebook implementations (sections 6.5 and 5.4) so an explanation
served here is the same object the thesis measured for stability under DP.

The Grad-CAM detail that matters: the backbone is frozen and normally runs under
``no_grad``, so the input must be cloned with ``requires_grad_(True)`` inside an
explicit ``enable_grad`` block. Without that the backward hook never fires and
every map silently comes back ``None`` -- no exception, just no explanation.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """Class-activation map for the image branch of a fusion model."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.encoder = model.image_encoder
        self.layer = self.encoder.features[-1]
        self.activations = None
        self.gradients = None
        self._handle = None

    def _forward_hook(self, module, inp, out):
        self.activations = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def _image_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Score a single view through the image pathway only.

        Late fusion exposes an image head directly. Early fusion concatenates,
        so the image half of the first layer's weight matrix is used -- the same
        slice the notebook takes, so the served map matches the measured one.
        """
        e = self.encoder._embed(x)
        head = self.model.head
        if hasattr(head, "img_head"):
            w = head.img_head.weight
            if w.shape[1] == 2 * e.shape[1]:  # concat view fusion -> frontal half
                return torch.nn.functional.linear(
                    e, w[:, : e.shape[1]], head.img_head.bias
                )
            return head.img_head(e)
        first = head.classifier[0]
        w = first.weight[:, : e.shape[1]]
        h = torch.nn.functional.relu(torch.nn.functional.linear(e, w, first.bias))
        return head.classifier[-1](h)

    def generate(self, image_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> tuple[Optional[np.ndarray], int]:
        self._handle = self.layer.register_forward_hook(self._forward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            self.gradients = None
            x = image_tensor.clone().detach().to(self.device).requires_grad_(True)

            with torch.enable_grad():
                logits = self._image_logits(x)
                if class_idx is None:
                    class_idx = int(logits[0].argmax().item())
                logits[0, class_idx].backward()

            if self.gradients is None:
                return None, class_idx

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(1)).squeeze()
            cam = cam.detach().cpu().numpy()
            rng = cam.max() - cam.min()
            if rng <= 1e-6:
                return None, class_idx
            return (cam - cam.min()) / (rng + 1e-8), class_idx
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None


def cam_overlay_png(original: Image.Image, cam: np.ndarray,
                    alpha: float = 0.45, size: int = 384) -> str:
    """Blend a CAM over the radiograph and return a base64 PNG data URI."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    base = original.convert("L").resize((size, size), Image.LANCZOS)
    base_rgb = np.stack([np.asarray(base, dtype=np.float32) / 255.0] * 3, axis=-1)

    heat = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    heat_rgb = cm.jet(heat)[..., :3]

    blended = (1 - alpha) * base_rgb + alpha * heat_rgb
    out = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Token occlusion
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def token_attribution(model, tokenizer, input_ids, attention_mask,
                      img_feat=None, device: str = "cpu",
                      top_k: int = 12, max_tokens: int = 128,
                      chunk: int = 32) -> list[dict]:
    """Mask each token in turn; record the change in mean predicted probability.

    Variants are batched -- one forward per chunk rather than one per token,
    which is the difference between a response in under a second and one in
    thirty.
    """
    model.eval()
    ids = input_ids.to(device)
    mask = attention_mask.to(device)
    n = min(int(mask.sum().item()), max_tokens)
    if n == 0:
        return []

    txt_feat = model.encode_text(ids, mask)
    if img_feat is None:
        img_feat = torch.zeros(1, model.image_encoder.out_features, device=device)
        img_present = torch.zeros(1, device=device)
    else:
        img_present = torch.ones(1, device=device)
    base = torch.sigmoid(
        model.head(img_feat, txt_feat, img_present, torch.ones(1, device=device))
    )[0]

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.pad_token_id

    deltas = []
    for s in range(0, n, chunk):
        idx = list(range(s, min(s + chunk, n)))
        variants = ids.repeat(len(idx), 1).clone()
        for row, t in enumerate(idx):
            variants[row, t] = mask_id
        tf = model.encode_text(variants, mask.repeat(len(idx), 1))
        probs = torch.sigmoid(
            model.head(
                img_feat.repeat(len(idx), 1), tf,
                img_present.repeat(len(idx)), torch.ones(len(idx), device=device),
            )
        )
        deltas.append((probs - base.unsqueeze(0)).abs().mean(1).cpu().numpy())

    deltas = np.concatenate(deltas)
    tokens = tokenizer.convert_ids_to_tokens(ids[0, :n].tolist())

    rows = [
        {"token": t, "influence": float(d), "position": i}
        for i, (t, d) in enumerate(zip(tokens, deltas))
        if t not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    rows.sort(key=lambda r: r["influence"], reverse=True)
    return rows[:top_k]


def modality_contribution(model, img_feat, txt_feat, device: str = "cpu") -> dict:
    """How much each modality moved the prediction, for this request.

    Runs the head three times -- image only, text only, both -- and reports the
    mean absolute difference from the joint prediction. The inference-time
    analogue of the modality ablation in section 4.4.
    """
    with torch.no_grad():
        one = torch.ones(1, device=device)
        zero = torch.zeros(1, device=device)
        zi = torch.zeros_like(img_feat)
        zt = torch.zeros_like(txt_feat)

        p_both = torch.sigmoid(model.head(img_feat, txt_feat, one, one))[0]
        p_img = torch.sigmoid(model.head(img_feat, zt, one, zero))[0]
        p_txt = torch.sigmoid(model.head(zi, txt_feat, zero, one))[0]

    return {
        "image_only_shift": float((p_both - p_txt).abs().mean()),
        "text_only_shift": float((p_both - p_img).abs().mean()),
        "note": "Mean |change| in predicted probability when the other modality "
                "is withheld. Larger means this request leaned more on that modality.",
    }
