"""
Architecture reconstruction for the deployed DP-CXR model.

The classes here mirror the notebook definitions exactly. They are re-declared
rather than unpickled on purpose: a pickled ``nn.Module`` carries the class path
and the torch version it was saved under, so it breaks the first time either
changes. A state dict plus an explicit class definition survives both.

Both encoders are frozen and are public pretrained checkpoints, so the bundle
ships only the head. The encoders download once and cache under ``MODEL_CACHE``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Cache the encoder weights next to the service rather than in the user's home,
# so a container rebuild does not re-download half a gigabyte.
MODEL_CACHE = Path(os.environ.get("DP_CXR_CACHE", Path(__file__).parent / ".cache"))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))


# ──────────────────────────────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────────────────────────────
class MultiViewImageEncoder(nn.Module):
    """Shared frozen CXR-pretrained DenseNet-121 over frontal + lateral.

    Shared weights across views, not one encoder per view: a lateral-specific
    encoder would only ever see the minority of studies that have one, and
    sharing halves the parameter count -- which matters under DP, where noise
    scales with the number of trainable parameters.
    """

    def __init__(self, weights: str = "densenet121-res224-all", fusion: str = "concat"):
        super().__init__()
        import torchxrayvision as xrv

        base = xrv.models.DenseNet(weights=weights)
        self.features = base.features
        self.fusion = fusion
        d = 1024
        self.out_features = 2 * d if fusion == "concat" else d
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.eval()

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.relu(self.features(x))
        return torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, frontal, lateral=None, has_lateral=None):
        ef = self._embed(frontal)
        if lateral is None:
            el = torch.zeros_like(ef)
            m = torch.zeros(ef.shape[0], 1, device=ef.device)
        else:
            el = self._embed(lateral)
            m = (
                has_lateral.view(-1, 1).float()
                if has_lateral is not None
                else torch.ones(ef.shape[0], 1, device=ef.device)
            )
            # Gate rather than zero-pad: an absent view contributes nothing
            # instead of contributing a zero vector the head must learn to ignore.
            el = el * m
        if self.fusion == "concat":
            return torch.cat([ef, el], dim=-1)
        if self.fusion == "max":
            return torch.maximum(ef, el)
        return (ef + el) / (1.0 + m).clamp(min=1.0)


class TextEncoder(nn.Module):
    """Frozen Bio_ClinicalBERT, [CLS] pooling."""

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name)
        self.out_features = self.bert.config.hidden_size
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]


# ──────────────────────────────────────────────────────────────────────────────
# Fusion heads
# ──────────────────────────────────────────────────────────────────────────────
class LateFusionModel(nn.Module):
    """Independent per-modality logits, averaged.

    Primary in the study: interpretable (each modality's contribution is a
    separate logit), degrades gracefully when a modality is absent, and small
    enough that DP noise does not swamp it.
    """

    def __init__(self, img_dim: int, txt_dim: int, num_classes: int):
        super().__init__()
        self.img_head = nn.Linear(img_dim, num_classes)
        self.txt_head = nn.Linear(txt_dim, num_classes)

    def forward(self, img_feat, txt_feat, img_present=None, txt_present=None):
        li = self.img_head(img_feat)
        lt = self.txt_head(txt_feat)
        if img_present is None and txt_present is None:
            return (li + lt) / 2.0
        # Average over the modalities actually supplied. Halving a single
        # available logit would shrink every probability toward 0.5 and make a
        # unimodal request look artificially uncertain.
        ip = (torch.ones(li.shape[0], 1, device=li.device)
              if img_present is None else img_present.view(-1, 1).float())
        tp = (torch.ones(lt.shape[0], 1, device=lt.device)
              if txt_present is None else txt_present.view(-1, 1).float())
        return (li * ip + lt * tp) / (ip + tp).clamp(min=1.0)


class EarlyFusionModel(nn.Module):
    """Concatenate features into an MLP -- the controlled comparison."""

    def __init__(self, img_dim: int, txt_dim: int, num_classes: int,
                 hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img_feat, txt_feat, img_present=None, txt_present=None):
        return self.classifier(torch.cat([img_feat, txt_feat], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# Assembled deployable model
# ──────────────────────────────────────────────────────────────────────────────
class DPCXRModel(nn.Module):
    """Encoders + trained head, with explicit modality presence flags.

    ``forward`` takes presence masks rather than inferring "missing" from an
    all-zero tensor: a legitimately dark radiograph and an absent one are not
    the same input, and guessing between them at inference is how a service
    quietly starts returning confident nonsense.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.labels = cfg["labels"]
        self.image_encoder = MultiViewImageEncoder(
            weights=cfg.get("xrv_weights", "densenet121-res224-all"),
            fusion=cfg.get("view_fusion", "concat"),
        )
        self.text_encoder = TextEncoder(cfg["bert_model"])
        img_dim = self.image_encoder.out_features
        txt_dim = self.text_encoder.out_features

        if cfg["fusion_type"] == "late":
            self.head = LateFusionModel(img_dim, txt_dim, cfg["num_classes"])
        else:
            self.head = EarlyFusionModel(
                img_dim, txt_dim, cfg["num_classes"],
                hidden=cfg.get("early_hidden", 512),
                dropout=cfg.get("early_dropout", 0.2),
            )
        self.eval()

    # -- feature extraction ---------------------------------------------------
    def encode_image(self, frontal, lateral=None, has_lateral=None):
        return self.image_encoder(frontal, lateral, has_lateral)

    def encode_text(self, ids, mask):
        return self.text_encoder(ids, mask)

    def forward(self, frontal=None, lateral=None, has_lateral=None,
                input_ids=None, attention_mask=None,
                img_present=None, txt_present=None, device="cpu"):
        if frontal is not None:
            img_feat = self.encode_image(frontal, lateral, has_lateral)
            bs = frontal.shape[0]
        else:
            bs = input_ids.shape[0]
            img_feat = torch.zeros(bs, self.image_encoder.out_features, device=device)

        if input_ids is not None:
            txt_feat = self.encode_text(input_ids, attention_mask)
        else:
            txt_feat = torch.zeros(bs, self.text_encoder.out_features, device=device)

        return self.head(img_feat, txt_feat, img_present, txt_present)

    # -- loading --------------------------------------------------------------
    def load_head(self, state_dict: dict) -> tuple[list, list]:
        """Load head weights saved by the notebook.

        The notebook's checkpoint keys are flat (``img_head.weight``,
        ``classifier.0.weight``); here the head sits one level down under
        ``head.``. Remap rather than requiring the notebook to know about the
        service's class layout.
        """
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("head."):
                remapped[k[5:]] = v
            elif k.startswith(("img_head.", "txt_head.", "classifier.")):
                remapped[k] = v
        missing, unexpected = self.head.load_state_dict(remapped, strict=False)
        return list(missing), list(unexpected)


def load_bundle(bundle_dir: str | Path, device: str = "cpu"):
    """Rebuild the model from an exported bundle.

    Returns ``(model, cfg, thresholds, model_card, text_rule)``.
    """
    b = Path(bundle_dir)
    required = ["bundle_config.json", "head_state_dict.pth"]
    for f in required:
        if not (b / f).exists():
            raise FileNotFoundError(
                f"{f} missing from {b}. Unpack dp_cxr_deployment_bundle.zip "
                f"(produced by notebook section 8.2) into this directory."
            )

    cfg = json.loads((b / "bundle_config.json").read_text())
    model = DPCXRModel(cfg).to(device)
    sd = torch.load(b / "head_state_dict.pth", map_location=device)
    missing, unexpected = model.load_head(sd)
    if missing:
        # Never silent: an unloaded head is a randomly initialised head, and it
        # will happily return plausible-looking probabilities.
        raise RuntimeError(
            f"Head weights incomplete -- missing {missing}. The bundle does not "
            f"match fusion_type={cfg['fusion_type']!r}. Re-export section 8.2."
        )
    model.eval()

    thresholds = (json.loads((b / "thresholds.json").read_text())
                  if (b / "thresholds.json").exists()
                  else {l: 0.5 for l in cfg["labels"]})
    card = (json.loads((b / "model_card.json").read_text())
            if (b / "model_card.json").exists() else {})
    text_rule = (json.loads((b / "text_rule.json").read_text())
                 if (b / "text_rule.json").exists() else {})
    return model, cfg, thresholds, card, text_rule
