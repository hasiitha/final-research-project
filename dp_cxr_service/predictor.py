"""
The prediction engine.

Handles the three request shapes the service accepts -- CXR only, report only,
CXR + report -- and is explicit about the fact that they are not equally
reliable. The model was trained on both modalities present; a unimodal request
is out of distribution, and the response says so rather than returning a bare
number that looks like the multimodal one.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .model import load_bundle
from .preprocessing import ImagePreprocessor, TextCleaner, build_tokenizer, tokenize
from .xai import GradCAM, cam_overlay_png, modality_contribution, token_attribution


class DPCXRPredictor:
    """Loads a bundle once, then serves predictions.

    Construction is expensive (two pretrained encoders); prediction is not.
    Instantiate once at application start, never per request.
    """

    def __init__(self, bundle_dir: str | Path = "bundle", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg, self.thresholds, self.card, text_rule = load_bundle(
            bundle_dir, self.device
        )
        self.labels: list[str] = self.cfg["labels"]
        self.tokenizer = build_tokenizer(self.cfg["bert_model"])
        self.text_cleaner = TextCleaner(text_rule)
        self.image_prep = ImagePreprocessor(
            img_size=self.cfg.get("img_size", 224),
            norm=self.cfg.get("image_norm"),
        )
        self.max_text_len = self.cfg.get("max_text_len", 128)

        priv = (self.card.get("privacy") or {})
        eps = priv.get("achieved_epsilon")
        print(f"[DPCXRPredictor] {self.cfg['run_name']} | {self.cfg['fusion_type']} fusion "
              f"| device={self.device}")
        print(f"  labels    : {', '.join(self.labels)}")
        print(f"  privacy   : {priv.get('mechanism', 'unknown')}"
              + (f" | placement {priv.get('placement')}" if priv.get("placement") else "")
              + (f" | eps={eps:.2f}" if isinstance(eps, (int, float)) else ""))
        if priv.get("mechanism") == "NONE — not private":
            print("  WARNING: this bundle is NOT differentially private.")

    # -- helpers --------------------------------------------------------------
    def _encode_image(self, image: Image.Image):
        t = self.image_prep(image).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(t, None, torch.zeros(1, device=self.device))
        return t, feat

    def _encode_text(self, text: str):
        ids, mask = tokenize(self.tokenizer, text, self.max_text_len)
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(ids, mask)
        return ids, mask, feat

    # -- main entry point -----------------------------------------------------
    def predict(
        self,
        image: Optional[Image.Image] = None,
        report_text: Optional[str] = None,
        include_gradcam: bool = True,
        include_token_attribution: bool = True,
        include_metadata: bool = True,
        top_k_tokens: int = 12,
    ) -> dict:
        t0 = time.perf_counter()
        if image is None and (report_text is None or not str(report_text).strip()):
            raise ValueError("Supply at least one of: image, report_text.")

        warnings: list[str] = []
        text_audit = None

        # ---- encode whatever was supplied -----------------------------------
        if image is not None:
            img_tensor, img_feat = self._encode_image(image)
            img_present = torch.ones(1, device=self.device)
        else:
            img_tensor = None
            img_feat = torch.zeros(1, self.model.image_encoder.out_features,
                                   device=self.device)
            img_present = torch.zeros(1, device=self.device)

        if report_text and str(report_text).strip():
            text_audit = self.text_cleaner.clean(report_text)
            warnings += text_audit["warnings"]
            if text_audit["text"]:
                ids, mask, txt_feat = self._encode_text(text_audit["text"])
                txt_present = torch.ones(1, device=self.device)
            else:
                ids = mask = None
                txt_feat = torch.zeros(1, self.model.text_encoder.out_features,
                                       device=self.device)
                txt_present = torch.zeros(1, device=self.device)
                warnings.append(
                    "The report contained no usable pre-diagnostic text; the "
                    "prediction is image-only."
                )
        else:
            ids = mask = None
            txt_feat = torch.zeros(1, self.model.text_encoder.out_features,
                                   device=self.device)
            txt_present = torch.zeros(1, device=self.device)

        modalities = []
        if float(img_present) > 0:
            modalities.append("image")
        if float(txt_present) > 0:
            modalities.append("text")
        if not modalities:
            raise ValueError("Nothing usable was supplied after preprocessing.")

        # The evaluation in the thesis is for both modalities present. A
        # unimodal request is out of distribution and should not be reported
        # with the same confidence.
        if len(modalities) == 1:
            warnings.append(
                f"Unimodal request ({modalities[0]} only). The model was trained "
                f"and evaluated with both modalities present; see the "
                f"missing-modality stress test for the expected degradation."
            )

        # ---- predict ---------------------------------------------------------
        with torch.no_grad():
            logits = self.model.head(img_feat, txt_feat, img_present, txt_present)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        predictions = []
        for i, label in enumerate(self.labels):
            thr = float(self.thresholds.get(label, 0.5))
            predictions.append({
                "label": label,
                "probability": round(float(probs[i]), 4),
                "threshold": round(thr, 4),
                "positive": bool(probs[i] >= thr),
            })
        predictions.sort(key=lambda r: r["probability"], reverse=True)

        response: dict = {
            "predictions": predictions,
            "modalities_used": modalities,
            "top_finding": predictions[0]["label"],
            "warnings": warnings,
        }

        # ---- explanations ----------------------------------------------------
        explanations: dict = {}

        if include_gradcam and img_tensor is not None:
            try:
                top_idx = self.labels.index(predictions[0]["label"])
                cam, cls = GradCAM(self.model, self.device).generate(img_tensor, top_idx)
                if cam is not None:
                    explanations["gradcam"] = {
                        "label": self.labels[cls],
                        "overlay_png": cam_overlay_png(image, cam),
                        "note": "Qualitative illustration of where the image branch "
                                "attended. A plausible heatmap is not evidence of "
                                "clinical reasoning.",
                    }
                else:
                    warnings.append("Grad-CAM produced no gradient signal for this input.")
            except Exception as e:  # never fail a prediction because of an explanation
                warnings.append(f"Grad-CAM unavailable: {type(e).__name__}: {e}")

        if include_token_attribution and ids is not None:
            try:
                explanations["token_attribution"] = {
                    "tokens": token_attribution(
                        self.model, self.tokenizer, ids, mask,
                        img_feat=img_feat if float(img_present) > 0 else None,
                        device=self.device, top_k=top_k_tokens,
                        max_tokens=self.max_text_len,
                    ),
                    "note": "Change in mean predicted probability when each token is "
                            "masked. Diagnostic terms appearing here indicate the "
                            "submitted text leaked the answer.",
                }
            except Exception as e:
                warnings.append(f"Token attribution unavailable: {type(e).__name__}: {e}")

        if len(modalities) == 2:
            try:
                explanations["modality_contribution"] = modality_contribution(
                    self.model, img_feat, txt_feat, self.device
                )
            except Exception as e:
                warnings.append(f"Modality contribution unavailable: {e}")

        if explanations:
            response["explanations"] = explanations

        # ---- provenance ------------------------------------------------------
        if include_metadata:
            priv = self.card.get("privacy", {})
            perf = self.card.get("performance", {})
            response["model_metadata"] = {
                "run_name": self.cfg["run_name"],
                "fusion_type": self.cfg["fusion_type"],
                "privacy": {
                    "mechanism": priv.get("mechanism"),
                    "placement": priv.get("placement"),
                    "epsilon": priv.get("achieved_epsilon"),
                    "delta": priv.get("delta"),
                    "noise_multiplier": priv.get("noise_multiplier"),
                    "privacy_unit": priv.get("privacy_unit"),
                },
                "calibration": {
                    "ece": perf.get("ece"),
                    "mean_brier": perf.get("mean_brier"),
                    "note": "ECE measured on the held-out test split. Probabilities "
                            "are not perfectly calibrated; read them as ranked "
                            "evidence, not as literal risks.",
                },
                "test_macro_auroc": perf.get("macro_auroc"),
                "selection": self.card.get("selection", {}),
            }

        response["warnings"] = warnings
        response["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        response["disclaimer"] = (
            "Research demonstration for an MSc dissertation. Not a medical device, "
            "not validated for clinical use, and not a diagnosis. Must not be used "
            "to inform patient care."
        )
        if text_audit is not None:
            response["text_processing"] = {
                k: v for k, v in text_audit.items() if k != "warnings"
            }
        return response
