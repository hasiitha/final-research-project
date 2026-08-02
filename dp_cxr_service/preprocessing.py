"""
Input preparation -- must match training exactly.

Two rules here are not cosmetic:

**The pre-diagnostic text rule.** CheXbert labels are extracted from the report's
Impression, and Findings restates the same conclusions. The model was trained on
neither. Tokenising a full report at inference feeds it label-bearing vocabulary
it never saw, which both degrades the prediction and reintroduces exactly the
leakage this study was built to remove. ``clean_report_text`` applies the regex
from notebook section 1.7, shipped in the bundle so the two cannot drift apart.

**The xrv normalisation.** TorchXRayVision DenseNet expects [-1024, 1024], not
ImageNet statistics. Getting this wrong produces confident, meaningless output --
no error, just wrong numbers.
"""
from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:                 # torch is only needed by ImagePreprocessor;
    import torch                  # importing it lazily keeps the text rule
                                  # unit-testable without the full ML stack

# Fallback copies of the section headers; the bundle's text_rule.json overrides
# these when present so the service tracks whatever the notebook actually used.
_PRE_DEFAULT = (
    r"(?:CLINICAL HISTORY|HISTORY|INDICATION|REASON FOR EXAM(?:INATION)?|"
    r"TECHNIQUE|COMPARISON|PROCEDURE COMMENTS)"
)
_ALL_DEFAULT = (
    r"(?:NARRATIVE|CLINICAL HISTORY|HISTORY|INDICATION|REASON FOR EXAM(?:INATION)?|"
    r"TECHNIQUE|COMPARISON|PROCEDURE COMMENTS|FINDINGS|IMPRESSION|"
    r"END OF IMPRESSION|SUMMARY|ACCESSION NUMBER)"
)

# Terms that appear only in a radiologist's conclusion. Used to warn when a
# caller submits text that looks diagnostic even though no section header
# matched -- a free-text note can leak the answer without any structure at all.
_DIAGNOSTIC_TERMS = (
    "atelecta", "cardiomeg", "edema", "effusion", "pneumon",
    "consolidat", "opacit", "infiltrat", "enlarged cardiomediastin",
)


class TextCleaner:
    """Applies the training-time pre-diagnostic rule, and reports what it did."""

    def __init__(self, text_rule: Optional[dict] = None):
        rule = text_rule or {}
        self.pre = rule.get("pre_sections", _PRE_DEFAULT)
        self.all_headers = rule.get("all_headers", _ALL_DEFAULT)
        self._re = re.compile(
            rf"({self.pre})\s*:?\s*(.*?)(?=\n?\s*{self.all_headers}\s*:|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        self._diag = re.compile(
            "(" + "|".join(t + "[a-z]*" for t in _DIAGNOSTIC_TERMS) + ")",
            re.IGNORECASE,
        )

    def clean(self, report: str) -> dict:
        """Return the model-safe text plus an audit of what was removed."""
        if not isinstance(report, str) or not report.strip():
            return {"text": "", "had_sections": False, "removed_diagnostic": False,
                    "warnings": ["empty report text"]}

        kept = [m.group(2) for m in self._re.finditer(report)]
        warnings: list[str] = []

        if kept:
            text = re.sub(r"\s+", " ", " ".join(kept)).strip()
            had_sections = True
        else:
            # No recognised headers. Rather than silently discard the input,
            # use it as-is and say so -- but check it for diagnostic vocabulary,
            # because unstructured text can still contain the answer.
            text = re.sub(r"\s+", " ", report).strip()
            had_sections = False
            warnings.append(
                "No pre-diagnostic section headers found; the text was used as "
                "supplied. If it contains Findings or Impression content, the "
                "prediction is not comparable to the reported evaluation."
            )

        leaked = self._diag.findall(text)
        if leaked:
            warnings.append(
                f"Diagnostic vocabulary present in the submitted text "
                f"({', '.join(sorted(set(w.lower() for w in leaked))[:5])}). The "
                f"model was trained on pre-diagnostic text only; treat this "
                f"prediction as unreliable."
            )

        return {
            "text": text,
            "had_sections": had_sections,
            "removed_diagnostic": bool(kept) and len(text) < len(report),
            "original_chars": len(report),
            "kept_chars": len(text),
            "warnings": warnings,
        }


class ImagePreprocessor:
    """PIL image -> the tensor the frozen DenseNet expects."""

    def __init__(self, img_size: int = 224, norm: Optional[dict] = None):
        self.img_size = img_size
        n = norm or {}
        self.scale = float(n.get("scale", 2048.0))
        self.offset = float(n.get("offset", -1024.0))

    def __call__(self, img: Image.Image) -> "torch.Tensor":
        """Returns (1, 1, H, W). Single channel -- xrv DenseNet is not RGB."""
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        a = np.asarray(img, dtype=np.float32) / 255.0
        # xrv scale: [0,1] -> [-1024, 1024]
        a = a * self.scale + self.offset
        import torch
        return torch.from_numpy(a).unsqueeze(0).unsqueeze(0)


def build_tokenizer(bert_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(bert_model)


def tokenize(tokenizer, text: str, max_len: int = 128):
    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]
