"""
Exercise every request shape the service accepts.

    python -m dp_cxr_service.test_client                     # HTTP, needs the server up
    python -m dp_cxr_service.test_client --direct            # in-process, no server
    python -m dp_cxr_service.test_client --image path/to.jpg

With no image supplied it generates a synthetic one, so the three code paths can
be checked before real data is on hand. A synthetic radiograph produces
meaningless probabilities -- it verifies plumbing, not accuracy.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

SAMPLE_REPORT = """CLINICAL HISTORY: 64-year-old male with acute shortness of breath
and a three-day history of productive cough. Known hypertension.

TECHNIQUE: Frontal and lateral chest radiographs.

COMPARISON: Prior study dated 12 March.

FINDINGS: Patchy airspace opacity in the right lower lobe. Small right pleural
effusion. Cardiomediastinal silhouette is enlarged.

IMPRESSION: Right lower lobe pneumonia with a small parapneumonic effusion.
Cardiomegaly.
"""


def _synthetic_cxr(size: int = 512):
    """A crude chest-shaped gradient. Plumbing test only."""
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2, size / 2
    body = np.exp(-(((x - cx) / (size * 0.34)) ** 2 + ((y - cy) / (size * 0.42)) ** 2))
    lungs = sum(
        -0.45 * np.exp(-(((x - cx + s * size * 0.16) / (size * 0.11)) ** 2
                         + ((y - cy + size * 0.03) / (size * 0.20)) ** 2))
        for s in (1, -1)
    )
    img = np.clip((body + lungs) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def _show(title: str, r: dict):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"  modalities : {', '.join(r.get('modalities_used', []))}")
    print(f"  latency    : {r.get('inference_ms')} ms")
    print("\n  Predictions:")
    for p in r.get("predictions", []):
        flag = "  <-- POSITIVE" if p["positive"] else ""
        bar = "#" * int(p["probability"] * 24)
        print(f"    {p['label']:<20} {p['probability']:.3f} "
              f"(thr {p['threshold']:.2f}) {bar}{flag}")

    ex = r.get("explanations", {})
    if "gradcam" in ex:
        n = len(ex["gradcam"].get("overlay_png", ""))
        print(f"\n  Grad-CAM   : {ex['gradcam']['label']} — {n:,}-char PNG data URI")
    if "token_attribution" in ex:
        print("\n  Most influential tokens:")
        for t in ex["token_attribution"]["tokens"][:8]:
            print(f"    {t['token']:<18} {t['influence']:.5f}")
    if "modality_contribution" in ex:
        m = ex["modality_contribution"]
        print(f"\n  Modality shift — image {m['image_only_shift']:.4f}, "
              f"text {m['text_only_shift']:.4f}")

    md = r.get("model_metadata")
    if md:
        pv = md["privacy"]
        eps = pv.get("epsilon")
        print(f"\n  Model      : {md['run_name']} ({md['fusion_type']} fusion)")
        print(f"  Privacy    : {pv.get('mechanism')}"
              + (f" | {pv.get('placement')}" if pv.get("placement") else "")
              + (f" | eps={eps:.2f}" if isinstance(eps, (int, float)) else ""))
        print(f"  Calibration: ECE {md['calibration'].get('ece')}")

    tp = r.get("text_processing")
    if tp:
        print(f"\n  Text       : {tp.get('original_chars')} -> {tp.get('kept_chars')} chars"
              f" | headers found: {tp.get('had_sections')}")

    for w in r.get("warnings", []):
        print(f"\n  WARNING: {w}")


def run_direct(image_path: str | None):
    from PIL import Image
    from .predictor import DPCXRPredictor

    bundle = Path(__file__).parent / "bundle"
    if not (bundle / "bundle_config.json").exists():
        print(f"No bundle at {bundle}. Unpack dp_cxr_deployment_bundle.zip there.")
        return 1

    p = DPCXRPredictor(bundle)
    img = Image.open(image_path) if image_path else _synthetic_cxr()
    if not image_path:
        print("\nNOTE: synthetic image — probabilities are meaningless, this only "
              "checks that the pipeline runs.")

    _show("1 · CXR + REPORT (both modalities)", p.predict(image=img, report_text=SAMPLE_REPORT))
    _show("2 · CXR ONLY", p.predict(image=img))
    _show("3 · REPORT ONLY", p.predict(report_text=SAMPLE_REPORT))

    print(f"\n{'=' * 68}\n4 · LEAKAGE CHECK — full report vs pre-diagnostic only\n{'=' * 68}")
    full = p.predict(report_text=SAMPLE_REPORT, include_token_attribution=False)
    pre = p.predict(
        report_text="CLINICAL HISTORY: 64-year-old male with shortness of breath.",
        include_token_attribution=False,
    )
    print(f"  {'label':<20}{'full report':>13}{'history only':>15}{'delta':>9}")
    fa = {x["label"]: x["probability"] for x in full["predictions"]}
    pa = {x["label"]: x["probability"] for x in pre["predictions"]}
    worst = 0.0
    for lab in fa:
        d = fa[lab] - pa[lab]
        worst = max(worst, abs(d))
        print(f"  {lab:<20}{fa[lab]:>13.3f}{pa[lab]:>15.3f}{d:>+9.3f}")
    print("\n  Both inputs are stripped to pre-diagnostic text server-side, so the")
    print("  columns should be close. A large delta means the stripping regex missed")
    print("  a section and diagnostic content reached the model.")
    print(f"  Largest delta: {worst:.3f}"
          + ("  <-- investigate the text rule" if worst > 0.15 else "  (acceptable)"))
    return 0


def run_http(base_url: str, image_path: str | None):
    import requests

    try:
        h = requests.get(f"{base_url}/health", timeout=10).json()
    except Exception as e:
        print(f"Cannot reach {base_url} ({e}).\nStart it with:\n"
              f"  uvicorn dp_cxr_service.app:app --port 8000")
        return 1
    print("Health:", json.dumps(h, indent=2))
    if not h.get("bundle_loaded"):
        return 1

    img = None
    if image_path:
        img = Path(image_path).read_bytes()
    else:
        buf = io.BytesIO()
        _synthetic_cxr().save(buf, format="PNG")
        img = buf.getvalue()
        print("\nNOTE: synthetic image — plumbing check only.")

    r = requests.post(f"{base_url}/predict",
                      files={"image": ("cxr.png", img, "image/png")},
                      data={"report_text": SAMPLE_REPORT}, timeout=120)
    r.raise_for_status()
    _show("1 · CXR + REPORT (multipart)", r.json())

    r = requests.post(f"{base_url}/predict",
                      files={"image": ("cxr.png", img, "image/png")}, timeout=120)
    _show("2 · CXR ONLY", r.json())

    r = requests.post(f"{base_url}/predict-json",
                      json={"report_text": SAMPLE_REPORT}, timeout=120)
    _show("3 · REPORT ONLY (JSON)", r.json())

    r = requests.post(f"{base_url}/predict-json", json={
        "image_base64": base64.b64encode(img).decode(),
        "report_text": SAMPLE_REPORT,
    }, timeout=120)
    _show("4 · BOTH (base64 JSON)", r.json())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="store_true", help="in-process, no HTTP server")
    ap.add_argument("--image", help="path to a real chest radiograph")
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    sys.exit(run_direct(a.image) if a.direct else run_http(a.url, a.image))
