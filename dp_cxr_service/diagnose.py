#!/usr/bin/env python3
"""Find out exactly which step of a prediction is failing.

The service catches errors and returns a one-line message, which is rarely
enough to locate the cause. This walks the same path stage by stage and prints a
full traceback at the first thing that breaks.

    source .venv/bin/activate
    python dp_cxr_service/diagnose.py
"""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so `dp_cxr_service` is importable

STEP = 0


def step(name):
    global STEP
    STEP += 1
    print(f"\n[{STEP}] {name}")


def died(e):
    print(f"    FAILED — {type(e).__name__}: {e}\n")
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("Send everything above. The last frame in the traceback is the cause.")
    sys.exit(1)


print("=" * 70)
print("DP-CXR prediction diagnostic")
print("=" * 70)

# ── 1 · imports ─────────────────────────────────────────────────────────────
step("Importing the ML stack")
try:
    import numpy as np
    import torch
    from PIL import Image
    print(f"    torch {torch.__version__} | numpy {np.__version__}")
    import transformers, torchxrayvision  # noqa: F401
    print(f"    transformers {transformers.__version__} | "
          f"torchxrayvision {torchxrayvision.__version__}")
except Exception as e:
    print("    Something is not installed. Re-run:")
    print("      pip install -r dp_cxr_service/requirements.txt")
    died(e)

# ── 2 · bundle ──────────────────────────────────────────────────────────────
step("Loading the bundle")
try:
    from dp_cxr_service.model import load_bundle
    bundle = HERE / "bundle"
    model, cfg, thresholds, card, text_rule = load_bundle(bundle, device="cpu")
    print(f"    model    {cfg['run_name']} ({cfg['fusion_type']} fusion)")
    print(f"    labels   {', '.join(cfg['labels'])}")
    print(f"    encoders downloaded and loaded")
except Exception as e:
    print("    This is where the encoders download. If it failed on a network")
    print("    error, check your connection and try again — they cache after.")
    died(e)

# ── 3 · shapes: the most likely silent mismatch ─────────────────────────────
step("Checking that the head matches the encoders")
try:
    img_dim = model.image_encoder.out_features
    txt_dim = model.text_encoder.out_features
    print(f"    image encoder outputs {img_dim}, text encoder outputs {txt_dim}")
    head = model.head
    if hasattr(head, "img_head"):
        w_i = head.img_head.weight.shape
        w_t = head.txt_head.weight.shape
        print(f"    img_head expects {w_i[1]}, txt_head expects {w_t[1]}")
        bad = []
        if w_i[1] != img_dim:
            bad.append(f"img_head wants {w_i[1]} but the encoder gives {img_dim}")
        if w_t[1] != txt_dim:
            bad.append(f"txt_head wants {w_t[1]} but the encoder gives {txt_dim}")
        if bad:
            print("\n    MISMATCH:")
            for b in bad:
                print("      " + b)
            print("\n    The head was trained against a differently-shaped encoder.")
            print("    Most often this is view_fusion: 'concat' in bundle_config.json")
            print("    doubling the image dimension when the model was trained")
            print("    frontal-only. Try setting \"view_fusion\": \"mean\" there.")
            sys.exit(1)
        print("    shapes agree")
except Exception as e:
    died(e)

# ── 4 · text path ───────────────────────────────────────────────────────────
step("Running the text pipeline")
SAMPLE = ("CLINICAL HISTORY: 58-year-old male, acute dyspnoea and fever.\n"
          "INDICATION: Rule out infection.\n"
          "TECHNIQUE: Frontal AP portable radiograph.\n"
          "FINDINGS: Dense right lower lobe consolidation.\n"
          "IMPRESSION: Pneumonia.")
try:
    from dp_cxr_service.preprocessing import TextCleaner
    cleaned = TextCleaner(text_rule).clean(SAMPLE)
    print(f"    kept: {cleaned['text'][:90]}")
    leaked = [w for w in ("consolidation", "pneumonia") if w in cleaned["text"].lower()]
    print(f"    Findings/Impression removed: {'NO — ' + str(leaked) if leaked else 'yes'}")
except Exception as e:
    died(e)

# ── 5 · a real prediction on a synthetic radiograph ─────────────────────────
step("Predicting on a synthetic image plus that text")
try:
    from dp_cxr_service.predictor import DPCXRPredictor
    pred = DPCXRPredictor(str(bundle), device="cpu")
    img = Image.fromarray(
        (np.random.default_rng(0).normal(0.5, 0.15, (320, 320)).clip(0, 1) * 255
         ).astype("uint8"), mode="L")
    out = pred.predict(image=img, report_text=SAMPLE,
                       include_gradcam=True, include_token_attribution=True)
    print(f"    modalities used : {out['modalities_used']}")
    print(f"    inference       : {out.get('inference_ms')} ms")
    print("\n    label                probability  threshold  call")
    for r in out["predictions"]:
        print(f"    {r['label']:<20} {r['probability']:>10.4f} {r['threshold']:>10.3f}"
              f"  {'POSITIVE' if r['positive'] else 'negative'}")
    ex = out.get("explanations", {})
    print(f"\n    Grad-CAM        : {'produced' if 'gradcam' in ex else 'ABSENT'}")
    print(f"    token attribution: {'produced' if 'token_attribution' in ex else 'ABSENT'}")
    for w in out.get("warnings", []):
        print(f"    warning: {w}")
except Exception as e:
    died(e)

# ── 6 · unimodal paths ──────────────────────────────────────────────────────
step("Checking image-only and text-only requests")
for label, kw in (("image only", dict(image=img)),
                  ("text only", dict(report_text=SAMPLE))):
    try:
        o = pred.predict(include_gradcam=False, include_token_attribution=False, **kw)
        print(f"    {label:<12} ok — top finding {o['top_finding']} "
              f"({o['predictions'][0]['probability']:.4f})")
    except Exception as e:
        print(f"    {label:<12} FAILED")
        died(e)

print("\n" + "=" * 70)
print("Every stage passed. The model predicts correctly.")
print("If the browser still reports a failure, the problem is between the page")
print("and the server — check the terminal running uvicorn for a traceback.")
print("=" * 70)
