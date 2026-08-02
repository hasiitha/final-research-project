# DP-CXR — Deployment Service

Serves the differentially private multimodal chest X-ray model exported by
§8.2 of the dissertation notebooks. Accepts a radiograph, a report, or both, and
returns probabilities with Grad-CAM, token attributions and privacy provenance.

> **Not a medical device.** Research demonstration for an MSc dissertation.
> Outputs are not a diagnosis and must not inform patient care.

---

## Quick start

**1 — Export the bundle.** Run §8.1 and §8.2 in either notebook. They rank every
DP run on a composite score and write `dp_cxr_deployment_bundle.zip`.

**2 — Unpack it here.**

```bash
unzip dp_cxr_deployment_bundle.zip -d dp_cxr_service/bundle/
```

**3 — Run.**

```bash
pip install -r dp_cxr_service/requirements.txt
uvicorn dp_cxr_service.app:app --reload --port 8000
```

Open **http://localhost:8000/docs** — upload a radiograph and paste a report
straight from the browser. No client needed, which makes it a clean viva demo.

Docker instead:

```bash
docker build -t dp-cxr dp_cxr_service/
docker run -p 8000:8000 dp-cxr
```

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Which bundle is loaded, ε, device |
| `GET` | `/model-card` | Privacy budget, per-label performance, limitations |
| `POST` | `/predict` | Multipart — image file and/or `report_text` |
| `POST` | `/predict-json` | JSON — base64 image and/or `report_text` |

### Examples

```bash
# Both modalities
curl -X POST http://localhost:8000/predict \
  -F "image=@cxr.jpg" \
  -F "report_text=CLINICAL HISTORY: 64M with shortness of breath. TECHNIQUE: PA and lateral."

# Image only
curl -X POST http://localhost:8000/predict -F "image=@cxr.jpg"

# Report only
curl -X POST http://localhost:8000/predict -F "report_text=CLINICAL HISTORY: cough, fever."
```

### Use it as a library

```python
from dp_cxr_service import DPCXRPredictor
from PIL import Image

p = DPCXRPredictor("dp_cxr_service/bundle")

r = p.predict(image=Image.open("cxr.jpg"), report_text="CLINICAL HISTORY: ...")
for pred in r["predictions"]:
    print(pred["label"], pred["probability"], pred["positive"])
```

---

## Response shape

```json
{
  "predictions": [
    {"label": "Pleural Effusion", "probability": 0.7421,
     "threshold": 0.31, "positive": true}
  ],
  "modalities_used": ["image", "text"],
  "top_finding": "Pleural Effusion",
  "explanations": {
    "gradcam": {"label": "Pleural Effusion", "overlay_png": "data:image/png;base64,..."},
    "token_attribution": {"tokens": [{"token": "dyspnea", "influence": 0.0231}]},
    "modality_contribution": {"image_only_shift": 0.11, "text_only_shift": 0.04}
  },
  "model_metadata": {
    "run_name": "dp_f_late_fusion",
    "privacy": {"mechanism": "DP-SGD (Opacus)", "placement": "DP-F",
                "epsilon": 7.94, "delta": 1e-05},
    "calibration": {"ece": 0.0412},
    "test_macro_auroc": 0.781
  },
  "warnings": [],
  "inference_ms": 284.3
}
```

---

## Three behaviours worth knowing

### Report text is stripped before the model sees it

The model was trained on **pre-diagnostic** sections only — clinical history,
indication, technique, comparison. CheXbert labels are extracted from the
Impression, and Findings restates the same conclusions, so feeding either lets the
model read the answer instead of assessing the radiograph.

The service applies the identical regex from §1.7, shipped inside the bundle as
`text_rule.json` so the two cannot drift apart. Post a full report and Findings
and Impression are removed server-side; `text_processing` in the response reports
how many characters were dropped.

If the text contains diagnostic vocabulary that survived stripping — an
unstructured note, say — a warning is attached. Take those predictions as
unreliable rather than impressive.

### Unimodal requests are out of distribution

The model was trained and evaluated with both modalities present. Image-only and
text-only requests work, and late fusion degrades gracefully by design, but the
expected drop is characterised by the missing-modality stress test (§5.6), not by
the headline AUROC. Every unimodal response carries a warning saying so.

Late fusion averages over the modalities actually supplied rather than halving a
single available logit — otherwise every unimodal prediction would be dragged
toward 0.5 and look artificially uncertain.

### Thresholds come from the bundle, not from 0.5

Per-label cut-offs were tuned on **validation** (never test) in §2.4. Prevalence
ranges from 0.4 % (Pneumonia) to 11 % (Pleural Effusion), so a flat 0.5 would
report almost nothing positive. The bundle carries the tuned thresholds and the
service uses them, which keeps `positive` consistent with every table in the
thesis.

---

## Files

| File | Role |
|---|---|
| `model.py` | Architecture, mirrored from the notebook; bundle loading |
| `preprocessing.py` | Pre-diagnostic text rule, xrv image normalisation |
| `xai.py` | Grad-CAM, token occlusion, modality contribution |
| `predictor.py` | Prediction engine — the piece to import |
| `app.py` | FastAPI routes |
| `test_client.py` | Exercises all three request shapes |
| `bundle/` | **Unpack the exported bundle here** |

### Bundle contents

| File | Why |
|---|---|
| `head_state_dict.pth` | The only trained weights — encoders are frozen and public |
| `bundle_config.json` | Architecture, labels, tokenizer, image transform |
| `thresholds.json` | Validation-tuned per-label cut-offs |
| `text_rule.json` | The §1.7 regex, so inference matches training |
| `model_card.json` | ε, δ, placement, metrics, cohort, limitations |
| `selection_ranking.csv` | Why this model and not another |

A few megabytes, not half a gigabyte: only the head is project-specific. The
frozen `densenet121-res224-all` and `Bio_ClinicalBERT` checkpoints download once
and cache under `.cache/` (baked into the Docker image at build time).

---

## Testing

```bash
python -m dp_cxr_service.test_client --direct              # in-process, no server
python -m dp_cxr_service.test_client                       # against a running server
python -m dp_cxr_service.test_client --direct --image cxr.jpg
```

Without `--image` a synthetic radiograph is generated. That checks the plumbing,
not the accuracy — the probabilities will be meaningless.

The `--direct` run also prints a **leakage check**: the same report submitted in
full and as history only. Both are stripped server-side, so the columns should be
close. A large delta means the regex missed a section.

---

## Troubleshooting

**`503 Model bundle not loaded`** — `bundle/bundle_config.json` is missing. Unpack
the zip into `dp_cxr_service/bundle/`.

**`RuntimeError: Head weights incomplete`** — the bundle's `fusion_type` disagrees
with the checkpoint. Re-export §8.2. This deliberately raises rather than warning:
an unloaded head is a randomly initialised head, and it will return
plausible-looking probabilities.

**Grad-CAM returns nothing** — the backbone is frozen, so gradients must be forced
through it (`requires_grad_(True)` inside `enable_grad`). Handled in `xai.py`; if a
particular input still produces no signal, a warning is attached instead of failing
the prediction.

**Slow first request** — the encoders are downloading (~500 MB). Cached afterwards;
the Docker build pre-bakes them.

**pip upgrades torch and breaks torchvision** — `requirements.txt` pins the stack
and holds opacus at 1.5.2 for exactly this reason. Opacus 1.6+ requires torch ≥ 2.6
and will drag the whole stack with it.
