# Application of Differential Privacy for Multimodal Healthcare Systems

A differentially private, multimodal chest-radiograph classifier and the software that serves it. The model predicts five chest findings from a frontal chest X-ray (CXR) and/or its radiology report, and is trained with **DP-SGD** so that a formal privacy guarantee (ε = 8, δ = 10⁻⁵) protects the patients whose studies were used in training.

This repository is the **implementation and deployment** side of the MSc dissertation *"Application of Differential Privacy for Multimodal Healthcare Systems"* (Hasitha Attanayake, IIT 20232040 / UoW W2053394, MSc Advanced Software Engineering, University of Westminster, supervised by Mr. Dinesh Asanka). The dissertation itself is the background study for this code; model training and evaluation were carried out in separate research notebooks (see [Training and evaluation](#training-and-evaluating-the-models)).

> **Not a medical device.** This is a research demonstration for an MSc dissertation. Outputs are not a diagnosis and must not be used to inform patient care.

---

## 1. What this project does

Given a chest radiograph, a report, or both, the system returns a calibrated probability for each of five findings — **Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion** — together with a Positive/Negative call, a Grad-CAM image explanation, token-level text attributions, and full privacy provenance (the achieved ε, the DP mechanism, and per-label performance).

The served model, `dp_f_late_fusion_tuned`, is a **late-fusion** classifier over two **frozen** encoders — a chest-X-ray-pretrained DenseNet-121 (TorchXRayVision) and Bio_ClinicalBERT — with **only the classification head trained under DP-SGD (placement DP-F)**. Freezing the encoders keeps the private parameter count tiny, which is what makes the privacy cost small.

| | Non-private | Private (ε = 8) |
|---|---|---|
| Macro-AUROC | 0.8206 | **0.7917** (−3.5 %) |
| Expected Calibration Error | — | 0.0145 |
| Achieved ε (δ = 10⁻⁵) | — | 7.996 |

Per-label test AUROC (private model): Pleural Effusion 0.846, Edema 0.815, Cardiomegaly 0.799, Consolidation 0.760, Atelectasis 0.738.

---

## 2. Repository layout

```
impl/
├── dp_cxr_service/            # FastAPI backend that serves the exported model
│   ├── app.py                 #   HTTP routes (/health, /model-card, /predict, /predict-json, /)
│   ├── predictor.py           #   DPCXRPredictor — the prediction engine (importable as a library)
│   ├── model.py               #   Architecture + bundle loading (mirrors the training notebook)
│   ├── preprocessing.py       #   Pre-diagnostic text rule + xrv image normalisation
│   ├── xai.py                 #   Grad-CAM, token occlusion, modality-contribution
│   ├── preflight.py           #   Verifies the bundle before startup
│   ├── diagnose.py            #   Step-by-step reproduction of a prediction (for debugging)
│   ├── test_client.py         #   Exercises all request shapes (incl. a leakage check)
│   ├── static/index.html      #   Built-in single-page UI (served at http://localhost:8000/)
│   ├── bundle/                #   >>> the exported model bundle goes here <<<  (see §5)
│   ├── requirements.txt       #   Inference dependencies (version floors)
│   ├── requirements-locked.txt#   Exact versions known to resolve on Python 3.12
│   ├── setup.sh / run.sh      #   One-command setup / start
│   ├── Dockerfile             #   CPU-only container
│   ├── README.md              #   Service-level documentation (endpoints, response shape)
│   ├── RUNBOOK.md             #   Operational runbook
│   └── SETUP_FROM_SCRATCH.md  #   Detailed first-time setup walkthrough
│
├── dp_cxr_frontend/           # React + Vite single-page client (optional, richer than the built-in UI)
│   ├── src/                   #   App.jsx, api.js, components/ (Findings, Explanations, …)
│   ├── package.json           #   React 18 + Vite 5
│   └── vite.config.js         #   Dev server on :5173, proxies API calls to :8000
│
├── notebooks/                 # Training, evaluation and data-preparation notebooks (Colab / Redivis)
├── test_evidence/             # Saved output of the test runs — see its README for the commands
└── samples/                   # Synthetic report text; instructions for obtaining a radiograph
```

### Where each submission requirement lives

| Requirement | Location |
|---|---|
| Complete source code | `dp_cxr_service/`, `dp_cxr_frontend/src/` |
| Model training and evaluation scripts | `notebooks/` |
| Data preprocessing and feature engineering | `notebooks/` (cohort, split, text rule, embedding cache); `dp_cxr_service/preprocessing.py` at inference time |
| Front-end | `dp_cxr_frontend/`, and the no-build UI at `dp_cxr_service/static/index.html` |
| Back-end | `dp_cxr_service/app.py`, `predictor.py`, `model.py`, `xai.py` |
| Database components | Not applicable — the service is stateless and uses no database |
| Configuration files | `dp_cxr_service/bundle/*.json`, `Dockerfile`, `vite.config.js` |
| Dependency / package files | `requirements.txt`, `requirements-locked.txt`, `package.json`, `package-lock.json` |
| Test files and test evidence | `dp_cxr_service/test_client.py`, `diagnose.py`, `preflight.py`; output in `test_evidence/` |
| Trained model files | `dp_cxr_service/bundle/` — 80 KB, committed |
| Sample input / dataset instructions | `samples/`, and §9 below |
| API integration instructions | §7 below, and `dp_cxr_service/RUNBOOK.md` |
| Deployment configuration | `Dockerfile`, `setup.sh`, `run.sh` |
| README | this file, plus one per component |

---

## 3. Software and hardware requirements

**Software**

- **Python 3.9–3.12** (3.12 recommended). Python 3.13 is not yet supported by the PyTorch wheels.
- **Node.js 18+** and npm — *only* if you want to run the standalone React frontend. The built-in UI needs no Node.
- **Git** (to obtain the code) and, optionally, **Docker** (to run the backend in a container).
- OS: macOS, Linux, or Windows.

**Hardware**

- Runs **CPU-only** — no GPU required (the Docker image installs CPU PyTorch). A GPU is optional and only speeds up inference.
- **≈ 4 GB RAM** free is comfortable; the model head is tiny but PyTorch and the frozen encoders occupy memory.
- **≈ 3 GB free disk**: ~2 GB for the Python/PyTorch dependencies plus ~500 MB for the frozen encoder weights that download on first use.

---

## 4. Languages, libraries and frameworks

| Area | Technology |
|---|---|
| Backend language | Python 3 |
| Web framework | FastAPI + Uvicorn (ASGI server) |
| Deep learning | PyTorch, TorchVision |
| Image encoder | TorchXRayVision (`densenet121-res224-all`, frozen) |
| Text encoder | Hugging Face Transformers + Tokenizers (`emilyalsentzer/Bio_ClinicalBERT`, frozen) |
| Explainability / imaging | NumPy, Pillow, Matplotlib |
| Differential privacy | Opacus (**training only** — the exported head has the Opacus wrapper stripped, so inference needs no DP library) |
| Frontend | JavaScript, React 18, Vite 5 |

---

## 5. Configuration — the model bundle (required)

The backend serves a pre-exported **model bundle**, committed to this repository at `dp_cxr_service/bundle/`. It is 80 KB in total because only the classification head was trained; both encoders are public checkpoints that download on first use. Nothing needs to be fetched before the service can load.

**Obtain and unpack it:**

1. Download `dp_cxr_deployment_bundle.zip` — exported by §8.2 of the training notebooks and stored in Google Drive at `MyDrive/dp_cxr_mv/`.
2. Unzip it so these files sit **directly** inside `dp_cxr_service/bundle/` (a nested `bundle/deployment_bundle/…` folder is the common mistake):

```bash
unzip dp_cxr_deployment_bundle.zip -d dp_cxr_service/bundle/
```

The bundle contains (all small — only the head is project-specific):

| File | Role |
|---|---|
| `head_state_dict.pth` | The only trained weights (~38 KB); encoders are frozen and public |
| `bundle_config.json` | Architecture, labels, tokenizer, image transform |
| `thresholds.json` | Validation-tuned per-label decision thresholds |
| `text_rule.json` | The pre-diagnostic-section regex, so inference matches training |
| `model_card.json` | ε, δ, placement, metrics, cohort, limitations |
| `selection_ranking.csv` | Why this run was chosen over the others |

**Environment variables** (optional; sensible defaults):

| Variable | Default | Purpose |
|---|---|---|
| `DP_CXR_BUNDLE` | `dp_cxr_service/bundle` | Path to the bundle directory |
| `DP_CXR_CACHE` | HF default | Where the frozen encoder weights are cached |

**Ports:** backend `8000`, React dev server `5173`.

`python dp_cxr_service/preflight.py` validates the bundle (file presence and checksums against `MANIFEST.json`) and prints a clear message if anything is missing.

---

## 6. Installation and dependency installation

Clone or open the project, then set up the backend and (optionally) the frontend.

### Backend — easiest path (one command)

From the project root (`impl/`):

```bash
bash dp_cxr_service/setup.sh
```

This checks your Python version, verifies the bundle is present, creates an isolated `.venv`, installs all dependencies (falling back to the locked versions if needed), verifies the imports, and starts the service on `http://127.0.0.1:8000`.

### Backend — manual path

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r dp_cxr_service/requirements.txt
# if resolution misbehaves, use the exact set instead:
# pip install -r dp_cxr_service/requirements-locked.txt
```

### Frontend (optional)

```bash
cd dp_cxr_frontend
npm install
```

---

## 7. Running the system

### Backend

Any one of:

```bash
# a) one-command setup + start
bash dp_cxr_service/setup.sh

# b) if already installed
uvicorn dp_cxr_service.app:app --reload --port 8000     # run from the project root

# c) Docker (CPU-only image)
docker build -t dp-cxr dp_cxr_service/
docker run -p 8000:8000 dp-cxr
```

Then open **http://localhost:8000/** for the built-in UI, or **http://localhost:8000/docs** for the interactive Swagger API (upload a radiograph and paste a report straight from the browser — a clean demo with no client needed).

> The **first** prediction downloads the frozen DenseNet-121 and Bio_ClinicalBERT encoders (~500 MB) and can take a minute. This happens once; they are cached afterwards (and pre-baked into the Docker image).

### Frontend (optional, richer UI)

With the backend already running on port 8000:

```bash
cd dp_cxr_frontend
npm run dev        # opens http://localhost:5173, proxies API calls to :8000
```

For a production build: `npm run build` (output in `dist/`), served by any static host.

### Using the API directly

```bash
# both modalities
curl -X POST http://localhost:8000/predict \
  -F "image=@cxr.jpg" \
  -F "report_text=CLINICAL HISTORY: 64M with shortness of breath. TECHNIQUE: PA and lateral."

# image only  /  report only
curl -X POST http://localhost:8000/predict -F "image=@cxr.jpg"
curl -X POST http://localhost:8000/predict -F "report_text=CLINICAL HISTORY: cough, fever."
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, which bundle is loaded, ε, device |
| `GET` | `/model-card` | Privacy budget, per-label performance, limitations |
| `POST` | `/predict` | Multipart — image file and/or `report_text` |
| `POST` | `/predict-json` | JSON — base64 image and/or `report_text` |
| `GET` | `/` , `/docs` | Built-in UI, interactive API docs |

Two behaviours worth knowing: report text is **stripped to its pre-diagnostic sections** server-side (Findings and Impression are removed to prevent the label leakage the study was designed to avoid), and the Positive/Negative call uses the **validation-tuned thresholds shipped in the bundle**, not a flat 0.5 (these findings are rare, so 0.5 would report almost nothing positive).

---

## 8. Testing

```bash
python -m dp_cxr_service.test_client --direct              # in-process, no server needed
python -m dp_cxr_service.test_client                       # against a running server
python -m dp_cxr_service.test_client --direct --image cxr.jpg
```

Without `--image` a synthetic radiograph is used, which checks the plumbing rather than the accuracy. The `--direct` run also prints a **leakage check** (the same report submitted in full vs history-only) — the two columns should be close, confirming the section-stripping works.

---

## 9. Dataset preparation

**Not required to run the service.** The deployed application takes an uploaded image and/or pasted report at request time, so no dataset needs to be downloaded to use it.

Dataset preparation is only relevant to **re-training or reproducing** the model, which is done in the research notebooks, not in this repository. For completeness, the training data is **CheXpert Plus** (Stanford AIMI), accessed **in-platform via Redivis** so that raw data never leaves its governed store. Preparation there consists of: filtering to frontal studies; a patient-level 70/15/15 split; the U-Zero uncertain-label policy; and restricting report text to its pre-diagnostic sections (the same `text_rule.json` regex the service applies at inference). The training cohort was ~19,300 studies. The dataset is used under its research licence; only de-identified data was processed.

---

## 10. Training and evaluating the models

**Training and evaluation are not performed in this repository.** They were carried out in the dissertation's Google Colab / Redivis notebooks (GPU: NVIDIA A100), which:

- build the cohort and the leakage-free text pipeline,
- train the frozen-encoder late/early-fusion heads, non-private and under DP-SGD (Opacus, RDP accountant, ε-sweep 1–8),
- evaluate utility (AUROC, AUPRC), calibration (ECE, Brier), a 0–80 % missing-modality stress test, and Grad-CAM / token explanations,
- rank all DP runs by a composite score and **export the winner as `dp_cxr_deployment_bundle.zip`** (notebook §8.1–8.2) — the bundle this service loads.

Every experiment's configuration, metrics and model weights are saved in `reproducibility_package.zip` (on Google Drive alongside the notebook outputs). This repository consumes the exported bundle; it does not retrain.

---

## 11. Authentication and user accounts

**Not applicable.** The service has no login, user accounts, roles, or authentication of any kind — it is a single-user local research demonstration, so there are no default credentials or test accounts. If it were ever exposed beyond localhost, an authentication layer would need to be added first.

---

## 12. External services and API keys

**No API keys are required to run the service.** Specifically:

- **Hugging Face** (Bio_ClinicalBERT) and **TorchXRayVision** (DenseNet-121) weights download from public endpoints on first use — no token needed. Setting `HF_TOKEN` is optional and only raises download rate limits.
- **Redivis** — an API token was needed only to access CheXpert Plus **during training/data extraction** in the notebooks. It is **not** needed to run this service.
- No database, cloud service, or paid third-party API is used at inference time; everything runs locally.

---

## 13. Known limitations

- **Not a medical device**; not validated for clinical use.
- **Single-institution training** (Stanford CheXpert Plus) — performance on other sites or scanners is unknown.
- **Text-dominant, data-bounded image branch.** The labels are derived from the reports (CheXbert), so the model's discriminative signal leans on text; even a state-of-the-art CXR encoder is limited on this cohort by label noise (the U-Zero policy makes uncertain findings such as Atelectasis hard to learn).
- **Rare findings are unreliable.** Low-prevalence conditions have wide, noisy estimates; DP-SGD degrades rare-class performance disproportionately.
- **Unimodal requests are out-of-distribution.** The model was trained with both modalities; image-only or text-only inputs work and degrade gracefully, but with a characterised drop (see the missing-modality stress test) — every unimodal response carries a warning.
- **Differential privacy has a cost**: ≈3.5 AUROC points at ε = 8, and a measurable calibration cost, relative to the non-private model.

---

## 14. Background and further reading

The full background study, methodology, results and critical evaluation are in the dissertation *"Application of Differential Privacy for Multimodal Healthcare Systems."* For service-specific details (full response schema, the three inference behaviours, troubleshooting), see `dp_cxr_service/README.md`, `dp_cxr_service/RUNBOOK.md`, and `dp_cxr_service/SETUP_FROM_SCRATCH.md`.

---

*Academic project — Informatics Institute of Technology, in collaboration with the University of Westminster, 2026.*
