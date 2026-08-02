# DP-CXR React frontend

A Vite + React client for the prediction service in `../dp_cxr_service`.

**This talks to the backend — it does not replace it.** Two things must be
running: the Python service on port 8000, and this app on port 5173.

---

## One-time setup

### 1. Install Node

You need Node to run any React app. Download the **LTS** installer from
<https://nodejs.org> and run it. Then open a **new** Terminal window and check:

```bash
node --version     # v20 or v22
npm --version
```

### 2. Install the app's packages

```bash
cd "/Users/hasithaattanayake/Documents/Msc/impl/dp_cxr_frontend"
npm install
```

About 200 MB, roughly a minute. Only needed once.

---

## Running it

You need **two Terminal windows**, both left open.

**Window 1 — the backend:**

```bash
cd "/Users/hasithaattanayake/Documents/Msc/impl"
source .venv/bin/activate
uvicorn dp_cxr_service.app:app --port 8000
```

**Window 2 — this app:**

```bash
cd "/Users/hasithaattanayake/Documents/Msc/impl/dp_cxr_frontend"
npm run dev
```

Your browser opens at <http://localhost:5173>. The header shows a green dot and
the model name once it reaches the backend.

---

## Why the requests work here

`vite.config.js` proxies `/health`, `/predict`, `/predict-json` and
`/model-card` to `127.0.0.1:8000`. The browser therefore sees every request as
*same-origin*, so the browser's cross-origin rules never apply. Calling
`http://localhost:8000` directly from a page served on `:5173` is a cross-origin
request, and that is one of the common causes of a request failing for reasons
that have nothing to do with the model.

If you change the backend port, change it in `vite.config.js` too.

---

## When a request fails

`src/api.js` distinguishes four failure modes instead of reporting them all as
"request failed", and the error panel tells you which one you hit:

| What you see | What it means | What to do |
|---|---|---|
| *Cannot reach the service* | Nothing is listening on 8000 | Window 1 is closed or crashed — restart it |
| **HTTP 503** | Service is up, no model loaded | `python dp_cxr_service/preflight.py` |
| **HTTP 500** | The model raised an exception | Read the traceback in Window 1; run `python dp_cxr_service/diagnose.py` |
| **HTTP 400** | Request rejected before the model | Supply an image or some text |

The single most useful habit: when the browser shows an error, **look at
Window 1**. The backend prints the full Python traceback there, and the last
frame is the actual cause.

---

## Layout

```
src/
  main.jsx                 entry point
  App.jsx                  page composition and request state
  api.js                   fetch calls, and the error classification above
  styles.css               all styling, no framework
  components/
    ImageDrop.jsx          drag-and-drop upload with preview
    ReportInput.jsx        report textarea and sample text
    Findings.jsx           probability bars with threshold markers
    Explanations.jsx       Grad-CAM, token influence, modality shift
    ModelMeta.jsx          privacy and calibration table, text audit
    ErrorPanel.jsx         failure display with the matching remedy
```

### The one design decision worth explaining

`Findings.jsx` does not draw the bars on a 0–1 axis. Probabilities from this
model run roughly 0.02–0.35 while the thresholds run 0.09–0.27, so a 0–1 axis
would render every finding as an identical sliver near zero, and a reader would
conclude the model predicts nothing. The bars are scaled to the visible range
and each carries its own threshold marker, because the comparison that decides
the output is probability against *its own* threshold, not against 0.5.

---

## Building for submission

```bash
npm run build      # static files land in dist/
npm run preview    # serve dist/ to check it
```

`dist/` is a plain folder of static files. Note that `npm run preview` does not
apply the dev proxy, so for a built copy you would either serve `dist/` from
FastAPI itself or enable CORS on the backend.

---

## A note on verification

I could not run `npm install` when writing this — the package registry was not
reachable from my environment — so the build is **unverified**. I checked it
statically instead: brackets balance in all nine source files, and all eight
relative imports resolve to exports that genuinely exist.

If `npm run dev` reports a syntax error, send me the message and the file and
line; it will be a quick fix.

There is also a zero-install fallback: the backend serves an equivalent
single-file interface at <http://localhost:8000> with no Node required.

---

**Not a medical device.** Research demonstration for an MSc dissertation.
