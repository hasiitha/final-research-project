import { useEffect, useState } from 'react'
import { getHealth, predict, ApiError } from './api.js'
import ImageDrop from './components/ImageDrop.jsx'
import ReportInput from './components/ReportInput.jsx'
import Findings from './components/Findings.jsx'
import Explanations from './components/Explanations.jsx'
import ModelMeta, { TextAudit } from './components/ModelMeta.jsx'
import ErrorPanel from './components/ErrorPanel.jsx'

export default function App() {
  const [health, setHealth] = useState({ state: 'checking' })
  const [image, setImage] = useState(null)
  const [report, setReport] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getHealth()
      .then((h) =>
        setHealth(
          h.bundle_loaded
            ? { state: 'ok', model: h.model, epsilon: h.epsilon }
            : { state: 'nobundle' },
        ),
      )
      .catch(() => setHealth({ state: 'down' }))
  }, [])

  async function run() {
    if (!image && !report.trim()) {
      alert('Supply a radiograph, report text, or both.')
      return
    }
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await predict({ image, reportText: report }))
    } catch (e) {
      setError(
        e instanceof ApiError ? e : new ApiError(e.message || String(e)),
      )
    } finally {
      setBusy(false)
    }
  }

  const badge = {
    checking: ['pill', 'dot', 'checking…'],
    ok: ['pill', 'dot live', health.model],
    nobundle: ['pill bad', 'dot down', 'no model bundle'],
    down: ['pill bad', 'dot down', 'service unreachable'],
  }[health.state]

  return (
    <>
      <header>
        <h1>DP-CXR</h1>
        <span className="sub">
          Differentially private multimodal chest radiograph prediction
        </span>
        <span className="spacer" />
        <span className={badge[0]}>
          <span className={badge[1]} />
          {badge[2]}
        </span>
        {health.epsilon != null && (
          <span className="pill">ε = {Number(health.epsilon).toFixed(2)}</span>
        )}
        <a className="pill" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
          API docs
        </a>
      </header>

      <main>
        <div>
          <ImageDrop file={image} onChange={setImage} />
          <ReportInput value={report} onChange={setReport} />
          <button className="go" onClick={run} disabled={busy}>
            {busy ? (
              <>
                <span className="spin" />
                Predicting…
              </>
            ) : (
              'Predict'
            )}
          </button>
          <p className="note" style={{ textAlign: 'center' }}>
            Supply an image, a report, or both.
          </p>
        </div>

        <div>
          {health.state === 'down' && !error && (
            <div className="card">
              <h2>Service not reachable</h2>
              <div className="warn">
                Nothing is answering on port 8000. Start the backend in a
                separate terminal:
              </div>
              <div className="mono">
                cd "/Users/hasithaattanayake/Documents/Msc/impl"
                <br />
                source .venv/bin/activate
                <br />
                uvicorn dp_cxr_service.app:app --port 8000
              </div>
            </div>
          )}

          {error && <ErrorPanel error={error} onRetry={run} />}

          {busy && (
            <div className="card">
              <div className="empty">
                <p>
                  Running the model…
                  <br />
                  <span style={{ fontSize: 12 }}>
                    The first request downloads the encoders and can take a minute.
                  </span>
                </p>
              </div>
            </div>
          )}

          {result && (
            <>
              {result.warnings?.length > 0 && (
                <div className="card">
                  <h2>Notes on this request</h2>
                  {result.warnings.map((w, i) => (
                    <div className="warn" key={i}>
                      {w}
                    </div>
                  ))}
                </div>
              )}
              <Findings data={result} />
              <Explanations data={result} />
              <TextAudit data={result} />
              <ModelMeta data={result} />
            </>
          )}

          {!result && !busy && !error && health.state !== 'down' && (
            <div className="card">
              <div className="empty">
                <p>Results will appear here.</p>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer>
        <div className="disclaimer">
          <strong>Not a medical device.</strong> Research demonstration for an
          MSc dissertation. Trained on one institution's data with labels derived
          from report text. Outputs are not a diagnosis and must not be used to
          inform patient care.
        </div>
      </footer>
    </>
  )
}
