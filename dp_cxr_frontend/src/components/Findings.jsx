/**
 * Per-pathology results: a probability bar and an automatic Positive / Negative call.
 *
 * The decision threshold each probability is compared against is intentionally NOT
 * shown in the UI. It is an internal, validation-fitted value that only confuses an
 * end user (a 0.15 "positive" looks wrong without the full story). The backend still
 * applies it and returns `positive`; here we simply surface that verdict and call out
 * the findings that came back positive.
 */
export default function Findings({ data }) {
  const preds = data.predictions || []
  if (!preds.length) return null

  const positives = preds.filter((p) => p.positive)

  // Scale the bars to the visible probability range. These findings are rare, so
  // probabilities run ~0.02-0.35; a 0-1 axis would squash them all into slivers.
  const span = Math.max(0.35, ...preds.map((p) => p.probability * 1.25))
  const pct = (v) => Math.min(100, (v / span) * 100)

  return (
    <div className="card">
      <h2>Findings</h2>

      <div className={positives.length ? 'posline' : 'posline none'}>
        {positives.length
          ? `Positive: ${positives.map((p) => p.label).join(', ')}`
          : 'No positive findings'}
      </div>

      {preds.map((p) => (
        <div className="finding" key={p.label}>
          <div className="fhead">
            <span className="fname">{p.label}</span>
            <span>
              <span className="fprob">{p.probability.toFixed(3)}</span>
              <span className={p.positive ? 'verdict yes' : 'verdict no'}>
                {p.positive ? 'Positive' : 'Negative'}
              </span>
            </span>
          </div>
          <div className="track">
            <div
              className={p.positive ? 'fill yes' : 'fill'}
              style={{ width: `${pct(p.probability)}%` }}
            />
          </div>
        </div>
      ))}

      <div className="legend">
        <span>modalities used: {(data.modalities_used || []).join(' + ') || 'none'}</span>
        {data.inference_ms != null && <span>{data.inference_ms} ms</span>}
      </div>
      <p className="note">
        Each finding is flagged Positive or Negative automatically. Probabilities are
        low because these conditions are uncommon (1.9–12.4 % of studies), so the model
        is calibrated to their true prevalence rather than to 50 %.
      </p>
    </div>
  )
}
