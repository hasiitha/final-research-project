const f = (v, n = 4) => (v == null ? '—' : Number(v).toFixed(n))

export function TextAudit({ data }) {
  const t = data.text_processing
  if (!t) return null
  return (
    <div className="card">
      <h2>Text the model actually saw</h2>
      <div className="mono">{t.text || 'nothing'}</div>
      <table className="meta" style={{ marginTop: 11 }}>
        <tbody>
          <tr>
            <td>Recognised section headers</td>
            <td>{t.had_sections ? 'yes' : 'no'}</td>
          </tr>
          <tr>
            <td>Characters submitted → kept</td>
            <td>
              {t.original_chars} → {t.kept_chars}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="note">
        The leakage control from the thesis, enforced at prediction time:
        Findings and Impression are removed, because the training labels were
        derived from them.
      </p>
    </div>
  )
}

export default function ModelMeta({ data }) {
  const md = data.model_metadata
  if (!md) return null
  const pv = md.privacy || {}
  const cal = md.calibration || {}
  return (
    <div className="card">
      <h2>Model and privacy</h2>
      <table className="meta">
        <tbody>
          <tr><td>Model</td><td>{md.run_name}</td></tr>
          <tr><td>Fusion</td><td>{md.fusion_type}</td></tr>
          <tr><td>Mechanism</td><td>{pv.mechanism || '—'}</td></tr>
          <tr><td>Placement</td><td>{pv.placement || '—'}</td></tr>
          <tr><td>Achieved ε</td><td>{f(pv.epsilon, 3)}</td></tr>
          <tr><td>δ</td><td>{pv.delta ?? '—'}</td></tr>
          <tr><td>Noise multiplier σ</td><td>{f(pv.noise_multiplier, 3)}</td></tr>
          <tr><td>Test macro-AUROC</td><td>{f(md.test_macro_auroc)}</td></tr>
          <tr><td>Expected calibration error</td><td>{f(cal.ece)}</td></tr>
        </tbody>
      </table>
      <p className="note">{cal.note}</p>
    </div>
  )
}
