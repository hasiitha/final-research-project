export default function Explanations({ data }) {
  const ex = data.explanations || {}
  const toks = ex.token_attribution?.tokens || []
  if (!ex.gradcam && !toks.length && !ex.modality_contribution) return null

  const maxInf = Math.max(...toks.map((t) => t.influence), 1)

  return (
    <div className="card">
      <h2>Explanation</h2>
      <div className="grid2">
        {ex.gradcam && (
          <div>
            <label>Grad-CAM — {ex.gradcam.label}</label>
            <img className="cam" src={ex.gradcam.overlay_png} alt="Grad-CAM overlay" />
            <p className="note">{ex.gradcam.note}</p>
          </div>
        )}
        {toks.length > 0 && (
          <div>
            <label>Token influence</label>
            <div>
              {toks.map((t, i) => (
                <span
                  key={`${t.position}-${i}`}
                  className="tok"
                  title={`influence ${t.influence.toExponential(2)}`}
                  style={{
                    background: `rgba(46,196,182,${(0.1 + 0.68 * (t.influence / maxInf)).toFixed(2)})`,
                  }}
                >
                  {t.token}
                </span>
              ))}
            </div>
            <p className="note">{ex.token_attribution.note}</p>
          </div>
        )}
      </div>
      {ex.modality_contribution && (
        <>
          <table className="meta" style={{ marginTop: 14 }}>
            <tbody>
              <tr>
                <td>Shift when text is withheld</td>
                <td>{ex.modality_contribution.image_only_shift.toFixed(4)}</td>
              </tr>
              <tr>
                <td>Shift when image is withheld</td>
                <td>{ex.modality_contribution.text_only_shift.toFixed(4)}</td>
              </tr>
            </tbody>
          </table>
          <p className="note">{ex.modality_contribution.note}</p>
        </>
      )}
    </div>
  )
}
