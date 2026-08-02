const SAMPLE = `CLINICAL HISTORY: 58-year-old male with acute shortness of breath and fever.
INDICATION: Rule out infection.
TECHNIQUE: Frontal AP portable chest radiograph.
COMPARISON: Prior study dated 2 January.`

export default function ReportInput({ value, onChange }) {
  return (
    <div className="card">
      <h2>Report text</h2>
      <label htmlFor="report">Pre-diagnostic sections only</label>
      <textarea
        id="report"
        value={value}
        placeholder={'CLINICAL HISTORY: …\nINDICATION: …\nTECHNIQUE: …'}
        onChange={(e) => onChange(e.target.value)}
      />
      <div className="row">
        <button className="link" onClick={() => onChange(SAMPLE)}>
          Insert a sample report
        </button>
        <button className="link" onClick={() => onChange('')}>
          Clear
        </button>
      </div>
      <p className="note">
        Paste a full report if you like — Findings and Impression are stripped
        server-side, exactly as during training. Whatever survives that rule is
        shown back to you after prediction.
      </p>
    </div>
  )
}
