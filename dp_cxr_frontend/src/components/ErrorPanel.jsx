export default function ErrorPanel({ error, onRetry }) {
  return (
    <div className="card">
      <h2>Request failed</h2>
      <div className="err">
        {error.status ? `HTTP ${error.status}\n\n` : ''}
        {error.message}
      </div>
      {error.hint && <div className="warn">{error.hint}</div>}
      <div className="row">
        <button className="link" onClick={onRetry}>
          Try again
        </button>
      </div>
      <p className="note">
        If this keeps happening, run{' '}
        <code>python dp_cxr_service/diagnose.py</code> in the activated
        environment. It walks the same prediction path stage by stage and stops
        at the first failure with a full traceback.
      </p>
    </div>
  )
}
