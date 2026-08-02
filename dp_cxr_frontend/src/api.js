/**
 * Client for the DP-CXR service.
 *
 * The error handling here is deliberate rather than defensive boilerplate. The
 * service can fail in four quite different ways and they need different fixes:
 *
 *   1. not running            -> fetch itself rejects, no HTTP status
 *   2. running, no bundle     -> 503 with a JSON detail
 *   3. running, model threw   -> 500, sometimes JSON, sometimes an HTML page
 *   4. bad request            -> 400 with a JSON detail
 *
 * Collapsing all of these into "request failed" is what makes the problem hard
 * to locate, so each one is reported as itself.
 */

export class ApiError extends Error {
  constructor(message, { status = null, hint = null, body = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.hint = hint
    this.body = body
  }
}

const OFFLINE_HINT =
  'The browser could not reach the service at all. Check that the terminal ' +
  'running "uvicorn dp_cxr_service.app:app --port 8000" is still open and has ' +
  'not exited.'

async function readBody(res) {
  // Read as text first. Calling res.json() on an HTML error page throws a
  // parser error that replaces the server's actual message.
  const raw = await res.text()
  try {
    return { json: JSON.parse(raw), raw }
  } catch {
    return { json: null, raw }
  }
}

function detailToString(detail) {
  if (detail == null) return null
  if (typeof detail === 'string') return detail
  // FastAPI validation errors arrive as an array of objects
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).join('.')}: ${d.msg || JSON.stringify(d)}`)
      .join('\n')
  }
  return JSON.stringify(detail, null, 2)
}

async function request(path, init) {
  let res
  try {
    res = await fetch(path, init)
  } catch (e) {
    throw new ApiError(`Cannot reach the service (${e.message})`, {
      hint: OFFLINE_HINT,
    })
  }

  const { json, raw } = await readBody(res)

  if (!res.ok) {
    const detail = json ? detailToString(json.detail) : null
    let hint = null
    if (res.status === 503) {
      hint =
        'The service started but no model bundle is loaded. Run ' +
        '"python dp_cxr_service/preflight.py" to check the bundle folder.'
    } else if (res.status === 500) {
      hint =
        'The model raised an exception. The full traceback is printed in the ' +
        'terminal running uvicorn — that is where the real cause is. ' +
        '"python dp_cxr_service/diagnose.py" reproduces it step by step.'
    } else if (res.status === 400) {
      hint = 'The request was rejected before reaching the model.'
    }
    throw new ApiError(detail || raw.slice(0, 600) || `HTTP ${res.status}`, {
      status: res.status,
      hint,
      body: raw,
    })
  }

  if (!json) {
    throw new ApiError('The service returned a response that was not JSON.', {
      status: res.status,
      body: raw.slice(0, 600),
    })
  }
  return json
}

export const getHealth = () => request('/health')

export const getModelCard = () => request('/model-card')

export function predict({ image, reportText, gradcam = true, tokens = true }) {
  const fd = new FormData()
  if (image) fd.append('image', image)
  if (reportText && reportText.trim()) fd.append('report_text', reportText)
  fd.append('include_gradcam', String(gradcam))
  fd.append('include_token_attribution', String(tokens))
  // Content-Type is deliberately not set: the browser must add the multipart
  // boundary itself, and setting it manually produces an unparseable body.
  return request('/predict', { method: 'POST', body: fd })
}
