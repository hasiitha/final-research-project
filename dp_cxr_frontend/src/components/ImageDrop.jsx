import { useCallback, useRef, useState } from 'react'

export default function ImageDrop({ file, onChange }) {
  const inputRef = useRef(null)
  const [preview, setPreview] = useState(null)
  const [over, setOver] = useState(false)

  const accept = useCallback(
    (f) => {
      if (!f) return
      if (!f.type.startsWith('image/')) {
        alert('That file is not an image.')
        return
      }
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(f)
      onChange(f)
    },
    [onChange],
  )

  const clear = (e) => {
    e.stopPropagation()
    setPreview(null)
    onChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="card">
      <h2>Frontal radiograph</h2>
      {preview ? (
        <div className="thumbWrap">
          <img src={preview} alt="selected radiograph" />
          <button onClick={clear}>Remove</button>
        </div>
      ) : (
        <div
          className={over ? 'drop over' : 'drop'}
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            accept(e.dataTransfer.files?.[0])
          }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#12798F" strokeWidth="1.6">
            <path d="M12 16V4m0 0L8 8m4-4 4 4" />
            <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          <p>
            <strong>Click to choose</strong> or drag an image here
          </p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => accept(e.target.files?.[0])}
      />
      {file && <p className="note">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
    </div>
  )
}
