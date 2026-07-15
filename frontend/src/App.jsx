import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, Loader2, RotateCcw, ShieldCheck, UploadCloud, WifiOff } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Simple state machine:
 *   idle → preview → analyzing → result | error
 *
 * A separate `backendStatus` state tracks the /health poll independently so
 * the upload UI is never blocked by a slow model load.
 */
export default function App() {
  const [status, setStatus] = useState('idle')
  const [imagePreview, setImagePreview] = useState(null)
  const [imageFile, setImageFile] = useState(null)
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking') // 'checking' | 'ok' | 'model_not_loaded' | 'unreachable'
  const inputRef = useRef(null)

  // ── Backend health poll ────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false

    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) })
        if (cancelled) return
        if (res.ok) {
          const data = await res.json()
          setBackendStatus(data.status === 'ok' ? 'ok' : 'model_not_loaded')
        } else {
          setBackendStatus('unreachable')
        }
      } catch {
        if (!cancelled) setBackendStatus('unreachable')
      }
    }

    checkHealth()
    // Re-poll every 15 s while the model may still be loading
    const interval = setInterval(checkHealth, 15_000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  // ── File handling ──────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setResult(null)
    setErrorMsg('')
    setStatus('preview')
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  const analyze = async () => {
    if (!imageFile) return
    setStatus('analyzing')
    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }
      const data = await res.json()
      setResult(data)
      setStatus('result')
      setBackendStatus('ok')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong reaching the model server.')
      setStatus('error')
    }
  }

  const reset = () => {
    setStatus('idle')
    setImagePreview(null)
    setImageFile(null)
    setResult(null)
    setErrorMsg('')
  }

  const isMalignant = result?.label === 'malignant'

  // ── Backend status banner ──────────────────────────────────────────────────
  const BackendBanner = () => {
    if (backendStatus === 'ok') return null

    const banners = {
      checking: {
        icon: <Loader2 size={14} className="animate-spin" />,
        text: 'Connecting to model server…',
        cls: 'bg-slate-100 text-slate-600',
      },
      model_not_loaded: {
        icon: <Loader2 size={14} className="animate-spin" />,
        text: 'Model is loading — predictions will be available shortly.',
        cls: 'bg-amber-50 text-amber-700',
      },
      unreachable: {
        icon: <WifiOff size={14} />,
        text: `Cannot reach the backend at ${API_URL}. Is the server running?`,
        cls: 'bg-red-50 text-red-700',
      },
    }

    const b = banners[backendStatus]
    if (!b) return null

    return (
      <motion.div
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        className={`w-full max-w-md mb-4 flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs font-medium ${b.cls}`}
      >
        {b.icon}
        <span>{b.text}</span>
      </motion.div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col items-center px-4 py-12 sm:py-16">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="text-center mb-6 max-w-lg"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-clinical-teal/10 text-clinical-teal text-xs font-semibold tracking-wide uppercase mb-4">
          <ShieldCheck size={14} />
          Research Prototype
        </div>
        <h1 className="font-heading text-3xl sm:text-4xl font-extrabold tracking-tight text-clinical-text">
          Skin Lesion Classifier
        </h1>
        <p className="mt-3 text-clinical-textMuted text-sm sm:text-base">
          Upload a dermoscopic image to get a benign&nbsp;/&nbsp;malignant assessment from a
          CNN trained for this task. This is a technical demo, not a diagnostic tool.
        </p>
      </motion.header>

      <BackendBanner />

      <div className="w-full max-w-md">
        <AnimatePresence mode="wait">
          {status === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.25 }}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`cursor-pointer rounded-2xl border-2 border-dashed p-12 flex flex-col items-center justify-center text-center transition-colors duration-200 bg-clinical-surface shadow-soft
                ${isDragging ? 'border-clinical-teal bg-clinical-teal/5' : 'border-clinical-border hover:border-clinical-tealLight'}`}
            >
              <UploadCloud size={36} className="text-clinical-teal mb-4" strokeWidth={1.5} />
              <p className="font-medium text-clinical-text">Drag &amp; drop an image here</p>
              <p className="text-sm text-clinical-textMuted mt-1">or click to browse (JPEG/PNG)</p>
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </motion.div>
          )}

          {(status === 'preview' || status === 'analyzing') && (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.25 }}
              className="rounded-2xl bg-clinical-surface shadow-soft overflow-hidden"
            >
              <div className="aspect-square w-full overflow-hidden bg-slate-100">
                <img src={imagePreview} alt="Uploaded lesion" className="w-full h-full object-cover" />
              </div>
              <div className="p-5 flex gap-3">
                <button
                  onClick={reset}
                  disabled={status === 'analyzing'}
                  className="flex-1 rounded-xl border border-clinical-border py-2.5 text-sm font-medium text-clinical-textMuted hover:bg-slate-50 transition-colors disabled:opacity-50"
                >
                  Choose different image
                </button>
                <button
                  onClick={analyze}
                  disabled={status === 'analyzing'}
                  className="flex-1 rounded-xl bg-clinical-teal py-2.5 text-sm font-semibold text-white hover:bg-clinical-teal/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-70"
                >
                  {status === 'analyzing' ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      Analyzing
                    </>
                  ) : (
                    'Analyze image'
                  )}
                </button>
              </div>
            </motion.div>
          )}

          {status === 'result' && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="rounded-2xl bg-clinical-surface shadow-softLg overflow-hidden"
            >
              <div className="aspect-square w-full overflow-hidden bg-slate-100">
                <img src={imagePreview} alt="Analyzed lesion" className="w-full h-full object-cover" />
              </div>

              <div className="p-6">
                <div
                  className={`flex items-center gap-2 rounded-xl px-4 py-3 mb-4
                    ${isMalignant ? 'bg-clinical-malignantBg text-clinical-malignant' : 'bg-clinical-benignBg text-clinical-benign'}`}
                >
                  {isMalignant ? <AlertTriangle size={20} /> : <ShieldCheck size={20} />}
                  <span className="font-heading font-bold text-lg capitalize">{result.label}</span>
                </div>

                <div className="mb-1 flex justify-between text-sm text-clinical-textMuted">
                  <span>Confidence</span>
                  <span className="font-medium text-clinical-text">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden mb-4">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut' }}
                    className={`h-full rounded-full ${isMalignant ? 'bg-clinical-malignant' : 'bg-clinical-benign'}`}
                  />
                </div>

                <div className="text-xs text-clinical-textMuted space-y-1 mb-5">
                  <div className="flex justify-between">
                    <span>Malignant probability</span>
                    <span>{(result.probability_malignant * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Decision threshold</span>
                    <span>{(result.threshold * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference time</span>
                    <span>{result.inference_time_ms} ms</span>
                  </div>
                </div>

                <p className="text-xs text-clinical-textMuted border-t border-clinical-border pt-4 mb-4">
                  This result comes from a research prototype and is not a substitute
                  for professional medical evaluation. Always consult a dermatologist
                  for any concerning skin lesion.
                </p>

                <button
                  onClick={reset}
                  className="w-full rounded-xl border border-clinical-border py-2.5 text-sm font-medium text-clinical-textMuted hover:bg-slate-50 transition-colors flex items-center justify-center gap-2"
                >
                  <RotateCcw size={14} />
                  Analyze another image
                </button>
              </div>
            </motion.div>
          )}

          {status === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.25 }}
              className="rounded-2xl bg-clinical-surface shadow-soft p-6 text-center"
            >
              <AlertTriangle size={32} className="text-clinical-malignant mx-auto mb-3" />
              <p className="font-medium text-clinical-text mb-1">Couldn't complete the analysis</p>
              <p className="text-sm text-clinical-textMuted mb-5">{errorMsg}</p>
              <button
                onClick={reset}
                className="rounded-xl bg-clinical-teal px-5 py-2.5 text-sm font-semibold text-white hover:bg-clinical-teal/90 transition-colors"
              >
                Try again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <footer className="mt-10 text-xs text-clinical-textMuted">
        Backend: {API_URL} &nbsp;·&nbsp; Threshold: {result?.threshold ?? '—'}
      </footer>
    </div>
  )
}
