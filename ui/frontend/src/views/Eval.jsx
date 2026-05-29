import { useState, useRef, useEffect } from 'react'

const DOMAINS = ['scheduling', 'dependencies', 'inventory', 'fsm', 'spatial', 'temporal']

const BACKEND_DOWN = 'Backend not reachable — start it first:\n  cd ui/backend && uvicorn main:app --port 8000'

async function parseErrorDetail(res) {
  try {
    const body = await res.json()
    return body.detail || `HTTP ${res.status}`
  } catch {
    return BACKEND_DOWN
  }
}

function AccuracyChart({ byDifficulty }) {
  const W = 380
  const H = 180
  const mt = 30, mb = 36, ml = 38, mr = 16
  const cW = W - ml - mr
  const cH = H - mt - mb

  const difficulties = [1, 2, 3, 4, 5]
  const barW = (cW / difficulties.length) * 0.55
  const slot = cW / difficulties.length

  const gridLines = [0, 25, 50, 75, 100]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md font-mono">
      <g transform={`translate(${ml},${mt})`}>
        {/* Grid lines + Y labels */}
        {gridLines.map((pct) => {
          const y = cH - (pct / 100) * cH
          return (
            <g key={pct}>
              <line x1={0} y1={y} x2={cW} y2={y} stroke="#2a2a3a" strokeWidth={1} />
              <text x={-6} y={y + 4} textAnchor="end" fill="#6b7280" fontSize={9}>
                {pct}%
              </text>
            </g>
          )
        })}

        {/* Bars */}
        {difficulties.map((d, i) => {
          const data = byDifficulty?.[String(d)]
          const accuracy = data && data.total > 0 ? data.correct / data.total : null
          const barH = accuracy !== null ? accuracy * cH : 0
          const x = i * slot + (slot - barW) / 2
          const y = cH - barH

          return (
            <g key={d}>
              {accuracy !== null ? (
                <>
                  <rect x={x} y={y} width={barW} height={barH} fill="#a3e635" rx={2} />
                  <text x={x + barW / 2} y={y - 5} textAnchor="middle" fill="#a3e635" fontSize={9}>
                    {(accuracy * 100).toFixed(0)}%
                  </text>
                </>
              ) : (
                <rect x={x} y={0} width={barW} height={cH} fill="#1a1a24" rx={2} />
              )}
              <text x={x + barW / 2} y={cH + 14} textAnchor="middle" fill="#9ca3af" fontSize={10}>
                D{d}
              </text>
              {data && (
                <text x={x + barW / 2} y={cH + 25} textAnchor="middle" fill="#6b7280" fontSize={8}>
                  {data.correct}/{data.total}
                </text>
              )}
            </g>
          )
        })}

        {/* X-axis */}
        <line x1={0} y1={cH} x2={cW} y2={cH} stroke="#2a2a3a" strokeWidth={1} />
      </g>
    </svg>
  )
}

function ResultRow({ item, index }) {
  return (
    <div
      className={`flex items-center gap-3 px-3 py-1.5 text-xs font-mono border-b border-[#1a1a24] animate-fade-in ${
        item.correct ? 'bg-[#1a2e10]/30' : 'bg-[#2e1010]/20'
      }`}
    >
      <span className="text-[#6b7280] w-6 text-right shrink-0">{index + 1}</span>
      <span className={`w-4 shrink-0 ${item.correct ? 'text-[#a3e635]' : 'text-[#f87171]'}`}>
        {item.correct ? '✓' : '✗'}
      </span>
      <span className="text-[#9ca3af] shrink-0">D{item.difficulty}</span>
      <div className="flex-1 min-w-0">
        <div className="w-full bg-[#2a2a3a] h-1.5 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#a3e635] rounded-full transition-all"
            style={{ width: `${(item.running_accuracy * 100).toFixed(1)}%` }}
          />
        </div>
      </div>
      <span className="text-[#e5e7eb] shrink-0 w-10 text-right">
        {(item.running_accuracy * 100).toFixed(1)}%
      </span>
    </div>
  )
}

export default function Eval() {
  const [domain, setDomain] = useState('scheduling')
  const [difficulty, setDifficulty] = useState('mixed')
  const [size, setSize] = useState(50)
  const [apiKey, setApiKey] = useState('')

  const [running, setRunning] = useState(false)
  const [results, setResults] = useState([])
  const [finalResult, setFinalResult] = useState(null)
  const [error, setError] = useState(null)

  const feedRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight
    }
  }, [results])

  async function runEval() {
    if (!apiKey.trim()) {
      setError('API key required')
      return
    }
    setRunning(true)
    setResults([])
    setFinalResult(null)
    setError(null)

    abortRef.current = new AbortController()

    try {
      const res = await fetch('/api/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain,
          difficulty: difficulty === 'mixed' ? null : parseInt(difficulty),
          size,
          api_key: apiKey,
        }),
        signal: abortRef.current.signal,
      })

      if (!res.ok) {
        throw new Error(await parseErrorDetail(res))
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()
        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.done) {
                setFinalResult(data)
              } else {
                setResults((prev) => [...prev, data])
              }
            } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        setError(e.message === 'Failed to fetch' ? BACKEND_DOWN : e.message)
      }
    } finally {
      setRunning(false)
    }
  }

  function stopEval() {
    abortRef.current?.abort()
    setRunning(false)
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-6">
      <h2 className="text-sm font-mono text-[#9ca3af] uppercase tracking-wider mb-5">
        Model Evaluation
      </h2>

      {/* Config */}
      <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] p-4 mb-4">
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Domain</label>
            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              disabled={running}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] rounded px-2 py-1.5 font-mono focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            >
              {DOMAINS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Difficulty</label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={running}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] rounded px-2 py-1.5 font-mono focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            >
              <option value="mixed">Mixed</option>
              {[1, 2, 3, 4, 5].map((d) => (
                <option key={d} value={d}>Level {d}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mb-3">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs text-[#6b7280] font-mono">Tasks</label>
            <span className="text-xs font-mono text-[#a3e635]">{size}</span>
          </div>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
            disabled={running}
            className="w-full disabled:opacity-40"
          />
          <div className="flex justify-between text-[10px] font-mono text-[#6b7280] mt-0.5">
            <span>10</span><span>500</span>
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-xs text-[#6b7280] font-mono mb-1">Anthropic API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-ant-…"
            disabled={running}
            className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] font-mono rounded px-3 py-1.5 focus:outline-none focus:border-[#a3e635] placeholder-[#3a3a4a] disabled:opacity-40"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={runEval}
            disabled={running || !apiKey.trim()}
            className="flex-1 py-2 bg-[#a3e635] text-black text-sm font-mono font-semibold rounded hover:bg-[#b5f23f] disabled:opacity-40 transition-colors"
          >
            {running ? `Running… (${results.length}/${size})` : 'Run Eval'}
          </button>
          {running && (
            <button
              onClick={stopEval}
              className="px-4 py-2 border border-[#7c2e2e] text-[#f87171] text-sm font-mono rounded hover:bg-[#2e1010] transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 px-3 py-2 bg-red-900/20 border border-red-800/50 text-red-400 text-sm font-mono rounded">
          {error}
        </div>
      )}

      {/* Live feed */}
      {results.length > 0 && (
        <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] overflow-hidden mb-4">
          <div className="px-3 py-2 border-b border-[#2a2a3a] flex items-center justify-between">
            <span className="text-xs font-mono text-[#9ca3af]">Results feed</span>
            {results.length > 0 && (
              <span className="text-xs font-mono text-[#6b7280]">
                {results.filter((r) => r.correct).length}/{results.length} correct
              </span>
            )}
          </div>
          <div
            ref={feedRef}
            className="max-h-72 overflow-y-auto"
          >
            {results.map((item, i) => (
              <ResultRow key={i} item={item} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Final result */}
      {finalResult && (
        <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] p-4 animate-fade-in">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-2xl font-mono font-bold text-[#a3e635]">
              {(finalResult.final_accuracy * 100).toFixed(1)}%
            </span>
            <span className="text-sm font-mono text-[#9ca3af]">overall accuracy</span>
            <span className="ml-auto text-xs font-mono text-[#6b7280]">
              {results.filter((r) => r.correct).length}/{results.length} correct
            </span>
          </div>

          <AccuracyChart byDifficulty={finalResult.by_difficulty} />

          <div className="mt-3 grid grid-cols-5 gap-1">
            {[1, 2, 3, 4, 5].map((d) => {
              const data = finalResult.by_difficulty?.[String(d)]
              if (!data) return null
              const acc = data.total > 0 ? data.correct / data.total : 0
              return (
                <div key={d} className="bg-[#0a0a0f] rounded p-2 text-center">
                  <div className="text-xs font-mono text-[#6b7280] mb-0.5">D{d}</div>
                  <div className="text-sm font-mono font-semibold text-[#e5e7eb]">
                    {(acc * 100).toFixed(0)}%
                  </div>
                  <div className="text-[10px] font-mono text-[#6b7280]">
                    {data.correct}/{data.total}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
