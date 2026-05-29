import { useState, useRef } from 'react'

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

function ProgressBar({ value, max }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="w-full bg-[#2a2a3a] h-2 rounded-full overflow-hidden">
      <div
        className="h-full bg-[#a3e635] rounded-full transition-all duration-150"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

function TaskPreview({ task, index }) {
  return (
    <div className="border border-[#2a2a3a] rounded bg-[#0a0a0f] overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-[#2a2a3a] bg-[#13131a]">
        <span className="text-xs font-mono text-[#6b7280]">#{index + 1}</span>
        <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-[#1f2717] text-[#a3e635]">
          {task.domain}
        </span>
        <span className="text-xs font-mono text-[#6b7280]">D{task.difficulty}</span>
        <span className="text-xs font-mono text-[#3a3a4a] ml-auto truncate max-w-[140px]">
          {task.task_id?.slice(0, 8)}…
        </span>
      </div>
      <div className="p-3">
        <div className="question-text text-xs text-[#9ca3af] leading-relaxed line-clamp-4">
          {task.question}
        </div>
        {task.distractors?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {task.distractors.map((d, i) => (
              <span
                key={i}
                className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#1a1a24] text-[#6b7280] border border-[#2a2a3a]"
              >
                {d}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function Export() {
  const [domain, setDomain] = useState('scheduling')
  const [size, setSize] = useState(100)
  const [seed, setSeed] = useState(42)
  const [difficulty, setDifficulty] = useState('mixed')

  const [generating, setGenerating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [preview, setPreview] = useState([])
  const [error, setError] = useState(null)
  const [done, setDone] = useState(false)

  const abortRef = useRef(false)

  async function generate() {
    setGenerating(true)
    setProgress(0)
    setPreview([])
    setError(null)
    setDone(false)
    abortRef.current = false

    const domains = domain === 'all' ? DOMAINS : [domain]
    const perDomain = Math.ceil(size / domains.length)
    const tasks = []

    try {
      let idx = 0
      for (const dom of domains) {
        for (let i = 0; i < perDomain && tasks.length < size; i++) {
          if (abortRef.current) break
          const taskSeed = seed + idx
          const diff = difficulty === 'mixed'
            ? Math.floor(Math.random() * 5) + 1
            : parseInt(difficulty)

          const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain: dom, difficulty: diff, seed: taskSeed }),
          })

          if (!res.ok) {
            throw new Error(await parseErrorDetail(res))
          }

          const task = await res.json()
          tasks.push(task)
          idx++

          if (tasks.length <= 3) {
            setPreview((prev) => [...prev, task])
          }
          setProgress(tasks.length)
        }
        if (abortRef.current) break
      }
    } catch (e) {
      if (!abortRef.current) {
        setError(e.message === 'Failed to fetch' ? BACKEND_DOWN : e.message)
      }
      setGenerating(false)
      return
    }

    // Build JSONL
    const lines = tasks.map((t) =>
      JSON.stringify({
        task_id: t.task_id,
        domain: t.domain,
        difficulty: t.difficulty,
        question: t.question,
        distractors: t.distractors,
        answer_hash: t.answer_hash,
      })
    )
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `vtask_${domain}_${size}_seed${seed}.jsonl`
    a.click()
    URL.revokeObjectURL(url)

    setGenerating(false)
    setDone(true)
  }

  function cancel() {
    abortRef.current = true
    setGenerating(false)
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-6">
      <h2 className="text-sm font-mono text-[#9ca3af] uppercase tracking-wider mb-5">
        Export Dataset
      </h2>

      <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] p-4 mb-4">
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Domain</label>
            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              disabled={generating}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] rounded px-2 py-1.5 font-mono focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            >
              <option value="all">All domains</option>
              {DOMAINS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Difficulty</label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              disabled={generating}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] rounded px-2 py-1.5 font-mono focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            >
              <option value="mixed">Mixed</option>
              {[1, 2, 3, 4, 5].map((d) => (
                <option key={d} value={d}>Level {d}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Size</label>
            <input
              type="number"
              value={size}
              onChange={(e) => setSize(Math.max(1, Math.min(10000, parseInt(e.target.value) || 1)))}
              disabled={generating}
              min={1}
              max={10000}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] font-mono rounded px-3 py-1.5 focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            />
          </div>
          <div>
            <label className="block text-xs text-[#6b7280] font-mono mb-1">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
              disabled={generating}
              className="w-full bg-[#0a0a0f] border border-[#2a2a3a] text-sm text-[#e5e7eb] font-mono rounded px-3 py-1.5 focus:outline-none focus:border-[#a3e635] disabled:opacity-40"
            />
          </div>
        </div>

        {/* Progress */}
        {generating && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-mono text-[#9ca3af]">Generating…</span>
              <span className="text-xs font-mono text-[#a3e635]">
                {progress}/{size}
              </span>
            </div>
            <ProgressBar value={progress} max={size} />
          </div>
        )}

        {done && !generating && (
          <div className="mb-4 px-3 py-2 bg-[#1a2e10] border border-[#4a7c2e] text-[#a3e635] text-sm font-mono rounded">
            ✓ Generated {progress} tasks — download started
          </div>
        )}

        {error && (
          <div className="mb-4 px-3 py-2 bg-red-900/20 border border-red-800/50 text-red-400 text-sm font-mono rounded">
            {error}
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={generate}
            disabled={generating}
            className="flex-1 py-2 bg-[#a3e635] text-black text-sm font-mono font-semibold rounded hover:bg-[#b5f23f] disabled:opacity-40 transition-colors"
          >
            {generating ? 'Generating…' : 'Generate & Download'}
          </button>
          {generating && (
            <button
              onClick={cancel}
              className="px-4 py-2 border border-[#7c2e2e] text-[#f87171] text-sm font-mono rounded hover:bg-[#2e1010] transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Preview */}
      {preview.length > 0 && (
        <div>
          <div className="text-xs font-mono text-[#6b7280] mb-2 uppercase tracking-wider">
            Preview — first {preview.length} tasks
          </div>
          <div className="flex flex-col gap-2">
            {preview.map((task, i) => (
              <TaskPreview key={task.task_id} task={task} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
