import { useState, useEffect, useRef } from 'react'

const DOMAINS = ['scheduling', 'dependencies', 'inventory', 'fsm', 'spatial', 'temporal']

const DOMAIN_SHORT = {
  scheduling: 'Scheduling',
  dependencies: 'Dependencies',
  inventory: 'Inventory',
  fsm: 'FSM',
  spatial: 'Spatial',
  temporal: 'Temporal',
}

function DifficultyPips({ value, onChange }) {
  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          className={`w-4 h-4 rounded-sm border transition-colors
            ${value === d
              ? 'bg-[#a3e635] border-[#a3e635]'
              : 'bg-transparent border-[#2a2a3a] hover:border-[#a3e635]'
            }`}
          title={`Difficulty ${d}`}
        />
      ))}
    </div>
  )
}

function MetaStrip({ domain, difficulty, taskId }) {
  if (!domain) return null
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-t border-[#2a2a3a] bg-[#0d0d14]">
      <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-[#1f2717] text-[#a3e635]">
        {DOMAIN_SHORT[domain] || domain}
      </span>
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((d) => (
          <div
            key={d}
            className={`w-2 h-2 rounded-full ${d <= difficulty ? 'bg-[#a3e635]' : 'bg-[#2a2a3a]'}`}
          />
        ))}
      </div>
      {taskId && (
        <span className="font-mono text-[11px] text-[#6b7280] ml-auto truncate max-w-[200px]">
          {taskId}
        </span>
      )}
    </div>
  )
}

const BACKEND_DOWN = 'Backend not reachable — start it first:\n  cd ui/backend && uvicorn main:app --port 8000'

// FastAPI always returns JSON errors. A non-JSON body means the Vite proxy
// couldn't reach the backend (ECONNREFUSED → any non-2xx proxy response).
async function parseErrorDetail(res) {
  try {
    const body = await res.json()
    return body.detail || `HTTP ${res.status}`
  } catch {
    return BACKEND_DOWN
  }
}

export default function Explorer() {
  const [domain, setDomain] = useState('scheduling')
  const [difficulty, setDifficulty] = useState(2)
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 9999))
  const [distractorMode, setDistractorMode] = useState(false)

  const [task, setTask] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [answer, setAnswer] = useState('')
  const [result, setResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const answerRef = useRef(null)

  // fetchTask accepts explicit values to avoid stale-closure bugs when called
  // immediately after a state update (e.g. newTask with a fresh seed).
  async function fetchTask(opts = {}) {
    const d = opts.domain ?? domain
    const diff = opts.difficulty ?? difficulty
    const s = opts.seed ?? seed

    setLoading(true)
    setError(null)
    setResult(null)
    setAnswer('')
    setTask(null)

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain: d, difficulty: diff, seed: s }),
      })
      if (!res.ok) {
        throw new Error(await parseErrorDetail(res))
      }
      const data = await res.json()
      setTask(data)
      setTimeout(() => answerRef.current?.focus(), 50)
    } catch (e) {
      setError(e.message === 'Failed to fetch' ? BACKEND_DOWN : e.message)
    } finally {
      setLoading(false)
    }
  }

  async function submitAnswer(ans) {
    if (!task || submitting) return
    setSubmitting(true)
    try {
      const res = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: task.task_id, answer: ans }),
      })
      if (!res.ok) {
        throw new Error(await parseErrorDetail(res))
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && answer.trim()) {
      submitAnswer(answer.trim())
    }
  }

  function newTask() {
    const s = Math.floor(Math.random() * 99999)
    setSeed(s)
    setDistractorMode(false)
    // Pass the new seed explicitly — state update (setSeed) is async,
    // so the closure would still see the old value if we relied on it.
    fetchTask({ seed: s })
  }

  // Single initial fetch on mount only.
  useEffect(() => {
    fetchTask()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="max-w-3xl mx-auto px-6 py-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-xs text-[#9ca3af] font-mono uppercase tracking-wider">Domain</label>
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="bg-[#13131a] border border-[#2a2a3a] text-sm text-[#e5e7eb] rounded px-2 py-1 font-mono focus:outline-none focus:border-[#a3e635]"
          >
            {DOMAINS.map((d) => (
              <option key={d} value={d}>{DOMAIN_SHORT[d]}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-[#9ca3af] font-mono uppercase tracking-wider">Diff</label>
          <DifficultyPips value={difficulty} onChange={setDifficulty} />
        </div>

        <div className="flex items-center gap-2 ml-auto">
          <span className="font-mono text-xs text-[#6b7280]">seed:{seed}</span>
          <button
            onClick={() => fetchTask()}
            disabled={loading}
            className="px-3 py-1.5 bg-[#a3e635] text-black text-xs font-mono font-semibold rounded hover:bg-[#b5f23f] disabled:opacity-40 transition-colors"
          >
            {loading ? 'Loading…' : 'New Task'}
          </button>
          <button
            onClick={newTask}
            disabled={loading}
            className="px-3 py-1.5 border border-[#2a2a3a] text-[#9ca3af] text-xs font-mono rounded hover:border-[#a3e635] hover:text-[#a3e635] disabled:opacity-40 transition-colors"
            title="Random seed + new task"
          >
            ↺ Rand
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 px-3 py-2 bg-red-900/20 border border-red-800/50 text-red-400 text-sm font-mono rounded">
          {error}
        </div>
      )}

      {/* Loading placeholder */}
      {loading && (
        <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] p-8 text-center">
          <div className="font-mono text-sm text-[#6b7280]">Generating task…</div>
        </div>
      )}

      {/* Task card */}
      {!loading && task && (
        <div className="rounded-lg border border-[#2a2a3a] bg-[#13131a] overflow-hidden animate-fade-in">
          {/* Question */}
          <div className="p-4 border-b border-[#2a2a3a]">
            <div className="question-text text-sm text-[#e5e7eb] leading-relaxed">
              {task.question}
            </div>
          </div>

          {/* Answer area */}
          <div className="p-4">
            {/* Distractor mode toggle — only when ≥2 options available */}
            {task.options && task.options.length >= 2 && (
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs text-[#6b7280] font-mono">Answer mode</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-[#9ca3af]">
                    {distractorMode ? 'Multiple choice' : 'Free text'}
                  </span>
                  <button
                    onClick={() => { setDistractorMode(!distractorMode); setAnswer('') }}
                    className={`relative w-10 h-5 rounded-full transition-colors ${
                      distractorMode ? 'bg-[#a3e635]' : 'bg-[#2a2a3a]'
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                        distractorMode ? 'translate-x-5' : ''
                      }`}
                    />
                  </button>
                </div>
              </div>
            )}

            {result ? (
              /* Result display */
              <div className={`rounded px-4 py-3 border animate-fade-in ${
                result.correct
                  ? 'bg-[#1a2e10] border-[#4a7c2e] text-[#a3e635]'
                  : 'bg-[#2e1010] border-[#7c2e2e] text-[#f87171]'
              }`}>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="text-lg">{result.correct ? '✓' : '✗'}</span>
                  <span className="font-mono font-semibold">
                    {result.correct ? 'Correct' : 'Incorrect'}
                  </span>
                  <span className="ml-auto font-mono text-sm opacity-70">
                    score: {result.score.toFixed(1)}
                  </span>
                </div>
                <div className="font-mono text-sm opacity-90">
                  Correct answer: <strong>{result.correct_answer}</strong>
                </div>
                {!result.correct && answer && (
                  <div className="font-mono text-xs opacity-60 mt-1">
                    Your answer: {answer}
                  </div>
                )}
              </div>
            ) : distractorMode && task.options ? (
              /* Multiple choice buttons */
              <div className="grid grid-cols-2 gap-2">
                {task.options.map((opt, i) => (
                  <button
                    key={i}
                    onClick={() => { setAnswer(opt); submitAnswer(opt) }}
                    disabled={submitting}
                    className="px-3 py-2.5 border border-[#2a2a3a] bg-[#0a0a0f] text-[#e5e7eb] font-mono text-sm rounded hover:border-[#a3e635] hover:bg-[#1a1a24] disabled:opacity-40 transition-colors text-left"
                  >
                    <span className="text-[#6b7280] mr-2 text-xs">
                      {String.fromCharCode(65 + i)}.
                    </span>
                    {opt}
                  </button>
                ))}
              </div>
            ) : (
              /* Free text input */
              <div className="flex gap-2">
                <input
                  ref={answerRef}
                  type="text"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your answer…"
                  disabled={submitting}
                  className="flex-1 bg-[#0a0a0f] border border-[#2a2a3a] text-[#e5e7eb] font-mono text-sm rounded px-3 py-2 focus:outline-none focus:border-[#a3e635] placeholder-[#3a3a4a] disabled:opacity-40"
                />
                <button
                  onClick={() => submitAnswer(answer.trim())}
                  disabled={!answer.trim() || submitting}
                  className="px-4 py-2 bg-[#a3e635] text-black text-sm font-mono font-semibold rounded hover:bg-[#b5f23f] disabled:opacity-40 transition-colors"
                >
                  {submitting ? '…' : 'Submit'}
                </button>
              </div>
            )}
          </div>

          {/* Metadata strip */}
          <MetaStrip domain={task.domain} difficulty={task.difficulty} taskId={task.task_id} />
        </div>
      )}
    </div>
  )
}
