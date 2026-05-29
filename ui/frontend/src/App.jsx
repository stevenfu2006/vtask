import { useState } from 'react'
import Explorer from './views/Explorer.jsx'
import Eval from './views/Eval.jsx'
import Export from './views/Export.jsx'

const NAV = [
  {
    id: 'explorer',
    label: 'Explorer',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="w-4 h-4">
        <rect x="2" y="2" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/>
        <rect x="11" y="2" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/>
        <rect x="2" y="11" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/>
        <rect x="11" y="11" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/>
      </svg>
    ),
  },
  {
    id: 'eval',
    label: 'Eval',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="w-4 h-4">
        <path d="M3 14l4-5 3 3 4-6 3 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="16" cy="5" r="1.5" fill="currentColor"/>
      </svg>
    ),
  },
  {
    id: 'export',
    label: 'Export',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="w-4 h-4">
        <path d="M10 3v9m0 0l-3-3m3 3l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M3 14v1a2 2 0 002 2h10a2 2 0 002-2v-1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-4 pt-5 pb-4">
      <svg viewBox="0 0 28 28" className="w-7 h-7 shrink-0">
        <rect x="1.5" y="1.5" width="25" height="25" rx="4" fill="none" stroke="#a3e635" strokeWidth="2"/>
        <polyline
          points="7,14 11,18 21,8"
          fill="none"
          stroke="#a3e635"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="font-mono font-semibold text-sm tracking-tight text-white">VTASK</span>
    </div>
  )
}

function Sidebar({ view, setView }) {
  return (
    <aside className="flex flex-col w-52 shrink-0 border-r border-[#2a2a3a] bg-[#13131a] h-full">
      <Logo />

      <div className="h-px bg-[#2a2a3a] mx-3 mb-3" />

      <nav className="flex flex-col gap-0.5 px-2 flex-1">
        {NAV.map((item) => {
          const active = view === item.id
          return (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={`flex items-center gap-2.5 px-3 py-2 rounded text-sm font-medium transition-colors text-left w-full
                ${active
                  ? 'bg-[#1f2717] text-[#a3e635]'
                  : 'text-[#9ca3af] hover:text-[#e5e7eb] hover:bg-[#1a1a24]'
                }`}
            >
              <span className={active ? 'text-[#a3e635]' : 'text-[#6b7280]'}>{item.icon}</span>
              {item.label}
            </button>
          )
        })}
      </nav>

      <div className="px-4 pb-5 pt-3 border-t border-[#2a2a3a]">
        <div className="font-mono text-xs text-[#6b7280]">VTASK v0.1.0</div>
        <div className="mt-1 inline-flex items-center gap-1 bg-[#1f2717] text-[#a3e635] text-xs font-mono px-2 py-0.5 rounded-full">
          6 domains
        </div>
      </div>
    </aside>
  )
}

export default function App() {
  const [view, setView] = useState('explorer')

  return (
    <div className="flex h-screen bg-[#0a0a0f] overflow-hidden">
      <Sidebar view={view} setView={setView} />
      <main className="flex-1 overflow-auto">
        {view === 'explorer' && <Explorer />}
        {view === 'eval' && <Eval />}
        {view === 'export' && <Export />}
      </main>
    </div>
  )
}
