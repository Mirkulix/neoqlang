import { useState, useEffect } from 'react'
import {
  MessageSquare,
  Target,
  Users,
  Brain,
  Server,
  Dna,
  GitBranch,
  Clock,
  Sun,
  Moon,
} from 'lucide-react'
import ChatView from './ChatView'
import ConsciousnessView from './ConsciousnessView'
import GoalsView from './GoalsView'
import AgentsView from './AgentsView'
import EvolutionView from './EvolutionView'
import GraphsView from './GraphsView'
import HistorieView from './HistorieView'
import ProviderView from './ProviderView'
import ActivityFeed from './ActivityFeed'

type Tab = 'chat' | 'goals' | 'agents' | 'consciousness' | 'provider' | 'evolution' | 'graphs' | 'historie'

const tabs: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'goals', label: 'Ziele', icon: Target },
  { id: 'agents', label: 'Agenten', icon: Users },
  { id: 'consciousness', label: 'Bewusstsein', icon: Brain },
  { id: 'provider', label: 'Provider', icon: Server },
  { id: 'evolution', label: 'Evolution', icon: Dna },
  { id: 'graphs', label: 'QLANG', icon: GitBranch },
  { id: 'historie', label: 'Historie', icon: Clock },
]

function renderView(tab: Tab, onNavigate: (tab: string) => void) {
  switch (tab) {
    case 'chat': return <ChatView />
    case 'goals': return <GoalsView />
    case 'agents': return <AgentsView />
    case 'consciousness': return <ConsciousnessView />
    case 'provider': return <ProviderView />
    case 'evolution': return <EvolutionView />
    case 'graphs': return <GraphsView />
    case 'historie': return <HistorieView onNavigate={onNavigate} />
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return localStorage.getItem('qo-theme') === 'dark'
  })
  const [connected, setConnected] = useState<boolean>(true)

  // Apply theme to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark')
      localStorage.setItem('qo-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
      localStorage.setItem('qo-theme', 'light')
    }
  }, [darkMode])

  // Poll /api/health every 30 seconds for connection status
  useEffect(() => {
    const checkHealth = () => {
      fetch('/api/health')
        .then(r => setConnected(r.ok))
        .catch(() => setConnected(false))
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [])

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey

      // Ctrl+1..8 — switch tabs
      if (ctrl && e.key >= '1' && e.key <= '8') {
        const idx = parseInt(e.key, 10) - 1
        if (idx < tabs.length) {
          e.preventDefault()
          setActiveTab(tabs[idx].id)
        }
        return
      }

      // Ctrl+K — focus chat input
      if (ctrl && e.key === 'k') {
        e.preventDefault()
        setActiveTab('chat')
        // ChatView listens to a custom event to focus its input
        window.dispatchEvent(new CustomEvent('qo:focus-chat-input'))
        return
      }

      // Escape — blur / close (dispatch event for consumers)
      if (e.key === 'Escape') {
        window.dispatchEvent(new CustomEvent('qo:escape'))
        return
      }

      // Ctrl+Enter / Cmd+Enter — send chat message
      if (ctrl && e.key === 'Enter') {
        if (activeTab === 'chat') {
          e.preventDefault()
          window.dispatchEvent(new CustomEvent('qo:send-chat'))
        }
        return
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeTab])

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>QO</h1>
        </div>
        <nav className="sidebar-nav">
          {tabs.map(tab => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                className={`nav-item${activeTab === tab.id ? ' active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon className="nav-icon" size={20} />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer" style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center' }}>
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setDarkMode(d => !d)}
            title={darkMode ? 'Light Mode' : 'Dark Mode'}
            aria-label={darkMode ? 'Light Mode aktivieren' : 'Dark Mode aktivieren'}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <div
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            title={connected ? 'Verbunden' : 'Nicht verbunden'}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: connected ? '#22c55e' : '#ef4444',
              display: 'inline-block',
              flexShrink: 0,
            }} />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {connected ? 'Online' : 'Offline'}
            </span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>v0.1.0</span>
        </div>
      </aside>

      <div className="main-area">
        <div className="main-content">
          {renderView(activeTab, (t) => setActiveTab(t as Tab))}
        </div>
        <div className="activity-panel">
          <ActivityFeed />
        </div>
      </div>
    </div>
  )
}
