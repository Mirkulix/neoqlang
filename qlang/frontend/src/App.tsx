import { useState } from 'react'
import {
  MessageSquare,
  Target,
  Users,
  Brain,
  Server,
  Dna,
  Clock,
} from 'lucide-react'
import ChatView from './ChatView'
import ConsciousnessView from './ConsciousnessView'
import GoalsView from './GoalsView'
import AgentsView from './AgentsView'
import EvolutionView from './EvolutionView'
import HistorieView from './HistorieView'
import ProviderView from './ProviderView'
import ActivityFeed from './ActivityFeed'

type Tab = 'chat' | 'goals' | 'agents' | 'consciousness' | 'provider' | 'evolution' | 'historie'

const tabs: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'goals', label: 'Ziele', icon: Target },
  { id: 'agents', label: 'Agenten', icon: Users },
  { id: 'consciousness', label: 'Bewusstsein', icon: Brain },
  { id: 'provider', label: 'Provider', icon: Server },
  { id: 'evolution', label: 'Evolution', icon: Dna },
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
    case 'historie': return <HistorieView onNavigate={onNavigate} />
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

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
        <div className="sidebar-footer">v0.1.0</div>
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
