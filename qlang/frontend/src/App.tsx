import React, { useState, useEffect } from 'react'
import ChatView from './ChatView'
import ConsciousnessView from './ConsciousnessView'
import GoalsView from './GoalsView'
import AgentsView from './AgentsView'
import EvolutionView from './EvolutionView'

const globalStyles = `
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #e0e0e0; }
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
`

type Tab = 'chat' | 'goals' | 'agents' | 'consciousness' | 'evolution'

const tabs: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'goals', label: 'Ziele' },
  { id: 'agents', label: 'Agenten' },
  { id: 'consciousness', label: 'Bewusstsein' },
  { id: 'evolution', label: 'Evolution' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = globalStyles
    document.head.appendChild(style)
    return () => { document.head.removeChild(style) }
  }, [])

  const tabStyle = (tab: Tab): React.CSSProperties => ({
    padding: '8px 20px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.15s ease',
    background: activeTab === tab ? '#7fdbca' : 'transparent',
    color: activeTab === tab ? '#0d1117' : '#8b949e',
  })

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':         return <ChatView />
      case 'goals':        return <GoalsView />
      case 'agents':       return <AgentsView />
      case 'consciousness': return <ConsciousnessView />
      case 'evolution':    return <EvolutionView />
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0d1117' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px 20px',
        borderBottom: '1px solid #21262d',
        background: '#161b22',
        gap: '24px',
        flexShrink: 0,
        flexWrap: 'wrap',
      }}>
        <span style={{ fontSize: '18px', fontWeight: 700, color: '#7fdbca', letterSpacing: '0.05em' }}>QO</span>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {tabs.map(tab => (
            <button key={tab.id} style={tabStyle(tab.id)} onClick={() => setActiveTab(tab.id)}>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {renderContent()}
      </div>
    </div>
  )
}
