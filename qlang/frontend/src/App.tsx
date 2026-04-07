import React, { useState, useEffect } from 'react'
import ChatView from './ChatView'
import ConsciousnessView from './ConsciousnessView'

const globalStyles = `
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #e0e0e0; }
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
`

type Tab = 'chat' | 'consciousness'

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
      }}>
        <span style={{ fontSize: '18px', fontWeight: 700, color: '#7fdbca', letterSpacing: '0.05em' }}>QO</span>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button style={tabStyle('chat')} onClick={() => setActiveTab('chat')}>
            Chat
          </button>
          <button style={tabStyle('consciousness')} onClick={() => setActiveTab('consciousness')}>
            Bewusstsein
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'chat' ? <ChatView /> : <ConsciousnessView />}
      </div>
    </div>
  )
}
