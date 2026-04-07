import React, { useState, useEffect } from 'react'

interface Agent {
  role: string
  status: string
  tasks_completed: number
  tasks_failed: number
}

const agentMeta: Record<string, { emoji: string; color: string }> = {
  CEO:        { emoji: '👔', color: '#7fdbca' },
  Researcher: { emoji: '🔬', color: '#82aaff' },
  Developer:  { emoji: '💻', color: '#c3e88d' },
  Guardian:   { emoji: '🛡️', color: '#f78c6c' },
  Strategist: { emoji: '📊', color: '#ffcb6b' },
  Artisan:    { emoji: '🎨', color: '#bb86fc' },
}

const statusStyle: Record<string, { color: string; label: string }> = {
  Idle:   { color: '#3fb950', label: 'Bereit' },
  Active: { color: '#1f6feb', label: 'Aktiv' },
  Error:  { color: '#f44336', label: 'Fehler' },
}

function AgentCard({ agent }: { agent: Agent }) {
  const meta = agentMeta[agent.role] ?? { emoji: '🤖', color: '#7fdbca' }
  const status = statusStyle[agent.status] ?? { color: '#8b949e', label: agent.status }

  return (
    <div style={{
      background: '#161b22',
      border: `1px solid ${meta.color}33`,
      borderRadius: '12px',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '28px', marginBottom: '4px' }}>{meta.emoji}</div>
          <div style={{ fontSize: '15px', fontWeight: 600, color: meta.color }}>{agent.role}</div>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          padding: '4px 10px',
          borderRadius: '10px',
          background: status.color + '1a',
          border: `1px solid ${status.color}44`,
        }}>
          <span style={{
            width: '7px', height: '7px',
            borderRadius: '50%',
            background: status.color,
            display: 'inline-block',
          }} />
          <span style={{ fontSize: '11px', fontWeight: 600, color: status.color }}>{status.label}</span>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: 'flex', gap: '16px' }}>
        <div>
          <div style={{ fontSize: '20px', fontWeight: 600, color: '#3fb950' }}>{agent.tasks_completed}</div>
          <div style={{ fontSize: '11px', color: '#484f58' }}>erledigt</div>
        </div>
        <div style={{ width: '1px', background: '#21262d' }} />
        <div>
          <div style={{ fontSize: '20px', fontWeight: 600, color: '#f85149' }}>{agent.tasks_failed}</div>
          <div style={{ fontSize: '11px', color: '#484f58' }}>fehlgesch.</div>
        </div>
      </div>
    </div>
  )
}

export default function AgentsView() {
  const [agents, setAgents] = useState<Agent[]>([])

  useEffect(() => {
    const fetchAgents = () => {
      fetch('/api/agents')
        .then(r => r.json())
        .then(data => setAgents(Array.isArray(data) ? data : (data?.agents ?? [])))
        .catch(() => {})
    }
    fetchAgents()
    const interval = setInterval(fetchAgents, 3000)
    return () => clearInterval(interval)
  }, [])

  if (agents.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#484f58' }}>
        Lade Agenten...
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '24px 20px' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: '14px',
      }}>
        {agents.map(agent => (
          <AgentCard key={agent.role} agent={agent} />
        ))}
      </div>
    </div>
  )
}
