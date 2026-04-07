import React, { useState, useEffect, useRef } from 'react'

const pulseStyle = `
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.3); }
  }
`

interface Subtask {
  description: string
  assigned_agent: string
  status: string
  result?: string
}

interface Goal {
  id: string
  description: string
  status: 'Pending' | 'InProgress' | 'Completed' | 'Failed'
  subtasks: Subtask[]
  result?: string
  created_at: string
}

const statusColor: Record<string, string> = {
  Pending: '#ffcb6b',
  InProgress: '#1f6feb',
  Completed: '#7fdbca',
  Failed: '#f44336',
}

const statusLabel: Record<string, string> = {
  Pending: 'Ausstehend',
  InProgress: 'In Arbeit',
  Completed: 'Erledigt',
  Failed: 'Fehlgeschlagen',
}

function StatusBadge({ status }: { status: string }) {
  const isInProgress = status === 'InProgress'
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '5px',
      padding: '2px 10px',
      borderRadius: '10px',
      fontSize: '11px',
      fontWeight: 600,
      background: (statusColor[status] ?? '#8b949e') + '22',
      color: statusColor[status] ?? '#8b949e',
      border: `1px solid ${statusColor[status] ?? '#8b949e'}44`,
    }}>
      {isInProgress && (
        <span style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: statusColor['InProgress'],
          display: 'inline-block',
          animation: 'pulse 1.2s ease-in-out infinite',
        }} />
      )}
      {statusLabel[status] ?? status}
    </span>
  )
}

function GoalCard({ goal }: { goal: Goal }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div style={{
      background: '#161b22',
      border: '1px solid #21262d',
      borderRadius: '10px',
      padding: '16px',
      marginBottom: '10px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '14px', color: '#c9d1d9', marginBottom: '6px' }}>{goal.description}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <StatusBadge status={goal.status} />
            <span style={{ fontSize: '11px', color: '#484f58' }}>
              {new Date(goal.created_at).toLocaleString('de-DE', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' })}
            </span>
            {goal.subtasks && goal.subtasks.length > 0 && (
              <button
                onClick={() => setExpanded(e => !e)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: '#7fdbca',
                  fontSize: '11px',
                  padding: '0',
                }}
              >
                {expanded ? '- Teilaufgaben ausblenden' : `+ ${goal.subtasks.length} Teilaufgabe${goal.subtasks.length !== 1 ? 'n' : ''}`}
              </button>
            )}
          </div>
        </div>
      </div>

      {expanded && goal.subtasks && goal.subtasks.length > 0 && (
        <div style={{ marginTop: '12px', borderTop: '1px solid #21262d', paddingTop: '12px' }}>
          {goal.subtasks.map((sub, i) => (
            <div key={i} style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '3px',
              padding: '8px 10px',
              background: '#0d1117',
              borderRadius: '6px',
              marginBottom: '6px',
              borderLeft: `3px solid ${statusColor[sub.status] ?? '#30363d'}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: '#c9d1d9' }}>{sub.description}</span>
                <StatusBadge status={sub.status} />
              </div>
              <div style={{ fontSize: '11px', color: '#8b949e' }}>
                Agent: <span style={{ color: '#7fdbca' }}>{sub.assigned_agent}</span>
              </div>
              {sub.result && (
                <div style={{ fontSize: '11px', color: '#8b949e', marginTop: '2px' }}>
                  Ergebnis: <span style={{ color: '#e0e0e0' }}>{sub.result}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {goal.status === 'Completed' && goal.result && (
        <div style={{
          marginTop: '10px',
          padding: '8px 12px',
          background: '#7fdbca11',
          border: '1px solid #7fdbca33',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#7fdbca',
        }}>
          {goal.result}
        </div>
      )}
    </div>
  )
}

export default function GoalsView() {
  const [goals, setGoals] = useState<Goal[]>([])
  const [input, setInput] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const styleRef = useRef<HTMLStyleElement | null>(null)

  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = pulseStyle
    document.head.appendChild(style)
    styleRef.current = style
    return () => {
      if (styleRef.current) document.head.removeChild(styleRef.current)
    }
  }, [])

  const fetchGoals = () => {
    fetch('/api/goals')
      .then(r => r.json())
      .then(data => setGoals(Array.isArray(data) ? data : []))
      .catch(() => {})
  }

  useEffect(() => {
    fetchGoals()
    const interval = setInterval(fetchGoals, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await fetch('/api/goals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: trimmed }),
      })
      if (!res.ok) throw new Error('Fehler beim Erstellen')
      setInput('')
      fetchGoals()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '24px 20px' }}>
      {/* New goal form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Neues Ziel beschreiben..."
            disabled={submitting}
            style={{
              flex: 1,
              background: '#161b22',
              border: '1px solid #21262d',
              borderRadius: '8px',
              padding: '10px 14px',
              color: '#e0e0e0',
              fontSize: '14px',
              outline: 'none',
            }}
          />
          <button
            type="submit"
            disabled={submitting || !input.trim()}
            style={{
              background: submitting ? '#21262d' : '#7fdbca',
              color: submitting ? '#8b949e' : '#0d1117',
              border: 'none',
              borderRadius: '8px',
              padding: '10px 18px',
              fontSize: '14px',
              fontWeight: 600,
              cursor: submitting ? 'not-allowed' : 'pointer',
              flexShrink: 0,
            }}
          >
            {submitting ? '...' : 'Starten'}
          </button>
        </div>
        {error && (
          <div style={{ marginTop: '8px', fontSize: '12px', color: '#f44336' }}>{error}</div>
        )}
      </form>

      {/* Goals list */}
      {goals.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#484f58', fontSize: '14px', marginTop: '48px' }}>
          Keine Ziele vorhanden. Erstelle ein neues Ziel oben.
        </div>
      ) : (
        goals.map(goal => <GoalCard key={goal.id} goal={goal} />)
      )}
    </div>
  )
}
