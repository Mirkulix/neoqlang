import React, { useState, useEffect } from 'react'

interface HistoryEntry {
  id: number
  timestamp: number
  action_type: string
  description: string
  details: string
}

const TYPE_CONFIG: Record<string, { icon: string; color: string; label: string }> = {
  chat:               { icon: '\u{1F4AC}', color: '#7fdbca', label: 'Chat' },
  goal_created:       { icon: '\u{1F3AF}', color: '#ffcb6b', label: 'Ziel erstellt' },
  goal_completed:     { icon: '\u2705',    color: '#4caf50', label: 'Ziel erledigt' },
  goal_failed:        { icon: '\u274C',    color: '#f44336', label: 'Ziel fehlgeschlagen' },
  agent_executed:     { icon: '\u{1F52C}', color: '#9c27b0', label: 'Agent ausgeführt' },
  evolution_analyzed: { icon: '\u{1F9EC}', color: '#2196f3', label: 'Evolution analysiert' },
  proposal_approved:  { icon: '\u2714',    color: '#4caf50', label: 'Vorschlag genehmigt' },
  proposal_rejected:  { icon: '\u2718',    color: '#f44336', label: 'Vorschlag abgelehnt' },
}

function relativeTime(timestamp: number): string {
  const now = Math.floor(Date.now() / 1000)
  const diff = now - timestamp
  if (diff < 60) return 'gerade eben'
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`
  return `vor ${Math.floor(diff / 86400)} Tagen`
}

function HistoryItem({ entry }: { entry: HistoryEntry }) {
  const [expanded, setExpanded] = useState(false)
  const cfg = TYPE_CONFIG[entry.action_type] ?? { icon: '\u{1F4CB}', color: '#8b949e', label: entry.action_type }

  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '8px' }}>
      {/* Timeline line + dot */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '24px', flexShrink: 0 }}>
        <div style={{
          width: '24px',
          height: '24px',
          borderRadius: '50%',
          background: cfg.color + '22',
          border: `2px solid ${cfg.color}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          flexShrink: 0,
        }}>
          {cfg.icon}
        </div>
        <div style={{ flex: 1, width: '2px', background: '#21262d', marginTop: '4px' }} />
      </div>

      {/* Content */}
      <div style={{
        flex: 1,
        background: '#161b22',
        border: `1px solid ${cfg.color}33`,
        borderRadius: '8px',
        padding: '8px 12px',
        marginBottom: '4px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
          <div>
            <span style={{ fontSize: '11px', fontWeight: 600, color: cfg.color, marginRight: '8px' }}>
              {cfg.label}
            </span>
            <span style={{ fontSize: '12px', color: '#c9d1d9' }}>
              {entry.description.length > 80 ? entry.description.slice(0, 80) + '…' : entry.description}
            </span>
          </div>
          <span style={{ fontSize: '10px', color: '#484f58', flexShrink: 0 }}>
            {relativeTime(entry.timestamp)}
          </span>
        </div>
        {entry.details && (
          <div style={{ marginTop: '4px' }}>
            <button
              onClick={() => setExpanded(e => !e)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '10px',
                color: '#8b949e',
                padding: 0,
              }}
            >
              {expanded ? '- Details ausblenden' : '+ Details anzeigen'}
            </button>
            {expanded && (
              <div style={{
                marginTop: '4px',
                fontSize: '11px',
                color: '#8b949e',
                background: '#0d1117',
                borderRadius: '4px',
                padding: '6px 8px',
                wordBreak: 'break-word',
              }}>
                {entry.details}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function HistorieView() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)

  const fetchHistory = () => {
    fetch('/api/history?limit=100')
      .then(r => r.json())
      .then(data => {
        setEntries(Array.isArray(data) ? data : [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }

  useEffect(() => {
    fetchHistory()
    const interval = setInterval(fetchHistory, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '24px 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ margin: 0, fontSize: '16px', color: '#7fdbca', fontWeight: 600 }}>Aktionsverlauf</h2>
        <span style={{ fontSize: '12px', color: '#484f58' }}>{entries.length} Einträge</span>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', color: '#484f58', fontSize: '14px', marginTop: '48px' }}>
          Laden...
        </div>
      ) : entries.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#484f58', fontSize: '14px', marginTop: '48px' }}>
          Noch keine Aktionen aufgezeichnet.
        </div>
      ) : (
        <div>
          {entries.map(entry => (
            <HistoryItem key={entry.id} entry={entry} />
          ))}
        </div>
      )}
    </div>
  )
}
