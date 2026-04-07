import React, { useState, useEffect, useRef } from 'react'

interface ActivityEntry {
  id: number
  message: string
  agent?: string
  level: string
  timestamp: number
}

const levelColor: Record<string, string> = {
  info: '#8b949e',
  success: '#3fb950',
  error: '#f44336',
  progress: '#1f6feb',
}

const levelDot: Record<string, string> = {
  info: '●',
  success: '✓',
  error: '✗',
  progress: '◌',
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  return `${hh}:${mm}:${ss}`
}

const MAX_ENTRIES = 50

let _entryCounter = 0

export default function ActivityFeed() {
  const [entries, setEntries] = useState<ActivityEntry[]>([])
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const es = new EventSource('/api/consciousness/stream')
    esRef.current = es

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'activity') {
          const entry: ActivityEntry = {
            id: ++_entryCounter,
            message: data.message,
            agent: data.agent ?? undefined,
            level: data.level ?? 'info',
            timestamp: data.timestamp,
          }
          setEntries(prev => {
            const next = [...prev, entry]
            return next.length > MAX_ENTRIES ? next.slice(next.length - MAX_ENTRIES) : next
          })
        }
        // "state" events are ignored here — handled by ConsciousnessView
      } catch {}
    }

    es.onerror = () => {
      // Silently ignore SSE errors — ConsciousnessView handles reconnect
    }

    return () => {
      es.close()
    }
  }, [])

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [entries])

  return (
    <div style={{
      background: '#0d1117',
      borderTop: '1px solid #21262d',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      flexShrink: 0,
    }}>
      {/* Header bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '4px 12px',
        borderBottom: '1px solid #21262d',
        background: '#161b22',
        flexShrink: 0,
      }}>
        <span style={{
          fontSize: '11px',
          fontWeight: 600,
          color: '#484f58',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}>
          Aktivitätslog
        </span>
        {entries.length > 0 && (
          <span style={{
            marginLeft: '8px',
            fontSize: '10px',
            color: '#30363d',
          }}>
            {entries.length} Einträge
          </span>
        )}
      </div>

      {/* Log entries */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '4px 0',
      }}>
        {entries.length === 0 ? (
          <div style={{
            padding: '8px 12px',
            fontSize: '12px',
            color: '#30363d',
            fontFamily: 'monospace',
          }}>
            Warte auf Aktivität...
          </div>
        ) : (
          entries.map(entry => (
            <div
              key={entry.id}
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: '8px',
                padding: '2px 12px',
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                fontSize: '12px',
                lineHeight: '1.5',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              <span style={{ color: '#484f58', flexShrink: 0 }}>
                {formatTime(entry.timestamp)}
              </span>
              <span style={{
                color: levelColor[entry.level] ?? '#8b949e',
                flexShrink: 0,
                fontSize: '10px',
              }}>
                {levelDot[entry.level] ?? '●'}
              </span>
              {entry.agent && (
                <span style={{ color: '#7fdbca', flexShrink: 0, fontWeight: 600 }}>
                  [{entry.agent}]
                </span>
              )}
              <span style={{
                color: levelColor[entry.level] ?? '#8b949e',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {entry.message}
              </span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
