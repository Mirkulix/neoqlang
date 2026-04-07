import React, { useState, useEffect, useRef } from 'react'

interface ConsciousnessState {
  mood: string
  energy: number
  heartbeat: number
  agents_active: number
  agents_idle: number
  tasks_completed: number
  tasks_failed: number
  values: {
    achtsamkeit: number
    anerkennung: number
    aufmerksamkeit: number
    entwicklung: number
    sinn: number
  }
}

const moodEmoji: Record<string, string> = {
  Learning: '📚',
  Focused: '🎯',
  Restless: '😤',
  Creating: '🎨',
  Reflecting: '🪞',
}

const valueColors: Record<string, string> = {
  achtsamkeit: '#7fdbca',
  anerkennung: '#bb86fc',
  aufmerksamkeit: '#f78c6c',
  entwicklung: '#ffcb6b',
  sinn: '#82aaff',
}

const valueLabels: Record<string, string> = {
  achtsamkeit: 'Achtsamkeit',
  anerkennung: 'Anerkennung',
  aufmerksamkeit: 'Aufmerksamkeit',
  entwicklung: 'Entwicklung',
  sinn: 'Sinn',
}

function energyColor(energy: number): string {
  if (energy > 60) return '#3fb950'
  if (energy > 30) return '#d29922'
  return '#f85149'
}

export default function ConsciousnessView() {
  const [state, setState] = useState<ConsciousnessState | null>(null)
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const applyState = (data: ConsciousnessState) => {
    setState(data)
  }

  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(() => {
      fetch('/api/consciousness/state')
        .then(r => r.json())
        .then(applyState)
        .catch(() => {})
    }, 2000)
    // fetch immediately
    fetch('/api/consciousness/state')
      .then(r => r.json())
      .then(applyState)
      .catch(() => {})
  }

  useEffect(() => {
    const es = new EventSource('/api/consciousness/stream')
    esRef.current = es

    es.onopen = () => {
      setConnected(true)
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ConsciousnessState
        applyState(data)
      } catch {}
    }

    es.onerror = () => {
      setConnected(false)
      es.close()
      startPolling()
    }

    return () => {
      es.close()
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  if (!state) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#484f58' }}>
        Verbinde mit Bewusstseinsstrom...
      </div>
    )
  }

  const emoji = moodEmoji[state.mood] ?? '🤔'

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '24px 20px' }}>
      {/* Connection indicator */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '16px' }}>
        <span style={{
          fontSize: '12px',
          color: connected ? '#3fb950' : '#d29922',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
        }}>
          <span style={{
            width: '7px', height: '7px', borderRadius: '50%',
            background: connected ? '#3fb950' : '#d29922',
            display: 'inline-block',
          }} />
          {connected ? 'Live-Stream' : 'Polling'}
        </span>
      </div>

      {/* Mood */}
      <div style={{
        background: '#161b22',
        border: '1px solid #21262d',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '16px',
        textAlign: 'center',
      }}>
        <div style={{ fontSize: '56px', marginBottom: '8px' }}>{emoji}</div>
        <div style={{ fontSize: '13px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px' }}>Stimmung</div>
        <div style={{ fontSize: '24px', fontWeight: 600, color: '#7fdbca' }}>{state.mood}</div>
      </div>

      {/* Energy + Heartbeat row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
        {/* Energy */}
        <div style={{
          background: '#161b22',
          border: '1px solid #21262d',
          borderRadius: '12px',
          padding: '16px',
        }}>
          <div style={{ fontSize: '12px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '10px' }}>Energie</div>
          <div style={{ fontSize: '28px', fontWeight: 600, color: energyColor(state.energy), marginBottom: '10px' }}>
            {Math.round(state.energy)}%
          </div>
          <div style={{ height: '6px', background: '#21262d', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${Math.min(100, state.energy)}%`,
              background: energyColor(state.energy),
              borderRadius: '3px',
              transition: 'width 0.5s ease',
            }} />
          </div>
        </div>

        {/* Heartbeat */}
        <div style={{
          background: '#161b22',
          border: '1px solid #21262d',
          borderRadius: '12px',
          padding: '16px',
        }}>
          <div style={{ fontSize: '12px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '10px' }}>Herzschlag</div>
          <div style={{ fontSize: '28px', fontWeight: 600, color: '#f85149', marginBottom: '4px' }}>
            ♥ {state.heartbeat}
          </div>
          <div style={{ fontSize: '12px', color: '#484f58' }}>Ticks</div>
        </div>
      </div>

      {/* Agents + Tasks row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
        {/* Agents */}
        <div style={{
          background: '#161b22',
          border: '1px solid #21262d',
          borderRadius: '12px',
          padding: '16px',
        }}>
          <div style={{ fontSize: '12px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '12px' }}>Agenten</div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '22px', fontWeight: 600, color: '#7fdbca' }}>{state.agents_active}</div>
              <div style={{ fontSize: '11px', color: '#484f58' }}>aktiv</div>
            </div>
            <div>
              <div style={{ fontSize: '22px', fontWeight: 600, color: '#8b949e' }}>{state.agents_idle}</div>
              <div style={{ fontSize: '11px', color: '#484f58' }}>bereit</div>
            </div>
          </div>
        </div>

        {/* Tasks */}
        <div style={{
          background: '#161b22',
          border: '1px solid #21262d',
          borderRadius: '12px',
          padding: '16px',
        }}>
          <div style={{ fontSize: '12px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '12px' }}>Aufgaben</div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '22px', fontWeight: 600, color: '#3fb950' }}>{state.tasks_completed}</div>
              <div style={{ fontSize: '11px', color: '#484f58' }}>erledigt</div>
            </div>
            <div>
              <div style={{ fontSize: '22px', fontWeight: 600, color: '#f85149' }}>{state.tasks_failed}</div>
              <div style={{ fontSize: '11px', color: '#484f58' }}>fehlgesch.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Values */}
      <div style={{
        background: '#161b22',
        border: '1px solid #21262d',
        borderRadius: '12px',
        padding: '20px',
      }}>
        <div style={{ fontSize: '12px', color: '#484f58', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '16px' }}>Werte</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {Object.entries(state.values).map(([key, val]) => (
            <div key={key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '13px', color: '#c9d1d9' }}>{valueLabels[key] ?? key}</span>
                <span style={{ fontSize: '13px', color: valueColors[key] ?? '#7fdbca', fontVariantNumeric: 'tabular-nums' }}>
                  {(val * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ height: '5px', background: '#21262d', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(100, val * 100)}%`,
                  background: valueColors[key] ?? '#7fdbca',
                  borderRadius: '3px',
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
