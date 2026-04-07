import { useState, useEffect } from 'react'
import { Dna, Play, Check, X, AlertTriangle, TrendingUp, FlaskConical } from 'lucide-react'

interface EvolutionState {
  strategy_weights?: Record<string, number>
  generation?: number
  fitness_score?: number
  entropy?: number
}

interface Pattern {
  id: string
  name: string
  description: string
  severity: string
  detected_at?: string
}

interface Proposal {
  id: string
  title: string
  description: string
  status: string
  proposed_at?: string
}

const severityBadge: Record<string, string> = {
  low: 'badge-info',
  medium: 'badge-pending',
  high: 'badge-error',
  info: 'badge-info',
}

export default function EvolutionView() {
  const [evoState, setEvoState] = useState<EvolutionState | null>(null)
  const [patterns, setPatterns] = useState<Pattern[]>([])
  const [proposals, setProposals] = useState<Proposal[]>([])
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAll = async () => {
    try {
      const [stateRes, patternsRes, proposalsRes] = await Promise.all([
        fetch('/api/evolution/state').catch(() => null),
        fetch('/api/evolution/patterns').catch(() => null),
        fetch('/api/evolution/proposals').catch(() => null),
      ])

      if (stateRes?.ok) {
        const data = await stateRes.json()
        setEvoState(data)
      }
      if (patternsRes?.ok) {
        const data = await patternsRes.json()
        setPatterns(Array.isArray(data) ? data : (data?.patterns ?? []))
      }
      if (proposalsRes?.ok) {
        const data = await proposalsRes.json()
        setProposals(Array.isArray(data) ? data : (data?.proposals ?? []))
      }
    } catch {
      // Silently handle — evolution endpoints may not exist yet
    }
  }

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleAnalyze = async () => {
    setAnalyzing(true)
    setError(null)
    try {
      const res = await fetch('/api/evolution/analyze', { method: 'POST' })
      if (!res.ok) throw new Error('Analyse fehlgeschlagen')
      await fetchAll()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setAnalyzing(false)
    }
  }

  const handleProposal = async (id: string, action: 'approve' | 'reject') => {
    try {
      await fetch(`/api/evolution/proposals/${id}/${action}`, { method: 'POST' })
      fetchAll()
    } catch {}
  }

  const hasData = evoState?.strategy_weights || patterns.length > 0 || proposals.length > 0

  return (
    <div className="view">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="evo-icon-wrapper">
            <Dna size={24} />
          </div>
          <div>
            <h2 className="heading" style={{ fontSize: '18px', margin: 0 }}>Evolution</h2>
            {evoState?.generation != null && (
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                Generation {evoState.generation}
                {evoState.fitness_score != null && ` \u00b7 Fitness: ${(evoState.fitness_score * 100).toFixed(1)}%`}
                {evoState.entropy != null && ` \u00b7 Entropie: ${evoState.entropy.toFixed(4)}`}
              </div>
            )}
          </div>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleAnalyze}
          disabled={analyzing}
        >
          <Play size={16} />
          {analyzing ? 'Analysiere...' : 'Analyse starten'}
        </button>
      </div>

      {error && (
        <div style={{ marginBottom: '16px', fontSize: '12px', color: 'var(--accent-danger)' }}>{error}</div>
      )}

      {!hasData && (
        <div className="empty-state">
          <Dna size={40} className="empty-icon" />
          <div className="empty-title">Evolution -- Phase 4</div>
          <div className="empty-hint">
            Starte eine Analyse, um Muster zu erkennen und Verbesserungsvorschl&auml;ge zu generieren.
          </div>
        </div>
      )}

      {hasData && (
        <>
          {/* Strategy weights */}
          {evoState?.strategy_weights && Object.keys(evoState.strategy_weights).length > 0 && (
            <div className="card-static" style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <TrendingUp size={16} style={{ color: 'var(--accent-primary)' }} />
                <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Strategie-Gewichte</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(evoState.strategy_weights).map(([key, val]) => (
                  <div key={key}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '12px' }}>{key}</span>
                      <span className="mono" style={{ fontSize: '12px', color: 'var(--accent-primary)' }}>
                        {(val * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${Math.min(100, val * 100)}%`,
                          background: 'var(--accent-primary)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patterns */}
          {patterns.length > 0 && (
            <div className="card-static" style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <AlertTriangle size={16} style={{ color: 'var(--accent-warning)' }} />
                <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Erkannte Muster</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {patterns.map(p => (
                  <div key={p.id} className="evo-pattern-item">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 500 }}>{p.name}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {p.description}
                        </div>
                      </div>
                      <span className={`badge ${severityBadge[p.severity] ?? 'badge-info'}`}>
                        {p.severity}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Proposals */}
          {proposals.length > 0 && (
            <div className="card-static">
              <h3 className="heading" style={{ fontSize: '13px', margin: '0 0 16px' }}>Vorschl&auml;ge</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {proposals.map(p => (
                  <div key={p.id} className="evo-proposal-card">
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 500 }}>{p.title}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                        {p.description}
                      </div>
                    </div>
                    {p.status === 'pending' && (
                      <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                        <button
                          className="btn btn-ghost"
                          style={{ color: 'var(--accent-success)', minHeight: '36px', padding: '4px 12px' }}
                          onClick={() => handleProposal(p.id, 'approve')}
                        >
                          <Check size={16} /> Annehmen
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ color: 'var(--accent-danger)', minHeight: '36px', padding: '4px 12px' }}
                          onClick={() => handleProposal(p.id, 'reject')}
                        >
                          <X size={16} /> Ablehnen
                        </button>
                      </div>
                    )}
                    {p.status !== 'pending' && (
                      <span
                        className={`badge ${p.status === 'approved' ? 'badge-completed' : 'badge-error'}`}
                        style={{ marginTop: '8px', alignSelf: 'flex-start' }}
                      >
                        {p.status === 'approved' ? 'Genehmigt' : 'Abgelehnt'}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Simulation Placeholder */}
      <div className="card-static evo-simulation-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
          <FlaskConical size={18} style={{ color: 'var(--accent-info)' }} />
          <h3 className="heading" style={{ fontSize: '13px', margin: 0 }}>Szenario-Simulation</h3>
          <span className="badge badge-info" style={{ marginLeft: 'auto' }}>Verf&uuml;gbar ab n&auml;chster Version</span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
          Vor der Ausf&uuml;hrung eines Ziels simuliert QO verschiedene Strategien und w&auml;hlt die beste.
        </p>
      </div>

      <style>{`
        .evo-icon-wrapper {
          width: 44px;
          height: 44px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
          color: var(--accent-primary);
        }
        .evo-pattern-item {
          padding: 10px 14px;
          background: var(--bg-primary);
          border-radius: var(--radius-sm);
          border-left: 3px solid var(--accent-warning);
        }
        .evo-proposal-card {
          padding: 14px 16px;
          background: var(--bg-primary);
          border-radius: var(--radius-md);
          border: 1px solid var(--border);
          display: flex;
          flex-direction: column;
        }
        .evo-simulation-section {
          margin-top: 16px;
          border-color: var(--accent-info);
          border-left: 3px solid var(--accent-info);
          opacity: 0.8;
        }
      `}</style>
    </div>
  )
}
