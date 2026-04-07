import React, { useEffect, useState } from 'react'

interface ProviderStats {
  name: string
  model: string
  requests: number
  total_tokens_estimate: number
  cost_usd: number
  avg_latency_ms: number
  status: string
}

interface ProvidersResponse {
  providers: ProviderStats[]
  total_cost_usd: number
  total_requests: number
}

interface CostSummary {
  total_cost_usd: number
  cloud_cost_usd: number
  groq_cost_usd: number
  local_cost_usd: number
  total_requests: number
  savings_vs_all_cloud_usd: number
}

const statusColor = (status: string): string => {
  switch (status) {
    case 'active':      return '#3fb950'
    case 'inactive':    return '#8b949e'
    case 'coming_soon': return '#d29922'
    case 'error':       return '#f85149'
    default:            return '#8b949e'
  }
}

const statusLabel = (status: string): string => {
  switch (status) {
    case 'active':      return 'Aktiv'
    case 'inactive':    return 'Inaktiv'
    case 'coming_soon': return 'Bald'
    case 'error':       return 'Fehler'
    default:            return status
  }
}

const fmtCost = (usd: number): string => `$${usd.toFixed(4)}`
const fmtTokens = (n: number): string => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

function StatusBadge({ status }: { status: string }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '5px',
      fontSize: '12px',
      fontWeight: 600,
      color: statusColor(status),
    }}>
      <span style={{
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        background: statusColor(status),
        display: 'inline-block',
      }} />
      {statusLabel(status)}
    </span>
  )
}

function LatencyBar({ ms, maxMs }: { ms: number; maxMs: number }) {
  const pct = maxMs > 0 ? Math.min((ms / maxMs) * 100, 100) : 0
  return (
    <div style={{ width: '100%', background: '#21262d', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
      <div style={{
        width: `${pct}%`,
        height: '100%',
        background: ms > 1000 ? '#d29922' : '#3fb950',
        borderRadius: '4px',
        transition: 'width 0.4s ease',
      }} />
    </div>
  )
}

function ProviderCard({ provider, maxLatency }: { provider: ProviderStats; maxLatency: number }) {
  return (
    <div style={{
      background: '#161b22',
      border: '1px solid #21262d',
      borderRadius: '10px',
      padding: '18px 20px',
      flex: '1 1 220px',
      minWidth: '200px',
      maxWidth: '340px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#e0e0e0' }}>{provider.name}</div>
          <div style={{ fontSize: '12px', color: '#8b949e', marginTop: '2px' }}>{provider.model || '—'}</div>
        </div>
        <StatusBadge status={provider.status} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        <div>
          <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '2px' }}>Requests</div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: '#7fdbca' }}>{provider.requests}</div>
        </div>
        <div>
          <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '2px' }}>Tokens (est.)</div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: '#7fdbca' }}>
            {fmtTokens(provider.total_tokens_estimate)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '2px' }}>Kosten</div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: '#e0e0e0' }}>{fmtCost(provider.cost_usd)}</div>
        </div>
        <div>
          <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '2px' }}>Latenz (avg)</div>
          <div style={{ fontSize: '18px', fontWeight: 700, color: '#e0e0e0' }}>
            {provider.avg_latency_ms > 0 ? `${provider.avg_latency_ms}ms` : '—'}
          </div>
        </div>
      </div>

      {provider.avg_latency_ms > 0 && (
        <div>
          <LatencyBar ms={provider.avg_latency_ms} maxMs={maxLatency} />
        </div>
      )}
    </div>
  )
}

function CostBar({ label, value, total, color }: { label: string; value: number; total: number; color: string }) {
  const pct = total > 0 ? (value / total) * 100 : 0
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px' }}>
        <span style={{ color: '#e0e0e0' }}>{label}</span>
        <span style={{ color: '#8b949e' }}>{fmtCost(value)}</span>
      </div>
      <div style={{ background: '#21262d', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`,
          minWidth: value > 0 ? '4px' : '0',
          height: '100%',
          background: color,
          borderRadius: '4px',
          transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  )
}

export default function ProviderView() {
  const [providers, setProviders] = useState<ProvidersResponse | null>(null)
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    try {
      const [provRes, costRes] = await Promise.all([
        fetch('/api/providers'),
        fetch('/api/providers/costs'),
      ])
      if (!provRes.ok || !costRes.ok) throw new Error('API error')
      const [prov, cost] = await Promise.all([provRes.json(), costRes.json()])
      setProviders(prov)
      setCostSummary(cost)
      setError(null)
    } catch (e) {
      setError('Fehler beim Laden der Provider-Daten')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [])

  const maxLatency = providers
    ? Math.max(...providers.providers.map(p => p.avg_latency_ms), 1)
    : 1

  return (
    <div style={{ padding: '20px', overflowY: 'auto', height: '100%' }}>
      {/* Header summary */}
      <div style={{
        background: '#161b22',
        border: '1px solid #21262d',
        borderRadius: '10px',
        padding: '18px 24px',
        marginBottom: '20px',
      }}>
        {loading ? (
          <div style={{ color: '#8b949e', fontSize: '14px' }}>Lade Daten…</div>
        ) : error ? (
          <div style={{ color: '#f85149', fontSize: '14px' }}>{error}</div>
        ) : costSummary ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '28px', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '4px' }}>Gesamtkosten</div>
              <div style={{ fontSize: '28px', fontWeight: 800, color: '#7fdbca' }}>
                {fmtCost(costSummary.total_cost_usd)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '4px' }}>Requests</div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#e0e0e0' }}>
                {costSummary.total_requests}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '4px' }}>
                Gesamt Tokens (est.)
              </div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#e0e0e0' }}>
                ~{fmtTokens(
                  providers?.providers.reduce((s, p) => s + p.total_tokens_estimate, 0) ?? 0
                )}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '4px' }}>
                Gespart vs. nur Cloud
              </div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#3fb950' }}>
                ~{fmtCost(costSummary.savings_vs_all_cloud_usd)}
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Provider cards */}
      {providers && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginBottom: '24px' }}>
          {providers.providers.map(p => (
            <ProviderCard key={p.name} provider={p} maxLatency={maxLatency} />
          ))}
        </div>
      )}

      {/* Cost breakdown bar chart */}
      {costSummary && (
        <div style={{
          background: '#161b22',
          border: '1px solid #21262d',
          borderRadius: '10px',
          padding: '18px 24px',
        }}>
          <div style={{ fontSize: '14px', fontWeight: 600, color: '#e0e0e0', marginBottom: '14px' }}>
            Kostenverteilung
          </div>
          {(() => {
            const max = Math.max(
              costSummary.groq_cost_usd,
              costSummary.cloud_cost_usd,
              costSummary.local_cost_usd,
              0.0001 // avoid divide-by-zero
            )
            return (
              <>
                <CostBar label="Groq (Free)" value={costSummary.groq_cost_usd} total={max} color="#7fdbca" />
                <CostBar label="Cloud LLM" value={costSummary.cloud_cost_usd} total={max} color="#d29922" />
                <CostBar label="QO-LLM (IGQK)" value={costSummary.local_cost_usd} total={max} color="#3fb950" />
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}
