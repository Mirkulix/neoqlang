import { useEffect, useState } from 'react'
import { Server } from 'lucide-react'

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

const statusBadge: Record<string, { cls: string; label: string }> = {
  active:      { cls: 'badge-idle', label: 'Aktiv' },
  inactive:    { cls: 'badge-info', label: 'Inaktiv' },
  coming_soon: { cls: 'badge-pending', label: 'Bald' },
  error:       { cls: 'badge-error', label: 'Fehler' },
}

const fmtCost = (usd: number): string => `$${usd.toFixed(4)}`
const fmtTokens = (n: number): string => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

function ProviderCard({ provider, maxLatency }: { provider: ProviderStats; maxLatency: number }) {
  const st = statusBadge[provider.status] ?? { cls: 'badge-info', label: provider.status }
  const latencyPct = maxLatency > 0 ? Math.min((provider.avg_latency_ms / maxLatency) * 100, 100) : 0

  return (
    <div className="card provider-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="provider-icon">
            <Server size={20} />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 600 }}>{provider.name}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              {provider.model || '\u2014'}
            </div>
          </div>
        </div>
        <span className={`badge ${st.cls}`}>
          <span className="badge-dot" />
          {st.label}
        </span>
      </div>

      <div className="provider-stats-grid">
        <div>
          <div className="label">Requests</div>
          <div className="stat-value" style={{ fontSize: '18px', color: 'var(--accent-primary)' }}>
            {provider.requests}
          </div>
        </div>
        <div>
          <div className="label">Tokens (est.)</div>
          <div className="stat-value" style={{ fontSize: '18px', color: 'var(--accent-primary)' }}>
            {fmtTokens(provider.total_tokens_estimate)}
          </div>
        </div>
        <div>
          <div className="label">Kosten</div>
          <div className="stat-value mono" style={{ fontSize: '18px' }}>
            {fmtCost(provider.cost_usd)}
          </div>
        </div>
        <div>
          <div className="label">Latenz (avg)</div>
          <div className="stat-value mono" style={{ fontSize: '18px' }}>
            {provider.avg_latency_ms > 0 ? `${provider.avg_latency_ms}ms` : '\u2014'}
          </div>
        </div>
      </div>

      {provider.avg_latency_ms > 0 && (
        <div className="progress-track" style={{ marginTop: '4px' }}>
          <div
            className="progress-fill"
            style={{
              width: `${latencyPct}%`,
              background: provider.avg_latency_ms > 1000
                ? 'var(--accent-warning)'
                : 'var(--accent-success)',
            }}
          />
        </div>
      )}
    </div>
  )
}

function CostBar({ label, value, maxVal, color }: { label: string; value: number; maxVal: number; color: string }) {
  const pct = maxVal > 0 ? (value / maxVal) * 100 : 0
  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px' }}>
        <span>{label}</span>
        <span className="mono" style={{ color: 'var(--text-secondary)' }}>{fmtCost(value)}</span>
      </div>
      <div className="progress-track" style={{ height: '8px' }}>
        <div
          className="progress-fill"
          style={{
            width: `${pct}%`,
            minWidth: value > 0 ? '4px' : '0',
            background: color,
          }}
        />
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
    } catch {
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
    <div className="view">
      {/* Summary header */}
      <div className="card-static" style={{ marginBottom: '20px' }}>
        {loading ? (
          <div style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Lade Daten...</div>
        ) : error ? (
          <div style={{ color: 'var(--accent-danger)', fontSize: '14px' }}>{error}</div>
        ) : costSummary ? (
          <div className="provider-summary">
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Gesamtkosten</div>
              <div className="stat-value mono" style={{ fontSize: '28px', color: 'var(--accent-primary)' }}>
                {fmtCost(costSummary.total_cost_usd)}
              </div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Requests</div>
              <div className="stat-value mono" style={{ fontSize: '22px' }}>
                {costSummary.total_requests}
              </div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Tokens (est.)</div>
              <div className="stat-value mono" style={{ fontSize: '22px' }}>
                ~{fmtTokens(providers?.providers.reduce((s, p) => s + p.total_tokens_estimate, 0) ?? 0)}
              </div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: '4px' }}>Gespart vs. Cloud</div>
              <div className="stat-value mono" style={{ fontSize: '22px', color: 'var(--accent-success)' }}>
                ~{fmtCost(costSummary.savings_vs_all_cloud_usd)}
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Provider cards */}
      {providers && (
        <div className="grid-3" style={{ marginBottom: '24px' }}>
          {providers.providers.map(p => (
            <ProviderCard key={p.name} provider={p} maxLatency={maxLatency} />
          ))}
        </div>
      )}

      {/* Cost breakdown */}
      {costSummary && (
        <div className="card-static">
          <h3 className="heading" style={{ fontSize: '14px', marginBottom: '16px' }}>Kostenverteilung</h3>
          {(() => {
            const max = Math.max(
              costSummary.groq_cost_usd,
              costSummary.cloud_cost_usd,
              costSummary.local_cost_usd,
              0.0001
            )
            return (
              <>
                <CostBar label="Groq (Free)" value={costSummary.groq_cost_usd} maxVal={max} color="var(--accent-info)" />
                <CostBar label="Cloud LLM" value={costSummary.cloud_cost_usd} maxVal={max} color="var(--accent-warning)" />
                <CostBar label="QO-LLM (IGQK)" value={costSummary.local_cost_usd} maxVal={max} color="var(--accent-success)" />
              </>
            )
          })()}
        </div>
      )}

      <style>{`
        .provider-card {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .provider-icon {
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-md);
          background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
          color: var(--accent-primary);
        }
        .provider-stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }
        .provider-summary {
          display: flex;
          flex-wrap: wrap;
          gap: 32px;
          align-items: center;
        }
      `}</style>
    </div>
  )
}
