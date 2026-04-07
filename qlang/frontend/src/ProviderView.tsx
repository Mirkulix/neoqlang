import { useEffect, useState } from 'react'
import { Server, Plus, Trash2, Power, Zap, Key, Check, X, Edit3 } from 'lucide-react'

// ─── Types ─────────────────────────────────────────────────────────────────

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

interface ModelOption {
  id: string
  name: string
  cost_per_1k: number
  recommended: boolean
}

interface ProviderTemplate {
  id: string
  name: string
  provider_type: string
  base_url: string
  models: ModelOption[]
  tier: number
  free: boolean
  description: string
}

interface ConfiguredProvider {
  id: string
  name: string
  provider_type: string
  model: string
  base_url: string | null
  enabled: boolean
  tier: number
  cost_per_1k_tokens: number
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const fmtCost = (usd: number): string => `$${usd.toFixed(4)}`
const fmtTokens = (n: number): string => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

const statusBadge: Record<string, { cls: string; label: string }> = {
  active:      { cls: 'badge-idle', label: 'Aktiv' },
  inactive:    { cls: 'badge-info', label: 'Inaktiv' },
  coming_soon: { cls: 'badge-pending', label: 'Bald' },
  error:       { cls: 'badge-error', label: 'Fehler' },
}

// ─── Sub-components ────────────────────────────────────────────────────────

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

// ─── Configured provider card ──────────────────────────────────────────────

function ConfiguredCard({
  provider,
  onToggle,
  onDelete,
  onTest,
  onUpdate,
}: {
  provider: ConfiguredProvider
  onToggle: (id: string) => void
  onDelete: (id: string) => void
  onTest: (id: string) => void
  onUpdate: () => void
}) {
  const [testing, setTesting] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editKey, setEditKey] = useState('')
  const [editModel, setEditModel] = useState(provider.model)
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; latency_ms: number; message: string } | null>(null)

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await fetch('/api/providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: provider.id }),
      })
      const data = await res.json()
      setTestResult(data)
    } catch {
      setTestResult({ success: false, latency_ms: 0, message: 'Netzwerkfehler' })
    } finally {
      setTesting(false)
      onTest(provider.id)
    }
  }

  return (
    <div className="card provider-card" style={{ opacity: provider.enabled ? 1 : 0.6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="provider-icon">
            <Server size={20} />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 600 }}>{provider.name}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              {provider.model}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          {provider.cost_per_1k_tokens === 0 && (
            <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '999px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', fontWeight: 600 }}>
              Kostenlos
            </span>
          )}
          <span className={`badge ${provider.enabled ? 'badge-idle' : 'badge-info'}`}>
            <span className="badge-dot" />
            {provider.enabled ? 'Aktiv' : 'Inaktiv'}
          </span>
        </div>
      </div>

      {testResult && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '12px',
          padding: '6px 10px',
          borderRadius: 'var(--radius-sm)',
          background: testResult.success
            ? 'color-mix(in srgb, var(--accent-success) 10%, transparent)'
            : 'color-mix(in srgb, var(--accent-danger) 10%, transparent)',
          color: testResult.success ? 'var(--accent-success)' : 'var(--accent-danger)',
        }}>
          {testResult.success ? <Check size={12} /> : <X size={12} />}
          {testResult.success ? `OK — ${testResult.latency_ms}ms` : testResult.message}
        </div>
      )}

      {editing && (
        <div style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div>
            <label className="label" style={{ display: 'block', marginBottom: '4px' }}>API Key</label>
            <div style={{ display: 'flex', gap: '6px' }}>
              <input
                className="input"
                type={showKey ? 'text' : 'password'}
                value={editKey}
                onChange={e => setEditKey(e.target.value)}
                placeholder="Neuer API Key (leer = unverändert)"
                style={{ flex: 1 }}
              />
              <button className="btn btn-ghost" style={{ minHeight: '36px', padding: '0 10px' }} onClick={() => setShowKey(k => !k)}>
                {showKey ? <X size={14} /> : <Key size={14} />}
              </button>
            </div>
          </div>
          <div>
            <label className="label" style={{ display: 'block', marginBottom: '4px' }}>Modell</label>
            <input
              className="input"
              value={editModel}
              onChange={e => setEditModel(e.target.value)}
              placeholder="Modell-ID"
            />
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn btn-primary"
              style={{ flex: 1, fontSize: '12px' }}
              disabled={saving}
              onClick={async () => {
                setSaving(true)
                const body: Record<string, string> = {}
                if (editKey) body.api_key = editKey
                if (editModel !== provider.model) body.model = editModel
                if (Object.keys(body).length > 0) {
                  await fetch(`/api/providers/${provider.id}/edit`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                  })
                }
                setSaving(false)
                setEditing(false)
                setEditKey('')
                onUpdate()
              }}
            >
              <Check size={13} /> Speichern
            </button>
            <button className="btn btn-ghost" style={{ fontSize: '12px' }} onClick={() => { setEditing(false); setEditKey('') }}>
              Abbrechen
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
        <button
          className="btn-ghost"
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '12px' }}
          onClick={handleTest}
          disabled={testing}
        >
          <Zap size={13} />
          {testing ? 'Teste…' : 'Testen'}
        </button>
        <button
          className="btn-ghost"
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '12px' }}
          onClick={() => { setEditing(e => !e); setEditModel(provider.model) }}
        >
          <Edit3 size={13} />
          Bearbeiten
        </button>
        <button
          className="btn-ghost"
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '12px' }}
          onClick={() => onToggle(provider.id)}
        >
          <Power size={13} />
          {provider.enabled ? 'Aus' : 'An'}
        </button>
        <button
          className="btn-ghost"
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6px 10px', color: 'var(--accent-danger)' }}
          onClick={() => onDelete(provider.id)}
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  )
}

// ─── Template card ─────────────────────────────────────────────────────────

function TemplateCard({
  template,
  selected,
  onSelect,
}: {
  template: ProviderTemplate
  selected: boolean
  onSelect: (t: ProviderTemplate) => void
}) {
  return (
    <div
      className="card"
      style={{
        cursor: 'pointer',
        border: selected ? '1px solid var(--accent-primary)' : '1px solid var(--border)',
        background: selected ? 'color-mix(in srgb, var(--accent-primary) 6%, var(--bg-card))' : 'var(--bg-card)',
        transition: 'border 0.15s, background 0.15s',
      }}
      onClick={() => onSelect(template)}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
        <div style={{ fontWeight: 600, fontSize: '14px' }}>{template.name}</div>
        {template.free && (
          <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '999px', background: 'color-mix(in srgb, var(--accent-success) 15%, transparent)', color: 'var(--accent-success)', fontWeight: 600, whiteSpace: 'nowrap' }}>
            Kostenlos
          </span>
        )}
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        {template.description}
      </div>
    </div>
  )
}

// ─── Add Provider Panel ─────────────────────────────────────────────────────

function AddProviderPanel({
  templates,
  onAdded,
  onClose,
}: {
  templates: ProviderTemplate[]
  onAdded: () => void
  onClose: () => void
}) {
  const sorted = [...templates].sort((a, b) => {
    if (a.free && !b.free) return -1
    if (!a.free && b.free) return 1
    return a.tier - b.tier
  })

  const [selected, setSelected] = useState<ProviderTemplate | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [model, setModel] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; latency_ms: number; message: string } | null>(null)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<string | null>(null)

  const selectTemplate = (t: ProviderTemplate) => {
    setSelected(t)
    const rec = t.models.find(m => m.recommended) ?? t.models[0]
    setModel(rec?.id ?? '')
    setApiKey('')
    setTestResult(null)
    setAddError(null)
  }

  const handleTest = async () => {
    if (!selected) return
    // First save a temporary config to test
    await fetch('/api/providers/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ template_id: selected.id, api_key: apiKey, model }),
    })
    setTesting(true)
    setTestResult(null)
    try {
      const res = await fetch('/api/providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: selected.id }),
      })
      const data = await res.json()
      setTestResult(data)
    } catch {
      setTestResult({ success: false, latency_ms: 0, message: 'Netzwerkfehler' })
    } finally {
      setTesting(false)
    }
  }

  const handleAdd = async () => {
    if (!selected) return
    setAdding(true)
    setAddError(null)
    try {
      const res = await fetch('/api/providers/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template_id: selected.id, api_key: apiKey, model }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      onAdded()
      onClose()
    } catch (e: unknown) {
      setAddError(e instanceof Error ? e.message : 'Fehler beim Hinzufügen')
    } finally {
      setAdding(false)
    }
  }

  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-card)',
      padding: '20px',
      marginTop: '16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 className="heading" style={{ fontSize: '14px', margin: 0 }}>Provider hinzufügen</h3>
        <button className="btn-ghost" onClick={onClose} style={{ padding: '4px 8px' }}>
          <X size={16} />
        </button>
      </div>

      {/* Template grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '10px', marginBottom: '20px' }}>
        {sorted.map(t => (
          <TemplateCard
            key={t.id}
            template={t}
            selected={selected?.id === t.id}
            onSelect={selectTemplate}
          />
        ))}
      </div>

      {/* Config form */}
      {selected && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px' }}>
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>
            {selected.name} konfigurieren
          </div>

          {/* API Key */}
          {selected.id !== 'ollama' && (
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                <Key size={12} style={{ display: 'inline', marginRight: '4px' }} />
                API Key
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  placeholder={`${selected.name} API Key`}
                  style={{
                    flex: 1,
                    padding: '7px 10px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)',
                    background: 'var(--bg-input, var(--bg-card))',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                    fontFamily: 'monospace',
                  }}
                />
                <button
                  className="btn-ghost"
                  onClick={() => setShowKey(v => !v)}
                  style={{ padding: '7px 10px', fontSize: '11px' }}
                >
                  {showKey ? 'Verstecken' : 'Zeigen'}
                </button>
              </div>
            </div>
          )}

          {/* Model */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
              Modell
            </label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              style={{
                width: '100%',
                padding: '7px 10px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
                background: 'var(--bg-input, var(--bg-card))',
                color: 'var(--text-primary)',
                fontSize: '13px',
              }}
            >
              {selected.models.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name}{m.recommended ? ' (empfohlen)' : ''}{m.cost_per_1k === 0 ? ' — kostenlos' : ` — $${m.cost_per_1k}/1K tok`}
                </option>
              ))}
            </select>
          </div>

          {/* Test result */}
          {testResult && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              padding: '8px 12px',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '12px',
              background: testResult.success
                ? 'color-mix(in srgb, var(--accent-success) 10%, transparent)'
                : 'color-mix(in srgb, var(--accent-danger) 10%, transparent)',
              color: testResult.success ? 'var(--accent-success)' : 'var(--accent-danger)',
            }}>
              {testResult.success ? <Check size={14} /> : <X size={14} />}
              {testResult.success
                ? `Verbindung erfolgreich — ${testResult.latency_ms}ms`
                : `Fehler: ${testResult.message}`}
            </div>
          )}

          {addError && (
            <div style={{ fontSize: '12px', color: 'var(--accent-danger)', marginBottom: '12px' }}>
              {addError}
            </div>
          )}

          {/* Actions */}
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              className="btn-ghost"
              onClick={handleTest}
              disabled={testing || (selected.id !== 'ollama' && !apiKey)}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}
            >
              <Zap size={14} />
              {testing ? 'Teste…' : 'Testen'}
            </button>
            <button
              className="btn-primary"
              onClick={handleAdd}
              disabled={adding || (selected.id !== 'ollama' && !apiKey)}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', marginLeft: 'auto' }}
            >
              <Plus size={14} />
              {adding ? 'Wird hinzugefügt…' : 'Hinzufügen'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main View ─────────────────────────────────────────────────────────────

export default function ProviderView() {
  const [providers, setProviders] = useState<ProvidersResponse | null>(null)
  const [costSummary, setCostSummary] = useState<CostSummary | null>(null)
  const [configured, setConfigured] = useState<ConfiguredProvider[]>([])
  const [templates, setTemplates] = useState<ProviderTemplate[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [showAdd, setShowAdd] = useState(false)

  const fetchData = async () => {
    try {
      const [provRes, costRes, confRes, tplRes] = await Promise.all([
        fetch('/api/providers'),
        fetch('/api/providers/costs'),
        fetch('/api/providers/configured'),
        fetch('/api/providers/templates'),
      ])
      if (!provRes.ok || !costRes.ok) throw new Error('API error')
      const [prov, cost, conf, tpls] = await Promise.all([
        provRes.json(), costRes.json(), confRes.json(), tplRes.json(),
      ])
      setProviders(prov)
      setCostSummary(cost)
      setConfigured(Array.isArray(conf) ? conf : [])
      setTemplates(Array.isArray(tpls) ? tpls : [])
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

  const handleToggle = async (id: string) => {
    await fetch(`/api/providers/${id}/toggle`, { method: 'PUT' })
    fetchData()
  }

  const handleDelete = async (id: string) => {
    await fetch(`/api/providers/${id}`, { method: 'DELETE' })
    fetchData()
  }

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

      {/* Active providers (runtime stats) */}
      {providers && (
        <div className="grid-3" style={{ marginBottom: '24px' }}>
          {providers.providers.map(p => (
            <ProviderCard key={p.name} provider={p} maxLatency={maxLatency} />
          ))}
        </div>
      )}

      {/* Configured providers section */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
          <h3 className="heading" style={{ fontSize: '14px', margin: 0 }}>
            Konfigurierte Provider ({configured.length})
          </h3>
          <button
            className="btn-primary"
            onClick={() => setShowAdd(v => !v)}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', padding: '7px 14px' }}
          >
            <Plus size={14} />
            Provider hinzufügen
          </button>
        </div>

        {configured.length > 0 && (
          <div className="grid-3" style={{ marginBottom: showAdd ? '0' : '0' }}>
            {configured.map(p => (
              <ConfiguredCard
                key={p.id}
                provider={p}
                onToggle={handleToggle}
                onDelete={handleDelete}
                onTest={() => {}}
                onUpdate={fetchData}
              />
            ))}
          </div>
        )}

        {configured.length === 0 && !showAdd && (
          <div style={{ color: 'var(--text-secondary)', fontSize: '13px', padding: '12px 0' }}>
            Noch keine Provider konfiguriert. Klicke "Provider hinzufügen" um zu starten.
          </div>
        )}

        {showAdd && (
          <AddProviderPanel
            templates={templates}
            onAdded={fetchData}
            onClose={() => setShowAdd(false)}
          />
        )}
      </div>

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
        .btn-ghost {
          background: transparent;
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          color: var(--text-primary);
          cursor: pointer;
          padding: 6px 12px;
          font-size: 13px;
          transition: background 0.15s;
        }
        .btn-ghost:hover {
          background: color-mix(in srgb, var(--accent-primary) 8%, transparent);
        }
        .btn-ghost:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .btn-primary {
          background: var(--accent-primary);
          border: none;
          border-radius: var(--radius-sm);
          color: #fff;
          cursor: pointer;
          padding: 6px 14px;
          font-size: 13px;
          font-weight: 600;
          transition: opacity 0.15s;
        }
        .btn-primary:hover {
          opacity: 0.88;
        }
        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  )
}
