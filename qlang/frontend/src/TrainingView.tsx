import { useState, useEffect } from 'react'
import { Cpu, Play, Download, BarChart3, Activity } from 'lucide-react'

interface TrainProgress {
  epoch: number
  total_epochs: number
  f32_accuracy: number
  ternary_accuracy: number
  pos_goodness: number
  neg_goodness: number
  elapsed_secs: number
  status: string
}

interface TrainResult {
  success: boolean
  f32_accuracy: number
  ternary_accuracy: number
  total_params: number
  f32_size_kb: number
  ternary_size_kb: number
  compression_ratio: number
  train_time_secs: number
  model_file: string | null
  epochs: TrainProgress[]
}

export default function TrainingView() {
  const [dataset, setDataset] = useState('mnist')
  const [epochs, setEpochs] = useState(10)
  const [samples, setSamples] = useState(10000)
  const [hidden, setHidden] = useState('256, 128')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<TrainResult | null>(null)

  const startTraining = async () => {
    setRunning(true)
    setResult(null)
    try {
      const hiddenLayers = hidden.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
      const res = await fetch('/api/training/qlang', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset,
          epochs,
          train_samples: samples,
          hidden_layers: hiddenLayers,
        }),
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setResult({
        success: false, f32_accuracy: 0, ternary_accuracy: 0, total_params: 0,
        f32_size_kb: 0, ternary_size_kb: 0, compression_ratio: 0, train_time_secs: 0,
        model_file: null,
        epochs: [{ epoch: 0, total_epochs: 0, f32_accuracy: 0, ternary_accuracy: 0,
          pos_goodness: 0, neg_goodness: 0, elapsed_secs: 0, status: String(e) }],
      })
    }
    setRunning(false)
  }

  return (
    <div className="view" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Config */}
      <div className="card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <Cpu size={18} />
          <h3 className="heading" style={{ margin: 0, fontSize: '14px' }}>QLANG Ternary Training</h3>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
            Forward-Forward, kein Backprop, {'{-1, 0, +1}'} Gewichte
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          <div>
            <label style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Dataset</label>
            <select value={dataset} onChange={e => setDataset(e.target.value)}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid var(--border)',
                background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }}>
              <option value="mnist">MNIST (echte Handschrift)</option>
              <option value="synthetic">Synthetisch (schnell)</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Epochen</label>
            <input type="number" value={epochs} onChange={e => setEpochs(parseInt(e.target.value) || 10)} min={1} max={100}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid var(--border)',
                background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </div>
          <div>
            <label style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Train Samples</label>
            <input type="number" value={samples} onChange={e => setSamples(parseInt(e.target.value) || 5000)} min={100} max={60000} step={1000}
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid var(--border)',
                background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </div>
          <div>
            <label style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Hidden Layers</label>
            <input type="text" value={hidden} onChange={e => setHidden(e.target.value)} placeholder="256, 128"
              style={{ width: '100%', padding: '8px', borderRadius: '8px', border: '1px solid var(--border)',
                background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
          </div>
        </div>

        <button className="btn btn-primary" onClick={startTraining} disabled={running}
          style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 24px', fontSize: '14px' }}>
          <Play size={16} />
          {running ? 'Training laeuft...' : 'Training starten'}
        </button>
      </div>

      {/* Progress */}
      {running && (
        <div className="card" style={{ padding: '20px', textAlign: 'center' }}>
          <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
            Training laeuft auf CPU...
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {epochs} Epochen, {samples} Samples. Geschaetzte Zeit: ~{Math.round(epochs * samples / 1500)}s
          </div>
          <div className="progress-track" style={{ marginTop: '12px', height: '6px' }}>
            <div className="progress-fill" style={{ width: '100%', background: 'var(--accent-primary)',
              animation: 'pulse 2s ease-in-out infinite' }} />
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>
                {(result.ternary_accuracy * 100).toFixed(1)}%
              </div>
              <div className="label">Ternary Accuracy</div>
            </div>
            <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-success)', fontFamily: 'var(--font-mono)' }}>
                {result.compression_ratio.toFixed(1)}x
              </div>
              <div className="label">Kompression</div>
            </div>
            <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-info)', fontFamily: 'var(--font-mono)' }}>
                {result.ternary_size_kb.toFixed(1)} KB
              </div>
              <div className="label">Modell-Groesse</div>
            </div>
            <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {result.train_time_secs.toFixed(1)}s
              </div>
              <div className="label">Trainingszeit</div>
            </div>
          </div>

          {/* Epoch details */}
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <BarChart3 size={16} />
              <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>Training Progress</h3>
            </div>

            {/* ASCII chart */}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', lineHeight: 1.6, overflowX: 'auto' }}>
              <div style={{ display: 'flex', gap: '8px', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', paddingBottom: '4px', marginBottom: '4px' }}>
                <span style={{ width: '50px' }}>Epoch</span>
                <span style={{ width: '60px' }}>f32</span>
                <span style={{ width: '60px' }}>Ternary</span>
                <span style={{ flex: 1 }}>Progress</span>
              </div>
              {result.epochs.map(ep => {
                const barLen = Math.round(ep.ternary_accuracy * 40)
                const f32BarLen = Math.round(ep.f32_accuracy * 40)
                return (
                  <div key={ep.epoch} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <span style={{ width: '50px', color: 'var(--text-muted)' }}>{ep.epoch}/{ep.total_epochs}</span>
                    <span style={{ width: '60px', color: 'var(--accent-secondary)' }}>{(ep.f32_accuracy * 100).toFixed(1)}%</span>
                    <span style={{ width: '60px', color: 'var(--accent-primary)' }}>{(ep.ternary_accuracy * 100).toFixed(1)}%</span>
                    <span style={{ flex: 1 }}>
                      <span style={{ color: 'var(--accent-secondary)' }}>{'█'.repeat(f32BarLen)}</span>
                      <span style={{ color: 'var(--accent-primary)' }}>{'▓'.repeat(Math.max(0, barLen))}</span>
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Model details */}
          <div className="card" style={{ padding: '20px' }}>
            <div style={{ fontSize: '13px', lineHeight: 2 }}>
              <div><strong>Method:</strong> Forward-Forward Ternary (kein Backprop)</div>
              <div><strong>Params:</strong> {result.total_params.toLocaleString()}</div>
              <div><strong>f32 Accuracy:</strong> {(result.f32_accuracy * 100).toFixed(1)}%</div>
              <div><strong>f32 Size:</strong> {result.f32_size_kb.toFixed(1)} KB</div>
              <div><strong>Ternary Size:</strong> {result.ternary_size_kb.toFixed(1)} KB</div>
              {result.model_file && (
                <div style={{ marginTop: '8px' }}>
                  <strong>Modell:</strong> <code style={{ fontSize: '12px' }}>{result.model_file}</code>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Live Monitor */}
      <LiveMonitor />

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}

// ============================================================
// Live Training Monitor — polls /api/training/monitor every 3s
// ============================================================

interface MonitorEpoch {
  epoch: number
  f32_accuracy: number
  ternary_accuracy: number
  pos_goodness: number
  neg_goodness: number
}

interface MonitorData {
  running: boolean
  epochs: MonitorEpoch[]
  current_epoch: number
  total_epochs: number
  best_f32: number
  best_ternary: number
}

function LiveMonitor() {
  const [data, setData] = useState<MonitorData | null>(null)

  useEffect(() => {
    const poll = () => {
      fetch('/api/training/monitor')
        .then(r => r.json())
        .then(setData)
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [])

  if (!data || data.epochs.length === 0) {
    return (
      <div className="card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Activity size={16} />
          <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>Live Training Monitor</h3>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
          Kein aktives Training. Starte mit <code>qlang train</code> oder dem Button oben.
        </div>
      </div>
    )
  }

  const progress = data.total_epochs > 0 ? (data.current_epoch / data.total_epochs) * 100 : 0
  const target90 = data.best_ternary >= 0.90

  return (
    <div className="card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Activity size={16} style={{ color: data.running ? 'var(--accent-success)' : 'var(--text-muted)' }} />
        <h3 className="heading" style={{ margin: 0, fontSize: '13px' }}>
          Live Training Monitor {data.running ? '(laeuft)' : '(beendet)'}
        </h3>
        <span style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-muted)' }}>
          Epoch {data.current_epoch}/{data.total_epochs}
        </span>
      </div>

      {/* Progress bar */}
      <div className="progress-track" style={{ height: '8px', marginBottom: '16px' }}>
        <div className="progress-fill" style={{
          width: `${progress}%`,
          background: target90 ? 'var(--accent-success)' : 'var(--accent-primary)',
          transition: 'width 0.5s ease',
        }} />
      </div>

      {/* Key metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '16px' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: '28px', fontWeight: 700, fontFamily: 'var(--font-mono)',
            color: target90 ? 'var(--accent-success)' : 'var(--accent-primary)',
          }}>
            {(data.best_ternary * 100).toFixed(1)}%
          </div>
          <div className="label">Best Ternary {target90 ? '(Ziel erreicht!)' : ''}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-secondary)' }}>
            {(data.best_f32 * 100).toFixed(1)}%
          </div>
          <div className="label">Best f32</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '28px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {data.epochs.length}
          </div>
          <div className="label">Epochen trainiert</div>
        </div>
      </div>

      {/* Accuracy chart (ASCII) */}
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: 1.5, maxHeight: '300px', overflowY: 'auto' }}>
        <div style={{ display: 'flex', gap: '6px', color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', paddingBottom: '2px', marginBottom: '4px' }}>
          <span style={{ width: '45px' }}>Epoch</span>
          <span style={{ width: '55px' }}>f32</span>
          <span style={{ width: '55px' }}>Ternary</span>
          <span style={{ width: '45px' }}>pg</span>
          <span style={{ width: '45px' }}>ng</span>
          <span style={{ flex: 1 }}>Ternary Accuracy</span>
        </div>
        {data.epochs.map(ep => {
          const barLen = Math.round(ep.ternary_accuracy * 50)
          const is90 = ep.ternary_accuracy >= 0.90
          return (
            <div key={ep.epoch} style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <span style={{ width: '45px', color: 'var(--text-muted)' }}>{ep.epoch}</span>
              <span style={{ width: '55px', color: 'var(--accent-secondary)' }}>{(ep.f32_accuracy * 100).toFixed(1)}%</span>
              <span style={{ width: '55px', color: is90 ? 'var(--accent-success)' : 'var(--accent-primary)' }}>
                {(ep.ternary_accuracy * 100).toFixed(1)}%
              </span>
              <span style={{ width: '45px', color: 'var(--text-muted)' }}>{ep.pos_goodness.toFixed(1)}</span>
              <span style={{ width: '45px', color: 'var(--text-muted)' }}>{ep.neg_goodness.toFixed(1)}</span>
              <span style={{ flex: 1, color: is90 ? 'var(--accent-success)' : 'var(--accent-primary)' }}>
                {'█'.repeat(barLen)}{is90 ? ' ✓' : ''}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
