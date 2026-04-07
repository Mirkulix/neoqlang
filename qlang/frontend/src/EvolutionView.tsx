import React from 'react'

export default function EvolutionView() {
  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '24px 20px' }}>
      <div style={{
        background: '#161b22',
        border: '1px solid #21262d',
        borderRadius: '12px',
        padding: '40px 32px',
        maxWidth: '560px',
        margin: '0 auto',
        textAlign: 'center',
      }}>
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>🧬</div>
        <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#7fdbca', margin: '0 0 12px' }}>
          Evolution — Phase 4
        </h2>
        <p style={{ fontSize: '14px', color: '#8b949e', lineHeight: '1.7', margin: 0 }}>
          Pattern-Erkennung und autonome Vorschläge werden in einer zukünftigen Version aktiviert.
        </p>

        <div style={{ marginTop: '32px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {['Muster-Erkennung', 'Autonome Verbesserungsvorschläge', 'Selbst-Optimierung', 'Langzeit-Gedächtnis'].map(feature => (
            <div key={feature} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 14px',
              background: '#0d1117',
              borderRadius: '8px',
              border: '1px solid #21262d',
            }}>
              <span style={{
                width: '8px', height: '8px',
                borderRadius: '50%',
                background: '#21262d',
                flexShrink: 0,
              }} />
              <span style={{ fontSize: '13px', color: '#484f58' }}>{feature}</span>
              <span style={{
                marginLeft: 'auto',
                fontSize: '10px',
                color: '#484f58',
                background: '#21262d',
                padding: '2px 8px',
                borderRadius: '8px',
              }}>Bald</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
