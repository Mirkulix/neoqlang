import React, { useState, useEffect, useRef } from 'react'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tier?: string
  timestamp?: string
}

interface ChatEntry {
  id?: string
  role: string
  content: string
  tier?: string
  timestamp?: string
}

interface ChatResponse {
  response: string
  tier?: string
}

export default function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    fetch('/api/chat/history')
      .then(r => r.json())
      .then((history: ChatEntry[]) => {
        if (Array.isArray(history)) {
          setMessages(history.map((entry, i) => ({
            id: entry.id ?? String(i),
            role: entry.role === 'user' ? 'user' : 'assistant',
            content: entry.content,
            tier: entry.tier,
            timestamp: entry.timestamp,
          })))
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const data: ChatResponse = await res.json()
      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        tier: data.tier,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Fehler: Verbindung zum Backend fehlgeschlagen.',
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Message list */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}>
        {messages.length === 0 && !loading && (
          <div style={{ textAlign: 'center', color: '#484f58', marginTop: '60px', fontSize: '15px' }}>
            Starte eine Unterhaltung mit QO
          </div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            <div
              style={{
                maxWidth: '70%',
                padding: '10px 14px',
                borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                background: msg.role === 'user' ? '#1f6feb' : '#161b22',
                color: '#e0e0e0',
                fontSize: '14px',
                lineHeight: '1.5',
                wordBreak: 'break-word',
                border: msg.role === 'assistant' ? '1px solid #21262d' : 'none',
              }}
            >
              {msg.content}
            </div>
            {msg.role === 'assistant' && msg.tier && (
              <span style={{
                fontSize: '11px',
                color: '#484f58',
                marginTop: '4px',
                marginLeft: '4px',
                background: '#161b22',
                border: '1px solid #21262d',
                padding: '1px 6px',
                borderRadius: '4px',
              }}>
                {msg.tier}
              </span>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <div style={{
              padding: '10px 14px',
              borderRadius: '16px 16px 16px 4px',
              background: '#161b22',
              border: '1px solid #21262d',
              display: 'flex',
              gap: '4px',
              alignItems: 'center',
            }}>
              <span style={{ animation: 'pulse 1.2s ease-in-out infinite', color: '#7fdbca', fontSize: '20px', lineHeight: 1 }}>•</span>
              <span style={{ animation: 'pulse 1.2s ease-in-out 0.2s infinite', color: '#7fdbca', fontSize: '20px', lineHeight: 1 }}>•</span>
              <span style={{ animation: 'pulse 1.2s ease-in-out 0.4s infinite', color: '#7fdbca', fontSize: '20px', lineHeight: 1 }}>•</span>
              <style>{`@keyframes pulse { 0%,100%{opacity:0.3} 50%{opacity:1} }`}</style>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div style={{
        padding: '16px 20px',
        borderTop: '1px solid #21262d',
        background: '#161b22',
        display: 'flex',
        gap: '10px',
        alignItems: 'flex-end',
      }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nachricht eingeben... (Enter zum Senden, Shift+Enter für Zeilenumbruch)"
          rows={1}
          style={{
            flex: 1,
            background: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: '8px',
            padding: '10px 14px',
            color: '#e0e0e0',
            fontSize: '14px',
            resize: 'none',
            outline: 'none',
            fontFamily: 'inherit',
            lineHeight: '1.5',
            maxHeight: '120px',
            overflowY: 'auto',
            transition: 'border-color 0.15s',
          }}
          onFocus={e => { e.target.style.borderColor = '#7fdbca' }}
          onBlur={e => { e.target.style.borderColor = '#30363d' }}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          style={{
            padding: '10px 18px',
            background: loading || !input.trim() ? '#21262d' : '#7fdbca',
            color: loading || !input.trim() ? '#484f58' : '#0d1117',
            border: 'none',
            borderRadius: '8px',
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 600,
            transition: 'all 0.15s',
            flexShrink: 0,
          }}
        >
          Senden
        </button>
      </div>
    </div>
  )
}
