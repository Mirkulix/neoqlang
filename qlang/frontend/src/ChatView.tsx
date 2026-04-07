import { useState, useEffect, useRef } from 'react'
import { Send } from 'lucide-react'

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

function LoadingDots() {
  return (
    <div className="chat-bubble chat-bubble-assistant">
      <span className="loading-dot" style={{ animationDelay: '0ms' }} />
      <span className="loading-dot" style={{ animationDelay: '200ms' }} />
      <span className="loading-dot" style={{ animationDelay: '400ms' }} />
    </div>
  )
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
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-title">Starte eine Konversation...</div>
            <div className="empty-hint">Schreibe eine Nachricht, um mit QO zu kommunizieren.</div>
          </div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`chat-row ${msg.role === 'user' ? 'chat-row-user' : 'chat-row-assistant'}`}
          >
            <div className={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}`}>
              {msg.content}
            </div>
            {msg.role === 'assistant' && msg.tier && (
              <span className="chat-tier-badge">{msg.tier}</span>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-row chat-row-assistant">
            <LoadingDots />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          className="input chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nachricht eingeben... (Enter zum Senden)"
          rows={1}
        />
        <button
          className="btn btn-primary chat-send-btn"
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          aria-label="Senden"
        >
          <Send size={18} />
        </button>
      </div>

      <style>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .chat-row {
          display: flex;
          flex-direction: column;
          animation: fadeIn 300ms ease-out;
        }
        .chat-row-user {
          align-items: flex-end;
        }
        .chat-row-assistant {
          align-items: flex-start;
        }
        .chat-bubble {
          max-width: 70%;
          padding: 12px 16px;
          font-size: 14px;
          line-height: 1.6;
          word-break: break-word;
        }
        .chat-bubble-user {
          background: var(--accent-primary);
          color: #ffffff;
          border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
        }
        .chat-bubble-assistant {
          background: var(--bg-surface);
          color: var(--text-primary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .chat-tier-badge {
          font-size: 10px;
          color: var(--text-muted);
          background: var(--bg-surface);
          border: 1px solid var(--border);
          padding: 1px 8px;
          border-radius: 12px;
          margin-top: 4px;
          margin-left: 4px;
        }
        .chat-input-area {
          padding: 16px 24px;
          border-top: 1px solid var(--border);
          background: var(--bg-surface);
          display: flex;
          gap: 12px;
          align-items: flex-end;
        }
        .chat-input {
          flex: 1;
          max-height: 120px;
          overflow-y: auto;
        }
        .chat-send-btn {
          flex-shrink: 0;
          min-width: 44px;
          padding: 8px 16px;
        }
        .loading-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--accent-primary);
          display: inline-block;
          animation: dotPulse 1.2s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
}
