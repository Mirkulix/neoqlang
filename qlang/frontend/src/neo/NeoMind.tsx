// NeoMind — holographic visualization of the QLANG Organism.
// Data sources (real API only, no fake data):
//   GET  /api/organism/status    polled every 1000ms
//   GET  /api/messages/stream    SSE bus (id, from, to, intent, ...)
// Pure React + inline SVG, custom force sim @ 60fps, no external deps.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface ApiSpecialist { name: string; invocations: number; success_rate: number }
interface OrganismStatus {
  generation: number;
  specialists: ApiSpecialist[];
  total_interactions: number;
  memory_items: number;
}
interface BusMessage {
  id: number; from: string; to: string; intent: string; graph_name: string; timestamp: number;
}
interface SimNode {
  id: string; label: string; role: string; isOrchestrator: boolean;
  invocations: number; successRate: number; lastActivity: number;
  x: number; y: number; vx: number; vy: number; fitness: number;
}
interface SimEdge { from: string; to: string; birth: number; intent: string }
interface DetailState { id: string; messages: BusMessage[] }

const ROLE_COLORS: Record<string, string> = {
  orchestrator: "#00e5ff", classifier: "#ff2bd6", memory: "#b388ff",
  language: "#ffd54a", math: "#64ffda", generic: "#7dffb0",
};
function roleOf(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("orchestr")) return "orchestrator";
  if (n.includes("class") || n.includes("topic")) return "classifier";
  if (n.includes("mem") || n.includes("recall")) return "memory";
  if (n.includes("lm") || n.includes("mamba") || n.includes("lang")) return "language";
  if (n.includes("math") || n.includes("calc")) return "math";
  return "generic";
}
const colorFor = (role: string) => ROLE_COLORS[role] ?? ROLE_COLORS.generic;

const WIDTH = 960, HEIGHT = 620, CX = WIDTH / 2, CY = HEIGHT / 2;

function stepSim(nodes: SimNode[], edges: SimEdge[], dt: number) {
  const REPULSION = 9000, CENTER_PULL = 0.008, ORCH_PULL = 0.012;
  const DAMPING = 0.82, EDGE_SPRING = 0.02, EDGE_REST = 180;
  // Pairwise repulsion
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const d2 = dx * dx + dy * dy + 0.01;
      const d = Math.sqrt(d2);
      const f = REPULSION / d2;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      if (!a.isOrchestrator) { a.vx += fx * dt; a.vy += fy * dt; }
      if (!b.isOrchestrator) { b.vx -= fx * dt; b.vy -= fy * dt; }
    }
  }
  const orch = nodes.find(n => n.isOrchestrator);
  if (orch) { orch.x = CX; orch.y = CY; orch.vx = 0; orch.vy = 0; }
  for (const n of nodes) {
    if (n.isOrchestrator) continue;
    n.vx += (CX - n.x) * CENTER_PULL * dt;
    n.vy += (CY - n.y) * CENTER_PULL * dt;
    if (orch) {
      n.vx += (orch.x - n.x) * ORCH_PULL * dt;
      n.vy += (orch.y - n.y) * ORCH_PULL * dt;
    }
  }
  const now = performance.now();
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (const e of edges) {
    if (now - e.birth > 1500) continue;
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
    const delta = (d - EDGE_REST) * EDGE_SPRING;
    const fx = (dx / d) * delta, fy = (dy / d) * delta;
    if (!a.isOrchestrator) { a.vx += fx * dt; a.vy += fy * dt; }
    if (!b.isOrchestrator) { b.vx -= fx * dt; b.vy -= fy * dt; }
  }
  for (const n of nodes) {
    if (n.isOrchestrator) continue;
    n.vx *= DAMPING; n.vy *= DAMPING;
    n.x += n.vx * dt; n.y += n.vy * dt;
    const p = 40;
    if (n.x < p) { n.x = p; n.vx = 0; }
    if (n.x > WIDTH - p) { n.x = WIDTH - p; n.vx = 0; }
    if (n.y < p) { n.y = p; n.vy = 0; }
    if (n.y > HEIGHT - p) { n.y = HEIGHT - p; n.vy = 0; }
  }
}

export const NeoMind: React.FC = () => {
  const [status, setStatus] = useState<OrganismStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailState | null>(null);
  const [, setTick] = useState(0);

  const nodesRef = useRef<SimNode[]>([]);
  const edgesRef = useRef<SimEdge[]>([]);
  const messagesRef = useRef<BusMessage[]>([]);
  const activityRef = useRef<Record<string, number>>({});

  const syncNodes = useCallback((data: OrganismStatus) => {
    const existing = new Map(nodesRef.current.map(n => [n.id, n]));
    const next: SimNode[] = [];
    const orch = existing.get("orchestrator");
    next.push(orch ?? {
      id: "orchestrator", label: "Orchestrator", role: "orchestrator",
      isOrchestrator: true, invocations: data.total_interactions, successRate: 1,
      lastActivity: 0, x: CX, y: CY, vx: 0, vy: 0, fitness: 1,
    });
    next[0].invocations = data.total_interactions;
    data.specialists.forEach((s, i) => {
      const prev = existing.get(s.name);
      const angle = (i / Math.max(1, data.specialists.length)) * Math.PI * 2;
      const r = 230;
      next.push(prev ? {
        ...prev, invocations: s.invocations, successRate: s.success_rate,
        fitness: Math.max(0.2, Math.min(1, s.success_rate)),
      } : {
        id: s.name, label: s.name, role: roleOf(s.name), isOrchestrator: false,
        invocations: s.invocations, successRate: s.success_rate, lastActivity: 0,
        x: CX + Math.cos(angle) * r, y: CY + Math.sin(angle) * r,
        vx: 0, vy: 0, fitness: Math.max(0.2, Math.min(1, s.success_rate)),
      });
    });
    nodesRef.current = next;
  }, []);

  // Poll /api/organism/status every 1s
  useEffect(() => {
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/organism/status");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: OrganismStatus = await res.json();
        if (cancelled) return;
        setStatus(data); setStatusError(null); syncNodes(data);
      } catch (err) {
        if (!cancelled) setStatusError((err as Error).message);
      }
    };
    fetchStatus();
    const id = window.setInterval(fetchStatus, 1000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [syncNodes]);

  // SSE /api/messages/stream
  useEffect(() => {
    const es = new EventSource("/api/messages/stream");
    es.onmessage = (ev) => {
      try {
        const msg: BusMessage = JSON.parse(ev.data);
        const buf = messagesRef.current;
        buf.push(msg);
        if (buf.length > 500) buf.shift();
        activityRef.current[msg.from] = performance.now();
        activityRef.current[msg.to] = performance.now();
        edgesRef.current.push({ from: msg.from, to: msg.to, birth: performance.now(), intent: msg.intent });
        if (edgesRef.current.length > 200) edgesRef.current.shift();
      } catch { /* ignore */ }
    };
    return () => es.close();
  }, []);

  // 60fps loop
  useEffect(() => {
    let raf = 0, last = performance.now();
    const loop = (t: number) => {
      const dt = Math.min(2, (t - last) / 16.67);
      last = t;
      const act = activityRef.current;
      for (const n of nodesRef.current) {
        const a = act[n.id];
        if (a && a > n.lastActivity) n.lastActivity = a;
      }
      stepSim(nodesRef.current, edgesRef.current, dt);
      const now = performance.now();
      edgesRef.current = edgesRef.current.filter(e => now - e.birth < 1500);
      setTick(x => (x + 1) & 0xffff);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Detail panel — refresh when selection changes + periodically for new msgs
  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    const refresh = () => {
      const related = messagesRef.current
        .filter(m => m.from === selectedId || m.to === selectedId)
        .slice(-10).reverse();
      setDetail({ id: selectedId, messages: related });
    };
    refresh();
    const id = window.setInterval(refresh, 500);
    return () => window.clearInterval(id);
  }, [selectedId]);

  const specialistCount = status?.specialists.length ?? 0;
  const idle = !status || specialistCount === 0;
  const selectedNode = useMemo(
    () => nodesRef.current.find(n => n.id === selectedId) ?? null,
    [selectedId, status]
  );
  const now = performance.now();

  const wrapStyle: React.CSSProperties = {
    background: "#0a0e27", color: "#d0e4ff",
    fontFamily: "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace",
    minHeight: HEIGHT + 80, padding: 16, borderRadius: 12, position: "relative",
  };
  const svgStyle: React.CSSProperties = {
    background: "radial-gradient(ellipse at center, #0d1436 0%, #050714 100%)",
    border: "1px solid #1a2347", borderRadius: 8,
    boxShadow: "inset 0 0 80px rgba(0,229,255,0.04)", flex: 1, maxWidth: WIDTH,
  };
  const asideStyle: React.CSSProperties = {
    width: 280, background: "#0d1436", border: "1px solid #1a2347",
    borderRadius: 8, padding: 14, fontSize: 12, minHeight: HEIGHT,
    boxShadow: "inset 0 0 20px rgba(0,229,255,0.04)",
  };

  return (
    <div style={wrapStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontWeight: 600, letterSpacing: 2, color: "#00e5ff", textShadow: "0 0 12px rgba(0,229,255,0.6)" }}>NEO MIND</h2>
          <div style={{ fontSize: 11, color: "#6a7aa8", marginTop: 2 }}>live organism topology — polling 1Hz + QLMS SSE</div>
        </div>
        <div style={{ fontSize: 12, textAlign: "right", lineHeight: 1.5 }}>
          <div>
            <span style={{ color: "#6a7aa8" }}>gen </span><span style={{ color: "#ff2bd6" }}>{status?.generation ?? "—"}</span>{"   "}
            <span style={{ color: "#6a7aa8" }}>spec </span><span style={{ color: "#64ffda" }}>{specialistCount}</span>{"   "}
            <span style={{ color: "#6a7aa8" }}>int </span><span style={{ color: "#ffd54a" }}>{status?.total_interactions ?? 0}</span>{"   "}
            <span style={{ color: "#6a7aa8" }}>mem </span><span style={{ color: "#b388ff" }}>{status?.memory_items ?? 0}</span>
          </div>
          {statusError && <div style={{ color: "#ff6b6b", fontSize: 10, marginTop: 4 }}>api error: {statusError}</div>}
        </div>
      </header>

      <div style={{ display: "flex", gap: 16 }}>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" style={svgStyle} onClick={() => setSelectedId(null)}>
          <defs>
            <filter id="nmBlur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" /></filter>
          </defs>
          <g stroke="#12204a" strokeWidth="0.5" opacity="0.5">
            {Array.from({ length: 12 }, (_, i) => <line key={`v${i}`} x1={(WIDTH / 12) * i} y1={0} x2={(WIDTH / 12) * i} y2={HEIGHT} />)}
            {Array.from({ length: 8 }, (_, i) => <line key={`h${i}`} x1={0} y1={(HEIGHT / 8) * i} x2={WIDTH} y2={(HEIGHT / 8) * i} />)}
          </g>
          <g>
            {edgesRef.current.map((e, i) => {
              const a = nodesRef.current.find(n => n.id === e.from);
              const b = nodesRef.current.find(n => n.id === e.to);
              if (!a || !b) return null;
              const alpha = Math.max(0, 1 - (now - e.birth) / 1500);
              return <line key={`e${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#00e5ff" strokeWidth={1.5} opacity={alpha * 0.8} strokeLinecap="round" />;
            })}
          </g>
          <g>
            {nodesRef.current.map(n => {
              const role = n.isOrchestrator ? "orchestrator" : n.role;
              const color = colorFor(role);
              const baseR = n.isOrchestrator ? 28 : 10 + n.fitness * 14;
              const since = now - n.lastActivity;
              const pulse = since < 1200 ? Math.max(0, 1 - since / 1200) : 0;
              const r = baseR + pulse * 8;
              const isSelected = selectedId === n.id;
              return (
                <g key={n.id} transform={`translate(${n.x},${n.y})`} style={{ cursor: "pointer" }}
                   onClick={(ev) => { ev.stopPropagation(); setSelectedId(n.id); }}>
                  <circle r={r + 14} fill={color} opacity={0.12 + pulse * 0.25} filter="url(#nmBlur)" />
                  {pulse > 0 && <circle r={r + 6 + pulse * 10} fill="none" stroke={color} strokeWidth={1} opacity={pulse * 0.7} />}
                  <circle r={r} fill="#0a0e27" stroke={color} strokeWidth={isSelected ? 3 : 1.5} />
                  <circle r={r - 4} fill={color} opacity={0.2} />
                  <text y={r + 16} textAnchor="middle" fontSize={n.isOrchestrator ? 13 : 11} fill={color} style={{ letterSpacing: 1 }}>{n.label}</text>
                  {!n.isOrchestrator && <text y={r + 30} textAnchor="middle" fontSize={9} fill="#6a7aa8">{n.invocations} · {(n.successRate * 100).toFixed(0)}%</text>}
                </g>
              );
            })}
          </g>
          {idle && (
            <g>
              <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#0a0e27" opacity={0.7} />
              <text x={CX} y={CY - 12} textAnchor="middle" fontSize={22} fill="#00e5ff" style={{ letterSpacing: 6 }}>ORGANISM IDLE</text>
              <text x={CX} y={CY + 16} textAnchor="middle" fontSize={12} fill="#6a7aa8">
                {statusError ? "cannot reach /api/organism/status" : "no specialists registered — send a message to wake the swarm"}
              </text>
            </g>
          )}
        </svg>

        <aside style={asideStyle}>
          {selectedNode ? (
            <div>
              <div style={{ color: colorFor(selectedNode.role), fontSize: 14, letterSpacing: 2, marginBottom: 8, textShadow: `0 0 8px ${colorFor(selectedNode.role)}55` }}>
                {selectedNode.label.toUpperCase()}
              </div>
              <div style={{ color: "#6a7aa8", marginBottom: 10 }}>
                <div>id: {selectedNode.id}</div>
                <div>role: {selectedNode.role}</div>
                <div>invocations: {selectedNode.invocations}</div>
                <div>success: {(selectedNode.successRate * 100).toFixed(1)}%</div>
                <div>fitness: {selectedNode.fitness.toFixed(2)}</div>
              </div>
              <div style={{ color: "#00e5ff", fontSize: 10, letterSpacing: 2, marginBottom: 6 }}>LAST 10 MESSAGES</div>
              {detail && detail.messages.length > 0 ? (
                <ul style={{ listStyle: "none", padding: 0, margin: 0, maxHeight: 420, overflowY: "auto" }}>
                  {detail.messages.map(m => (
                    <li key={m.id} style={{ padding: "4px 0", borderBottom: "1px solid #1a2347", fontSize: 10 }}>
                      <div style={{ color: "#ffd54a" }}>#{m.id} · {m.intent}</div>
                      <div style={{ color: "#6a7aa8" }}>{m.from} → {m.to}</div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div style={{ color: "#6a7aa8", fontStyle: "italic" }}>no recent messages on the bus</div>
              )}
            </div>
          ) : (
            <div style={{ color: "#6a7aa8" }}>
              <div style={{ color: "#00e5ff", letterSpacing: 2, marginBottom: 8 }}>INSPECT</div>
              click any node to inspect its role, fitness and the last messages that traversed it on the QLMS bus.
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

export default NeoMind;
