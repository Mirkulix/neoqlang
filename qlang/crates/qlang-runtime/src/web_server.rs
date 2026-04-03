//! WebSocket server for streaming QLANG events to the web dashboard.
//!
//! Implements HTTP file serving and WebSocket protocol (RFC 6455) using only `std::net`.
//! No external crates are used for SHA-1, Base64, or WebSocket framing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// WebEvent
// ---------------------------------------------------------------------------

/// Events that can be broadcast to connected WebSocket clients.
#[derive(Clone, Debug)]
pub enum WebEvent {
    GraphNodeExecuted {
        node_id: u32,
        op: String,
        shape: String,
        time_us: u64,
        values: Option<Vec<f32>>,
    },
    TrainingEpoch {
        epoch: usize,
        loss: f32,
        accuracy: f32,
    },
    AgentMessage {
        from: String,
        to: String,
        message: String,
    },
    CompressionResult {
        method: String,
        ratio: f32,
        accuracy_before: f32,
        accuracy_after: f32,
    },
    SystemLog {
        level: String,
        message: String,
    },
    GraphLoaded {
        name: String,
        num_nodes: usize,
        num_edges: usize,
    },
    ModelSaved {
        name: String,
        version: String,
    },
}

impl WebEvent {
    /// Serialize to JSON manually (no serde needed).
    pub fn to_json(&self) -> String {
        match self {
            WebEvent::GraphNodeExecuted { node_id, op, shape, time_us, values } => {
                let vals = match values {
                    Some(v) => {
                        let items: Vec<String> = v.iter().map(|f| format!("{f}")).collect();
                        format!("[{}]", items.join(","))
                    }
                    None => "null".to_string(),
                };
                format!(
                    r#"{{"type":"GraphNodeExecuted","node_id":{node_id},"op":"{op}","shape":"{shape}","time_us":{time_us},"values":{vals}}}"#,
                )
            }
            WebEvent::TrainingEpoch { epoch, loss, accuracy } => {
                format!(
                    r#"{{"type":"TrainingEpoch","epoch":{epoch},"loss":{loss},"accuracy":{accuracy}}}"#,
                )
            }
            WebEvent::AgentMessage { from, to, message } => {
                let msg = json_escape(message);
                format!(
                    r#"{{"type":"AgentMessage","from":"{from}","to":"{to}","message":"{msg}"}}"#,
                )
            }
            WebEvent::CompressionResult { method, ratio, accuracy_before, accuracy_after } => {
                format!(
                    r#"{{"type":"CompressionResult","method":"{method}","ratio":{ratio},"accuracy_before":{accuracy_before},"accuracy_after":{accuracy_after}}}"#,
                )
            }
            WebEvent::SystemLog { level, message } => {
                let msg = json_escape(message);
                format!(
                    r#"{{"type":"SystemLog","level":"{level}","message":"{msg}"}}"#,
                )
            }
            WebEvent::GraphLoaded { name, num_nodes, num_edges } => {
                format!(
                    r#"{{"type":"GraphLoaded","name":"{name}","num_nodes":{num_nodes},"num_edges":{num_edges}}}"#,
                )
            }
            WebEvent::ModelSaved { name, version } => {
                format!(
                    r#"{{"type":"ModelSaved","name":"{name}","version":"{version}"}}"#,
                )
            }
        }
    }
}

/// Escape special characters for JSON string values.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SHA-1 (RFC 3174) — minimal implementation
// ---------------------------------------------------------------------------

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;

    // Pad message: append 0x80, then zeros, then 64-bit big-endian length
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process 512-bit (64-byte) blocks
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

// ---------------------------------------------------------------------------
// Base64 encode
// ---------------------------------------------------------------------------

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// WebSocket accept key computation
// ---------------------------------------------------------------------------

const WS_MAGIC: &str = "258EAFA5-E914-47DA-95CA-5AB5DC786C11";

pub fn compute_accept_key(client_key: &str) -> String {
    let mut input = String::with_capacity(client_key.len() + WS_MAGIC.len());
    input.push_str(client_key.trim());
    input.push_str(WS_MAGIC);
    let hash = sha1(input.as_bytes());
    base64_encode(&hash)
}

// ---------------------------------------------------------------------------
// WebSocket frame encoding / decoding
// ---------------------------------------------------------------------------

/// A decoded WebSocket frame.
#[derive(Debug, Clone)]
pub struct WsFrame {
    pub opcode: u8,
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Encode a frame for sending from server to client (unmasked).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // FIN + opcode
        buf.push(0x80 | (self.opcode & 0x0F));
        let len = self.payload.len();
        if len < 126 {
            buf.push(len as u8);
        } else if len <= 0xFFFF {
            buf.push(126);
            buf.push((len >> 8) as u8);
            buf.push((len & 0xFF) as u8);
        } else {
            buf.push(127);
            buf.extend_from_slice(&(len as u64).to_be_bytes());
        }
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from a client stream (expects masked frames from client).
    pub fn decode(stream: &mut impl Read) -> std::io::Result<Self> {
        let mut header = [0u8; 2];
        stream.read_exact(&mut header)?;

        let opcode = header[0] & 0x0F;
        let masked = (header[1] & 0x80) != 0;
        let len_byte = header[1] & 0x7F;

        let payload_len: usize = if len_byte < 126 {
            len_byte as usize
        } else if len_byte == 126 {
            let mut buf = [0u8; 2];
            stream.read_exact(&mut buf)?;
            u16::from_be_bytes(buf) as usize
        } else {
            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf)?;
            u64::from_be_bytes(buf) as usize
        };

        let mask_key = if masked {
            let mut key = [0u8; 4];
            stream.read_exact(&mut key)?;
            Some(key)
        } else {
            None
        };

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            stream.read_exact(&mut payload)?;
        }

        // Unmask
        if let Some(key) = mask_key {
            for i in 0..payload.len() {
                payload[i] ^= key[i % 4];
            }
        }

        Ok(WsFrame { opcode, payload })
    }

    /// Create a text frame.
    pub fn text(msg: &str) -> Self {
        WsFrame {
            opcode: 0x1,
            payload: msg.as_bytes().to_vec(),
        }
    }

    /// Create a close frame.
    pub fn close() -> Self {
        WsFrame {
            opcode: 0x8,
            payload: Vec::new(),
        }
    }

    /// Create a pong frame with given payload.
    pub fn pong(payload: Vec<u8>) -> Self {
        WsFrame {
            opcode: 0xA,
            payload,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn parse_http_request(reader: &mut BufReader<&TcpStream>) -> Option<(String, String, HashMap<String, String>)> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            break;
        }
        if let Some((key, val)) = line.split_once(':') {
            headers.insert(key.trim().to_lowercase(), val.trim().to_string());
        }
    }

    Some((method, path, headers))
}

fn content_type_for(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

fn send_http_response(stream: &mut TcpStream, status: u16, status_text: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn send_404(stream: &mut TcpStream) {
    send_http_response(stream, 404, "Not Found", "text/plain", b"404 Not Found");
}

// ---------------------------------------------------------------------------
// WebServer
// ---------------------------------------------------------------------------

/// Shared list of connected WebSocket client streams.
type Clients = Arc<Mutex<Vec<Arc<Mutex<TcpStream>>>>>;

/// A WebSocket + HTTP server for the QLANG dashboard.
pub struct WebServer {
    clients: Clients,
    web_root: String,
}

impl WebServer {
    /// Start the HTTP + WebSocket server on the given port.
    ///
    /// This blocks the calling thread. The returned `WebServerHandle` can be
    /// used from other threads to broadcast events.
    pub fn start(port: u16, web_root: String) -> std::io::Result<WebServerHandle> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))?;
        let clients: Clients = Arc::new(Mutex::new(Vec::new()));
        let handle = WebServerHandle {
            clients: Arc::clone(&clients),
        };

        let server = WebServer {
            clients,
            web_root,
        };

        thread::spawn(move || {
            server.run(listener);
        });

        Ok(handle)
    }

    fn run(&self, listener: TcpListener) {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let clients = Arc::clone(&self.clients);
                    let web_root = self.web_root.clone();
                    thread::spawn(move || {
                        handle_connection(stream, clients, &web_root);
                    });
                }
                Err(e) => {
                    eprintln!("[web_server] accept error: {e}");
                }
            }
        }
    }
}

/// Handle for broadcasting events to connected WebSocket clients.
#[derive(Clone)]
pub struct WebServerHandle {
    clients: Clients,
}

impl WebServerHandle {
    /// Broadcast a `WebEvent` to all connected WebSocket clients.
    pub fn broadcast(&self, event: WebEvent) {
        let json = event.to_json();
        let frame = WsFrame::text(&json).encode();
        let mut clients = self.clients.lock().unwrap();
        clients.retain(|client| {
            let mut stream = client.lock().unwrap();
            stream.write_all(&frame).is_ok()
        });
    }
}

fn handle_connection(stream: TcpStream, clients: Clients, web_root: &str) {
    let peer = stream.peer_addr().ok();

    // Clone for the BufReader; we need the original for writing
    let read_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(&read_stream);

    let (method, path, headers) = match parse_http_request(&mut reader) {
        Some(v) => v,
        None => return,
    };

    // Check for WebSocket upgrade on /ws
    if method == "GET"
        && path == "/ws"
        && headers.get("upgrade").map(|v| v.eq_ignore_ascii_case("websocket")).unwrap_or(false)
    {
        if let Some(key) = headers.get("sec-websocket-key") {
            let accept = compute_accept_key(key);
            let response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            );

            let mut ws_stream = stream;
            if ws_stream.write_all(response.as_bytes()).is_err() {
                return;
            }
            let _ = ws_stream.flush();

            if let Some(addr) = peer {
                eprintln!("[web_server] WebSocket connected: {addr}");
            }

            let client = Arc::new(Mutex::new(ws_stream.try_clone().unwrap()));
            {
                let mut list = clients.lock().unwrap();
                list.push(Arc::clone(&client));
            }

            // Read loop for this WebSocket client
            handle_websocket(ws_stream, &clients, &client);

            // Remove client on disconnect
            {
                let mut list = clients.lock().unwrap();
                list.retain(|c| !Arc::ptr_eq(c, &client));
            }

            if let Some(addr) = peer {
                eprintln!("[web_server] WebSocket disconnected: {addr}");
            }
        }
        return;
    }

    // Regular HTTP file serving
    let mut stream = stream;
    if method != "GET" {
        send_404(&mut stream);
        return;
    }

    let file_path = match path.as_str() {
        "/" => "index.html".to_string(),
        "/demo" => "demo.html".to_string(),
        p => {
            let p = p.trim_start_matches('/');
            // Security: reject path traversal
            if p.contains("..") {
                send_404(&mut stream);
                return;
            }
            p.to_string()
        }
    };

    // Only serve known file types
    if !(file_path.ends_with(".html")
        || file_path.ends_with(".js")
        || file_path.ends_with(".css")
        || file_path.ends_with(".json"))
    {
        send_404(&mut stream);
        return;
    }

    let full_path = format!("{web_root}/{file_path}");
    match std::fs::read(&full_path) {
        Ok(body) => {
            let ct = content_type_for(&file_path);
            send_http_response(&mut stream, 200, "OK", ct, &body);
        }
        Err(_) => {
            send_404(&mut stream);
        }
    }
}

fn handle_websocket(mut stream: TcpStream, _clients: &Clients, _client: &Arc<Mutex<TcpStream>>) {
    loop {
        let frame = match WsFrame::decode(&mut stream) {
            Ok(f) => f,
            Err(_) => break,
        };

        match frame.opcode {
            0x1 => {
                // Text frame — we don't expect client messages, but we could handle them.
            }
            0x8 => {
                // Close frame — send close back
                let close = WsFrame::close().encode();
                let _ = stream.write_all(&close);
                break;
            }
            0x9 => {
                // Ping — respond with pong
                let pong = WsFrame::pong(frame.payload).encode();
                let _ = stream.write_all(&pong);
            }
            0xA => {
                // Pong — ignore
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha1_empty() {
        let hash = sha1(b"");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn test_sha1_abc() {
        let hash = sha1(b"abc");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn test_sha1_longer() {
        let hash = sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_websocket_accept_key() {
        // Verified against Python hashlib.sha1 + base64
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = compute_accept_key(key);
        assert_eq!(accept, "gp5ZIgwWHrq79z38B+BCpwLlQMw=");

        // Another known test vector (verified with Python hashlib)
        let key2 = "x3JJHMbDL1EzLkh9GBhXDw==";
        let accept2 = compute_accept_key(key2);
        assert_eq!(accept2, "/Um7YN7xo2/kpzeG5juBUMMrN08=");
    }

    #[test]
    fn test_ws_frame_encode_short() {
        let frame = WsFrame::text("hello");
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x81); // FIN + text opcode
        assert_eq!(encoded[1], 5);    // payload length
        assert_eq!(&encoded[2..], b"hello");
    }

    #[test]
    fn test_ws_frame_encode_medium() {
        // 200-byte payload (uses 2-byte extended length)
        let payload = "x".repeat(200);
        let frame = WsFrame::text(&payload);
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x81);
        assert_eq!(encoded[1], 126);   // extended length marker
        assert_eq!(encoded[2], 0);     // length high byte
        assert_eq!(encoded[3], 200);   // length low byte
        assert_eq!(encoded.len(), 4 + 200);
    }

    #[test]
    fn test_ws_frame_decode_roundtrip() {
        // Simulate a masked client frame
        let payload = b"hello";
        let mask_key: [u8; 4] = [0x37, 0xfa, 0x21, 0x3d];
        let mut masked_payload = payload.to_vec();
        for i in 0..masked_payload.len() {
            masked_payload[i] ^= mask_key[i % 4];
        }

        let mut frame_data = Vec::new();
        frame_data.push(0x81); // FIN + text
        frame_data.push(0x80 | 5); // masked + length 5
        frame_data.extend_from_slice(&mask_key);
        frame_data.extend_from_slice(&masked_payload);

        let mut cursor = std::io::Cursor::new(frame_data);
        let decoded = WsFrame::decode(&mut cursor).unwrap();
        assert_eq!(decoded.opcode, 0x1);
        assert_eq!(decoded.payload, b"hello");
    }

    #[test]
    fn test_ws_frame_close() {
        let frame = WsFrame::close();
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x88); // FIN + close opcode
        assert_eq!(encoded[1], 0);
    }

    #[test]
    fn test_ws_frame_pong() {
        let frame = WsFrame::pong(b"pingdata".to_vec());
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x8A); // FIN + pong opcode
        assert_eq!(encoded[1], 8);
        assert_eq!(&encoded[2..], b"pingdata");
    }

    #[test]
    fn test_event_json_graph_node_executed() {
        let event = WebEvent::GraphNodeExecuted {
            node_id: 1,
            op: "MatMul".to_string(),
            shape: "[2,3]".to_string(),
            time_us: 42,
            values: Some(vec![1.0, 2.0]),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"GraphNodeExecuted\""));
        assert!(json.contains("\"node_id\":1"));
        assert!(json.contains("\"op\":\"MatMul\""));
        assert!(json.contains("\"time_us\":42"));
        assert!(json.contains("[1,2]"));
    }

    #[test]
    fn test_event_json_training_epoch() {
        let event = WebEvent::TrainingEpoch {
            epoch: 5,
            loss: 0.5,
            accuracy: 0.9,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"TrainingEpoch\""));
        assert!(json.contains("\"epoch\":5"));
    }

    #[test]
    fn test_event_json_system_log_escape() {
        let event = WebEvent::SystemLog {
            level: "info".to_string(),
            message: "line1\nline2\"quoted\"".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\\n"));
        assert!(json.contains("\\\"quoted\\\""));
    }

    #[test]
    fn test_event_json_agent_message() {
        let event = WebEvent::AgentMessage {
            from: "agent_a".to_string(),
            to: "agent_b".to_string(),
            message: "hello".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"AgentMessage\""));
        assert!(json.contains("\"from\":\"agent_a\""));
    }

    #[test]
    fn test_event_json_compression_result() {
        let event = WebEvent::CompressionResult {
            method: "ternary".to_string(),
            ratio: 0.25,
            accuracy_before: 0.95,
            accuracy_after: 0.93,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"CompressionResult\""));
        assert!(json.contains("\"method\":\"ternary\""));
    }

    #[test]
    fn test_event_json_graph_loaded() {
        let event = WebEvent::GraphLoaded {
            name: "test_graph".to_string(),
            num_nodes: 10,
            num_edges: 15,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"GraphLoaded\""));
        assert!(json.contains("\"num_nodes\":10"));
    }

    #[test]
    fn test_event_json_model_saved() {
        let event = WebEvent::ModelSaved {
            name: "model1".to_string(),
            version: "v1.0".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"ModelSaved\""));
        assert!(json.contains("\"version\":\"v1.0\""));
    }

    #[test]
    fn test_event_json_graph_node_no_values() {
        let event = WebEvent::GraphNodeExecuted {
            node_id: 2,
            op: "ReLU".to_string(),
            shape: "[4]".to_string(),
            time_us: 10,
            values: None,
        };
        let json = event.to_json();
        assert!(json.contains("\"values\":null"));
    }
}
