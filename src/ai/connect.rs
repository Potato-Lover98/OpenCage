//! `/connect`: link OpenCage sessions over the network into a shared room.
//!
//! The creator hosts an encrypted TCP relay; joiners connect by room id. Every message is
//! AES-256-GCM encrypted with a random per-room key that travels inside the room id, so only
//! holders of the id can join or read traffic — a connection whose first frame doesn't decrypt
//! is dropped. Prompts typed in any session are broadcast to the whole room.
//!
//! NOTE: the host binds `0.0.0.0`, but reachability across the internet still depends on the
//! host's network (NAT/port-forwarding) — there is no hosted relay.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rand::RngCore;
use serde_json::{Value, json};

/// Connected peers' write handles, keyed by remote address (host: all peers; joiner: the host).
pub type PeerList = Arc<Mutex<Vec<(String, TcpStream)>>>;

/// A message received from a peer, for display in the chat.
pub struct PeerMsg {
    pub from: String,
    pub text: String,
}

/// A fresh random 256-bit room key.
pub fn new_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key
}

/// An empty peer list for a host.
pub fn new_peers() -> PeerList {
    Arc::new(Mutex::new(Vec::new()))
}

/// Encode a room id as `ip:port:base64(key)`.
pub fn make_room_id(ip: &str, port: u16, key: &[u8; 32]) -> String {
    format!("{ip}:{port}:{}", STANDARD.encode(key))
}

fn parse_room_id(id: &str) -> Result<(String, u16, [u8; 32])> {
    let parts: Vec<&str> = id.trim().splitn(3, ':').collect();
    if parts.len() != 3 {
        bail!("Invalid room id (expected ip:port:key)");
    }
    let ip = parts[0].to_string();
    let port: u16 = parts[1].parse().context("Invalid port in room id")?;
    let key_bytes = STANDARD
        .decode(parts[2].trim())
        .context("Invalid key in room id")?;
    let key: [u8; 32] = key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("Room key must decode to 32 bytes"))?;
    Ok((ip, port, key))
}

/// This machine's LAN IP (best effort), for the same-network room id.
pub fn local_ip() -> String {
    use std::net::UdpSocket;
    UdpSocket::bind("0.0.0.0:0")
        .ok()
        .and_then(|s| {
            s.connect("8.8.8.8:80").ok()?;
            s.local_addr().ok()
        })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

/// This machine's public IP (best effort), for the over-the-internet room id.
pub fn public_ip() -> Option<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let body = client.get("https://api.ipify.org").send().ok()?.text().ok()?;
    let ip = body.trim().to_string();
    (!ip.is_empty() && ip.len() <= 45 && ip.chars().all(|c| c.is_ascii_hexdigit() || c == '.' || c == ':'))
        .then_some(ip)
}

/// Encrypt `{from,text}` into a base64 line ready to send (nonce ++ ciphertext).
pub fn encode_msg(key: &[u8; 32], from: &str, text: &str) -> Option<String> {
    let plain = json!({ "from": from, "text": text }).to_string();
    encrypt(key, plain.as_bytes())
}

fn encrypt(key: &[u8; 32], plain: &[u8]) -> Option<String> {
    let cipher = Aes256Gcm::new_from_slice(key).ok()?;
    let mut nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce);
    let ct = cipher.encrypt(Nonce::from_slice(&nonce), plain).ok()?;
    let mut buf = Vec::with_capacity(12 + ct.len());
    buf.extend_from_slice(&nonce);
    buf.extend_from_slice(&ct);
    Some(STANDARD.encode(buf))
}

fn decrypt(key: &[u8; 32], line: &str) -> Option<Vec<u8>> {
    let raw = STANDARD.decode(line.trim()).ok()?;
    if raw.len() < 13 {
        return None;
    }
    let cipher = Aes256Gcm::new_from_slice(key).ok()?;
    let (nonce, ct) = raw.split_at(12);
    cipher.decrypt(Nonce::from_slice(nonce), ct).ok()
}

/// Write `line` to every peer (optionally skipping one address); drop peers that fail.
pub fn forward(peers: &PeerList, line: &str, exclude: Option<&str>) {
    let payload = if line.ends_with('\n') {
        line.to_string()
    } else {
        format!("{line}\n")
    };
    if let Ok(mut list) = peers.lock() {
        list.retain_mut(|(addr, stream)| {
            if exclude == Some(addr.as_str()) {
                return true;
            }
            stream.write_all(payload.as_bytes()).and_then(|_| stream.flush()).is_ok()
        });
    }
}

/// Host the room: accept encrypted peers, relay their messages to everyone else, and surface
/// them locally via `inbox`.
pub fn host_room(
    listener: TcpListener,
    key: [u8; 32],
    peers: PeerList,
    inbox: Sender<PeerMsg>,
    stop: Arc<AtomicBool>,
) {
    let _ = listener.set_nonblocking(true);
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, addr)) => {
                let addr = addr.to_string();
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let Ok(read_clone) = stream.try_clone() else {
                    continue;
                };
                let mut reader = BufReader::new(read_clone);
                let mut first = String::new();
                // Authenticate: the first frame must decrypt with the room key.
                if reader.read_line(&mut first).unwrap_or(0) == 0 || decrypt(&key, &first).is_none() {
                    continue;
                }
                let name = serde_json::from_slice::<Value>(&decrypt(&key, &first).unwrap_or_default())
                    .ok()
                    .and_then(|v| v["from"].as_str().map(str::to_string))
                    .unwrap_or_else(|| "peer".to_string());
                let _ = stream.set_read_timeout(None);
                let _ = inbox.send(PeerMsg {
                    from: "room".to_string(),
                    text: format!("{name} joined the room."),
                });
                if let Ok(mut list) = peers.lock() {
                    list.push((addr.clone(), stream));
                }
                let peers_r = peers.clone();
                let inbox_r = inbox.clone();
                std::thread::spawn(move || {
                    relay_reader(reader, key, &peers_r, &inbox_r, Some(&addr));
                    let _ = inbox_r.send(PeerMsg {
                        from: "room".to_string(),
                        text: "A peer left the room.".to_string(),
                    });
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(150)),
        }
    }
}

/// Join a room by id. Returns the peer list (the single host stream) and the room key, so the
/// caller can broadcast its own prompts.
pub fn join_room(
    id: &str,
    name: &str,
    inbox: Sender<PeerMsg>,
    stop: Arc<AtomicBool>,
) -> Result<(PeerList, [u8; 32])> {
    let (ip, port, key) = parse_room_id(id)?;
    let stream = TcpStream::connect((ip.as_str(), port))
        .with_context(|| format!("Could not reach the room host at {ip}:{port}"))?;
    // Handshake: prove we hold the key by sending an encrypted hello.
    let hello = encode_msg(&key, name, "").ok_or_else(|| anyhow!("encryption failed"))?;
    let mut writer = stream.try_clone()?;
    writer
        .write_all(format!("{hello}\n").as_bytes())
        .context("Failed to send handshake")?;
    writer.flush().ok();

    let peers: PeerList = Arc::new(Mutex::new(vec![("host".to_string(), stream.try_clone()?)]));
    let inbox_r = inbox.clone();
    let stop_r = stop.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stream);
        // Joiner doesn't relay; only the host does. Stop is checked between lines.
        relay_reader_stoppable(reader, key, &inbox_r, &stop_r);
        let _ = inbox_r.send(PeerMsg {
            from: "room".to_string(),
            text: "Disconnected from room.".to_string(),
        });
    });
    Ok((peers, key))
}

/// Read encrypted lines; surface each to `inbox` and (host) rebroadcast to other peers.
fn relay_reader(
    mut reader: BufReader<TcpStream>,
    key: [u8; 32],
    peers: &PeerList,
    inbox: &Sender<PeerMsg>,
    origin: Option<&str>,
) {
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if let Some((from, text)) = decode_line(&key, &line) {
                    if !text.is_empty() {
                        let _ = inbox.send(PeerMsg { from, text });
                        forward(peers, &line, origin);
                    }
                }
            }
        }
    }
}

fn relay_reader_stoppable(
    mut reader: BufReader<TcpStream>,
    key: [u8; 32],
    inbox: &Sender<PeerMsg>,
    stop: &Arc<AtomicBool>,
) {
    let mut line = String::new();
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if let Some((from, text)) = decode_line(&key, &line) {
                    if !text.is_empty() {
                        let _ = inbox.send(PeerMsg { from, text });
                    }
                }
            }
        }
    }
}

fn decode_line(key: &[u8; 32], line: &str) -> Option<(String, String)> {
    let plain = decrypt(key, line)?;
    let v: Value = serde_json::from_slice(&plain).ok()?;
    let from = v["from"].as_str().unwrap_or("peer").to_string();
    let text = v["text"].as_str().unwrap_or("").to_string();
    Some((from, text))
}
