use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use reqwest::blocking::Client;
use reqwest::blocking::multipart;

#[cfg(feature = "local_voice")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "local_voice")]
use cpal::{Sample, SampleFormat, Stream, StreamConfig};
#[cfg(feature = "local_voice")]
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::core::models::Settings;

#[cfg(feature = "local_voice")]
pub struct VoiceRecorder {
    _stream: Stream,
    samples: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    pub level: Arc<AtomicU8>,
}

#[cfg(not(feature = "local_voice"))]
pub struct VoiceRecorder {
    samples: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    pub level: Arc<AtomicU8>,
}

impl VoiceRecorder {
    pub fn start() -> Result<Self> {
        #[cfg(not(feature = "local_voice"))]
        {
            anyhow::bail!(
                "Microphone capture is disabled in this build (compiled without `local_voice`). \
                 On macOS, rebuild on a Mac with default features, or cross-compile from Linux with \
                 a macOS SDK plus `--features local_voice`. Groq/OpenAI STT still works when a \
                 mic-capable build records audio."
            );
        }
        #[cfg(feature = "local_voice")]
        {
            start_voice_recorder_impl()
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn sample_buffer(&self) -> Arc<Mutex<Vec<i16>>> {
        self.samples.clone()
    }

    #[cfg(feature = "local_voice")]
    fn finish_impl(self) -> Result<(Vec<i16>, u32)> {
        let Self {
            _stream,
            samples,
            sample_rate,
            ..
        } = self;
        drop(_stream);
        let buf = samples.lock().unwrap().clone();
        Ok((buf, sample_rate))
    }

    /// Stop capture and return mono PCM plus sample rate (no disk I/O).
    pub fn finish(self) -> Result<(Vec<i16>, u32)> {
        #[cfg(feature = "local_voice")]
        {
            self.finish_impl()
        }
        #[cfg(not(feature = "local_voice"))]
        {
            let Self {
                samples,
                sample_rate,
                ..
            } = self;
            let buf = samples.lock().unwrap().clone();
            Ok((buf, sample_rate))
        }
    }

    /// Stop capture, write WAV under a temp path (legacy / debug).
    pub fn stop_and_save_wav(self) -> Result<PathBuf> {
        let (samples, sample_rate) = self.finish()?;
        let dir = tempfile::tempdir_in(std::env::temp_dir())?.keep();
        let path = dir.join("opencage_voice.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: HoundSampleFormat::Int,
        };
        let mut writer =
            WavWriter::create(&path, spec).context("Failed to create WAV writer")?;
        for s in samples {
            writer
                .write_sample(s)
                .context("Failed to write WAV sample")?;
        }
        writer.finalize().context("Failed to finalize WAV file")?;
        Ok(path)
    }
}

#[cfg(feature = "local_voice")]
fn start_voice_recorder_impl() -> Result<VoiceRecorder> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No default microphone input device found")?;
    let config = device
        .default_input_config()
        .context("Failed to fetch microphone input config")?;
    let sample_format = config.sample_format();
    let stream_config: StreamConfig = config.clone().into();
    let sample_rate = stream_config.sample_rate.0;

    let samples: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    let level = Arc::new(AtomicU8::new(0));

    let samples_cb = samples.clone();
    let level_cb = level.clone();

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| write_samples_f32(data, &samples_cb, &level_cb),
            |err| eprintln!("Voice stream error: {err}"),
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| write_samples_i16(data, &samples_cb, &level_cb),
            |err| eprintln!("Voice stream error: {err}"),
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| write_samples_u16(data, &samples_cb, &level_cb),
            |err| eprintln!("Voice stream error: {err}"),
            None,
        ),
        other => anyhow::bail!("Unsupported microphone sample format: {other:?}"),
    }
    .context("Failed to build microphone input stream")?;

    stream.play().context("Failed to start microphone stream")?;
    Ok(VoiceRecorder {
        _stream: stream,
        samples,
        sample_rate,
        level,
    })
}

#[cfg(feature = "local_voice")]
fn write_samples_f32(data: &[f32], samples: &Arc<Mutex<Vec<i16>>>, level: &Arc<AtomicU8>) {
    let mut peak = 0.0f32;
    let mut guard = samples.lock().unwrap();
    for s in data {
        let v = s.clamp(-1.0, 1.0);
        if v.abs() > peak {
            peak = v.abs();
        }
        let i = (v * i16::MAX as f32) as i16;
        guard.push(i);
    }
    drop(guard);
    let lvl = (peak * 100.0).clamp(0.0, 100.0) as u8;
    level.store(lvl, Ordering::Relaxed);
}

#[cfg(feature = "local_voice")]
fn write_samples_i16(data: &[i16], samples: &Arc<Mutex<Vec<i16>>>, level: &Arc<AtomicU8>) {
    let mut peak = 0i16;
    let mut guard = samples.lock().unwrap();
    for s in data {
        if s.abs() > peak {
            peak = s.abs();
        }
        guard.push(*s);
    }
    drop(guard);
    let lvl = ((peak as f32 / i16::MAX as f32) * 100.0).clamp(0.0, 100.0) as u8;
    level.store(lvl, Ordering::Relaxed);
}

#[cfg(feature = "local_voice")]
fn write_samples_u16(data: &[u16], samples: &Arc<Mutex<Vec<i16>>>, level: &Arc<AtomicU8>) {
    let mut peak = 0i16;
    let mut guard = samples.lock().unwrap();
    for s in data {
        let i = i16::from_sample(*s);
        if i.abs() > peak {
            peak = i.abs();
        }
        guard.push(i);
    }
    drop(guard);
    let lvl = ((peak as f32 / i16::MAX as f32) * 100.0).clamp(0.0, 100.0) as u8;
    level.store(lvl, Ordering::Relaxed);
}

#[derive(Debug, Clone)]
pub struct CloudSttConfig {
    pub backend: SttBackend,
}

#[derive(Debug, Clone)]
pub enum SttBackend {
    Local {
        model_path: PathBuf,
    },
    Cloud {
        provider: &'static str,
        url: String,
        model: String,
        api_key: String,
    },
}

impl CloudSttConfig {
    pub fn from_settings(settings: &Settings) -> Result<Self> {
        #[cfg(feature = "local_voice")]
        if let Some(path) = resolve_whisper_model_path() {
            return Ok(Self {
                backend: SttBackend::Local { model_path: path },
            });
        }
        if let Some(k) = settings
            .groq_api_key
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return Ok(Self {
                backend: SttBackend::Cloud {
                    provider: "Groq",
                    url: "https://api.groq.com/openai/v1/audio/transcriptions".to_string(),
                    model: "whisper-large-v3-turbo".to_string(),
                    api_key: k,
                },
            });
        }
        if let Some(k) = settings
            .openai_api_key
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return Ok(Self {
                backend: SttBackend::Cloud {
                    provider: "OpenAI",
                    url: "https://api.openai.com/v1/audio/transcriptions".to_string(),
                    model: "whisper-1".to_string(),
                    api_key: k,
                },
            });
        }
        anyhow::bail!(
            "No STT backend configured. Either set Groq/OpenAI key in Settings (no model download),{}",
            if cfg!(feature = "local_voice") {
                " or provide OPENCAGE_WHISPER_MODEL for offline whisper-rs."
            } else {
                " (offline whisper requires a build with `--features local_voice`.)"
            }
        )
    }

    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            SttBackend::Local { .. } => "offline whisper-rs",
            SttBackend::Cloud { provider, .. } => provider,
        }
    }
}

#[derive(Debug)]
pub enum VoiceSttUpdate {
    /// Interim transcript (updates while you speak).
    Partial(String),
    /// Full transcript after the mic stops (append this to the input field).
    Final(Result<String>),
}

fn pcm_to_wav_bytes(samples: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).context("WAV encode failed")?;
        for s in samples {
            writer.write_sample(*s)?;
        }
        writer.finalize().context("WAV finalize failed")?;
    }
    Ok(cursor.into_inner())
}

fn tail_window(samples: &[i16], sample_rate: u32, max_secs: u32) -> &[i16] {
    let cap = (sample_rate as usize).saturating_mul(max_secs as usize);
    if samples.len() > cap {
        &samples[samples.len() - cap..]
    } else {
        samples
    }
}

fn transcribe_pcm(cfg: &CloudSttConfig, pcm: &[i16], sample_rate: u32) -> Result<String> {
    if pcm.is_empty() {
        return Ok(String::new());
    }
    match &cfg.backend {
        SttBackend::Local { .. } => {
            #[cfg(feature = "local_voice")]
            {
                let mono_f32 = pcm_to_f32_mono_16k(pcm, sample_rate);
                transcribe_with_whisper_rs(cfg, &mono_f32)
            }
            #[cfg(not(feature = "local_voice"))]
            {
                anyhow::bail!("Offline whisper is not available in this build (enable `local_voice`).")
            }
        }
        SttBackend::Cloud { .. } => {
            let wav = pcm_to_wav_bytes(pcm, sample_rate)?;
            transcribe_cloud_wav_bytes(cfg, wav)
        }
    }
}

/// Background partials while `recording_active` is true; one `Final` after it goes false.
pub fn spawn_live_stt(
    samples: Arc<Mutex<Vec<i16>>>,
    sample_rate: u32,
    recording_active: Arc<AtomicBool>,
    cfg: CloudSttConfig,
    tx: std::sync::mpsc::Sender<VoiceSttUpdate>,
) {
    while recording_active.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(300));
        if !recording_active.load(Ordering::SeqCst) {
            break;
        }
        let _ = tx.send(VoiceSttUpdate::Partial("…listening…".to_string()));
    }
    let snap = samples.lock().map(|g| g.clone()).unwrap_or_default();
    let res = transcribe_pcm(&cfg, &snap, sample_rate);
    let _ = tx.send(VoiceSttUpdate::Final(res));
}

pub fn render_level_bar(level: u8) -> String {
    let width = 20usize;
    let filled = ((level as usize) * width) / 100;
    let bar_filled = "█".repeat(filled.min(width));
    let bar_empty = "░".repeat(width.saturating_sub(filled));
    format!("🎤 [{bar_filled}{bar_empty}] {level}%")
}

#[cfg(feature = "local_voice")]
fn transcribe_with_whisper_rs(cfg: &CloudSttConfig, audio: &[f32]) -> Result<String> {
    let model_path = match &cfg.backend {
        SttBackend::Local { model_path } => model_path,
        _ => anyhow::bail!("Expected local whisper backend"),
    };
    let model = model_path
        .to_str()
        .context("Whisper model path is not valid UTF-8")?;
    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .with_context(|| format!("Failed to load whisper model {}", model_path.display()))?;
    let mut state = ctx.create_state().context("Failed to create whisper state")?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(4);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_translate(false);
    params.set_language(Some("en"));
    state
        .full(params, audio)
        .context("Offline whisper transcription failed")?;

    let mut out = String::new();
    let segments = state.full_n_segments().context("Could not read whisper segments")?;
    for i in 0..segments {
        let seg = state
            .full_get_segment_text(i)
            .context("Could not read whisper segment text")?;
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(seg.trim());
    }
    Ok(out.trim().to_string())
}

fn transcribe_cloud_wav_bytes(cfg: &CloudSttConfig, wav: Vec<u8>) -> Result<String> {
    let (url, model, api_key) = match &cfg.backend {
        SttBackend::Cloud {
            url,
            model,
            api_key,
            ..
        } => (url, model, api_key),
        _ => anyhow::bail!("Expected cloud STT backend"),
    };
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| Client::new());
    let part = multipart::Part::bytes(wav)
        .file_name("chunk.wav")
        .mime_str("audio/wav")
        .context("multipart mime")?;
    let form = multipart::Form::new()
        .part("file", part)
        .text("model", model.clone())
        .text("response_format", "json");
    let resp = client
        .post(url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .context("STT HTTP request failed")?;
    let status = resp.status();
    let body = resp.text().context("STT empty response body")?;
    if !status.is_success() {
        anyhow::bail!("STT HTTP {status}: {}", body.trim().chars().take(220).collect::<String>());
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body)
        && let Some(t) = v.get("text").and_then(|x| x.as_str())
    {
        return Ok(t.trim().to_string());
    }
    Ok(body.trim().to_string())
}

fn pcm_to_f32_mono_16k(pcm: &[i16], sample_rate: u32) -> Vec<f32> {
    let src: Vec<f32> = pcm.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
    if sample_rate == 16_000 {
        return src;
    }
    if src.is_empty() {
        return src;
    }
    let ratio = 16_000.0f32 / sample_rate as f32;
    let out_len = ((src.len() as f32) * ratio).max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f32 / ratio;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f32;
        let a = src[idx.min(src.len() - 1)];
        let b = src[(idx + 1).min(src.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(feature = "local_voice")]
fn resolve_whisper_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OPENCAGE_WHISPER_MODEL") {
        let p = PathBuf::from(path.trim());
        if p.exists() {
            return Some(p);
        }
    }
    let mut candidates = Vec::new();
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".cache/opencage/models/ggml-base.en.bin"));
        candidates.push(home.join(".cache/opencage/models/ggml-small.en.bin"));
        candidates.push(home.join(".cache/whisper/ggml-base.en.bin"));
        candidates.push(home.join(".cache/whisper/ggml-small.en.bin"));
    }
    candidates.push(PathBuf::from("models/ggml-base.en.bin"));
    candidates.push(PathBuf::from("models/ggml-small.en.bin"));
    candidates.into_iter().find(|p| p.exists())
}
