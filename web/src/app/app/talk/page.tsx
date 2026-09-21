"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE, Voice, api, fmtMs, getKey, preferredVoice, rememberVoice } from "@/lib/api";
import { PcmPlayer } from "@/lib/audio";
import { Field, Stat } from "@/components/ui";

type Msg = { role: "user" | "assistant"; text: string; pending?: boolean };
type Timings = { stt_ms: number | null; llm_first_token_ms: number | null; first_audio_ms: number | null; total_ms: number | null; client_first_audio_ms?: number | null };

const PERSONAS: Record<string, string> = {
  assistant: "You are a friendly, concise voice assistant on a phone call. Reply in one to three short sentences. Never use markdown, lists, emojis or symbols - only plain spoken sentences.",
  recruiter: "You are Priya, a recruiter calling a candidate about full-stack developer roles. Be warm and brisk. Confirm you are speaking to the right person, then explain we found matching roles and ask them to update their CV on our website. One to two short sentences per turn.",
  support: "You are a calm customer support agent for an online store. Ask one clarifying question at a time, keep replies under two sentences, never use lists or markdown.",
};

export default function Talk() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("default");
  const [language, setLanguage] = useState("auto");
  const [speed, setSpeed] = useState(() => {
    try {
      return Number(localStorage.getItem("va_talk_speed") || 1.1);
    } catch {
      return 1.1;
    }
  });
  const [persona, setPersona] = useState("assistant");
  const [system, setSystem] = useState(PERSONAS.assistant);
  const [status, setStatus] = useState<{ llm: string; llm_ok: boolean; stt: string | false } | null>(null);
  const [conn, setConn] = useState<"connecting" | "ready" | "closed">("connecting");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [phase, setPhase] = useState<"idle" | "recording" | "hearing" | "thinking" | "speaking">("idle");
  const [timings, setTimings] = useState<Timings | null>(null);
  const [typed, setTyped] = useState("");
  const [err, setErr] = useState("");
  const ws = useRef<WebSocket | null>(null);
  const player = useRef<PcmPlayer | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const tRelease = useRef<number>(0);
  const clientFirst = useRef<number | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api<{ voices: Voice[] }>("/v1/voices")
      .then((r) => {
        setVoices(r.voices);
        setVoice(preferredVoice(r.voices));
      })
      .catch(() => {});
    connect();
    return () => {
      ws.current?.close();
      player.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, phase]);

  useEffect(() => {
    send({ type: "config", voice, language: language === "auto" ? null : language, system, speed });
    try {
      localStorage.setItem("va_talk_speed", String(speed));
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice, language, system, speed, conn]);

  function connect() {
    const url = `${API_BASE.replace(/^http/, "ws")}/v1/talk?api_key=${encodeURIComponent(getKey())}`;
    const s = new WebSocket(url);
    s.binaryType = "arraybuffer";
    ws.current = s;
    setConn("connecting");
    s.onopen = () => setConn("ready");
    s.onclose = () => setConn("closed");
    s.onerror = () => setErr("connection failed - is the server running?");
    s.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        if (clientFirst.current === null && tRelease.current) clientFirst.current = performance.now() - tRelease.current;
        setPhase("speaking");
        player.current?.push(new Uint8Array(ev.data));
        return;
      }
      const m = JSON.parse(ev.data);
      switch (m.type) {
        case "ready":
          setStatus({ llm: m.llm, llm_ok: m.llm_ok, stt: m.stt });
          break;
        case "transcript":
          setMsgs((h) => [...h.filter((x) => !x.pending), { role: "user", text: m.text }, { role: "assistant", text: "", pending: true }]);
          setPhase("thinking");
          break;
        case "token":
          setMsgs((h) => h.map((x, i) => (i === h.length - 1 && x.role === "assistant" ? { ...x, text: x.text + m.text } : x)));
          break;
        case "done":
          setMsgs((h) => h.map((x) => ({ ...x, pending: false })));
          setTimings({ ...m.timings, client_first_audio_ms: clientFirst.current });
          // stay in "speaking" until the buffered audio drains
          waitForSilence();
          break;
        case "error":
          setErr(m.message);
          setMsgs((h) => h.filter((x) => !x.pending));
          setPhase("idle");
          break;
        case "reset":
          setMsgs([]);
          break;
      }
    };
  }

  function send(obj: unknown) {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(obj));
  }

  function waitForSilence() {
    const tick = () => {
      const buffered = player.current?.buffered() ?? 0;
      if (buffered > 0.05) setTimeout(tick, 100);
      else setPhase("idle");
    };
    tick();
  }

  async function ensurePlayer() {
    if (!player.current) player.current = new PcmPlayer(24000);
    await player.current.resume();
  }

  async function startRec() {
    setErr("");
    await ensurePlayer();
    // barge-in: stop whatever it is saying
    player.current?.stop();
    send({ type: "cancel" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      const mr = new MediaRecorder(stream);
      chunks.current = [];
      mr.ondataavailable = (e) => chunks.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: mr.mimeType || "audio/webm" });
        const b64 = await blobToB64(blob);
        tRelease.current = performance.now();
        clientFirst.current = null;
        setTimings(null);
        setPhase("hearing");
        setMsgs((h) => [...h, { role: "user", text: "…", pending: true }]);
        send({ type: "turn", audio: b64, mime: blob.type });
      };
      mr.start();
      rec.current = mr;
      setPhase("recording");
    } catch {
      setErr("microphone access denied");
    }
  }
  function stopRec() {
    if (rec.current?.state === "recording") rec.current.stop();
    rec.current = null;
  }

  async function sendTyped() {
    if (!typed.trim()) return;
    await ensurePlayer();
    player.current?.stop();
    tRelease.current = performance.now();
    clientFirst.current = null;
    setTimings(null);
    setPhase("hearing");
    send({ type: "text", text: typed.trim() });
    setTyped("");
  }

  const recording = phase === "recording";
  const disabled = conn !== "ready" || (status !== null && !status.llm_ok);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Talk</h1>
          <p className="text-sm text-fg-2">Hold the button, speak, let go. Everything runs on this machine: Whisper → LLM → <span className="mono">{voice}</span>.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className={`pill ${conn === "ready" ? "text-good" : "text-warn"}`}>{conn === "ready" ? "● connected" : conn}</span>
          {status && <span className={`pill ${status.llm_ok ? "" : "text-bad"}`} title={status.llm}>LLM: {status.llm_ok ? status.llm.split(" @ ")[0] : "unavailable"}</span>}
          {status && <span className="pill">{status.stt ? "Whisper ready" : "STT off"}</span>}
        </div>
      </div>

      {status && !status.llm_ok && (
        <div className="card border-bad/40 p-4 text-sm">
          <b>No language model reachable.</b> {status.llm}. Install Ollama and run <span className="mono">ollama pull qwen2.5:3b-instruct</span>, or point <span className="mono">TTS_LLM_BASE_URL</span> at any OpenAI-compatible endpoint.
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Heard you" value={fmtMs(timings?.stt_ms)} sub="speech → text" tone={timings?.stt_ms == null ? undefined : timings.stt_ms < 400 ? "good" : "warn"} />
        <Stat label="Started thinking" value={fmtMs(timings?.llm_first_token_ms)} sub="first LLM token" />
        <Stat label="Started speaking" value={fmtMs(timings?.first_audio_ms)} sub={timings?.client_first_audio_ms ? `${Math.round(timings.client_first_audio_ms)} ms as you heard it` : "first audio byte"} tone={timings?.first_audio_ms == null ? undefined : timings.first_audio_ms < 1000 ? "good" : "warn"} />
        <Stat label="Finished" value={fmtMs(timings?.total_ms)} sub="whole reply generated" />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="card flex min-h-[520px] flex-col p-5">
          <div className="flex-1 space-y-3 overflow-y-auto pr-1">
            {msgs.length === 0 && (
              <div className="flex h-full items-center justify-center text-center text-sm text-fg-3">
                Say hello. Try &quot;Hi, who am I speaking to?&quot; or ask something in Hindi.
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${m.role === "user" ? "bg-accent text-white" : "bg-bg-elev"}`}>
                  {m.text || <span className="eq" aria-hidden><span /><span /><span /><span /></span>}
                </div>
              </div>
            ))}
            <div ref={bottom} />
          </div>

          <div className="mt-4 flex flex-col items-center gap-3 border-t pt-4">
            <button
              className={`flex h-24 w-24 select-none items-center justify-center rounded-full text-3xl transition ${recording ? "scale-110 bg-bad text-white shadow-[0_0_0_12px_rgba(255,93,108,0.25)]" : "bg-accent text-white hover:brightness-110"} disabled:opacity-40`}
              disabled={disabled}
              onMouseDown={startRec}
              onMouseUp={stopRec}
              onMouseLeave={() => recording && stopRec()}
              onTouchStart={(e) => {
                e.preventDefault();
                startRec();
              }}
              onTouchEnd={(e) => {
                e.preventDefault();
                stopRec();
              }}
              aria-label="hold to talk"
            >
              {recording ? "●" : "🎙"}
            </button>
            <div className="h-5 text-xs text-fg-3">
              {phase === "recording" && "Listening… release to send"}
              {phase === "hearing" && "Transcribing…"}
              {phase === "thinking" && "Thinking…"}
              {phase === "speaking" && "Speaking — press the button to interrupt"}
              {phase === "idle" && "Hold to talk"}
            </div>
            <form
              className="flex w-full max-w-md gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                sendTyped();
              }}
            >
              <input className="input" placeholder="or type a message…" value={typed} onChange={(e) => setTyped(e.target.value)} disabled={disabled} />
              <button className="btn btn-ghost" type="submit" disabled={disabled || !typed.trim()}>
                Send
              </button>
            </form>
            {err && <div className="text-xs text-bad">{err}</div>}
          </div>
        </div>

        <div className="card space-y-4 p-5">
          <Field label="Voice">
            <select
              className="select"
              value={voice}
              onChange={(e) => {
                setVoice(e.target.value);
                rememberVoice(e.target.value);
              }}
            >
              {voices.map((x) => (
                <option key={x.voice_id} value={x.voice_id}>{x.voice_id}{x.builtin ? " (built-in)" : ""}</option>
              ))}
            </select>
          </Field>
          <Field label="Language" hint="auto = detect from speech">
            <select className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </Field>
          <Field label={`Speaking speed · ${speed.toFixed(2)}×`} hint="pitch stays the same">
            <input type="range" min={0.8} max={1.5} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label="Persona">
            <select
              className="select"
              value={persona}
              onChange={(e) => {
                setPersona(e.target.value);
                setSystem(PERSONAS[e.target.value]);
              }}
            >
              <option value="assistant">Assistant</option>
              <option value="recruiter">Recruiter (Priya)</option>
              <option value="support">Support agent</option>
            </select>
          </Field>
          <Field label="System prompt">
            <textarea className="textarea min-h-[140px] text-xs" value={system} onChange={(e) => setSystem(e.target.value)} />
          </Field>
          <p className="-mt-2 text-[11px] text-fg-3">The reply language is added automatically from what you speak (or the Language setting) - no need to mention it here.</p>
          <button className="btn btn-ghost w-full justify-center" onClick={() => send({ type: "reset" })}>
            Clear conversation
          </button>
          <div className="rounded-lg border border-dashed p-3 text-[11px] leading-relaxed text-fg-3">
            The pipeline starts speaking as soon as the first sentence of the reply is complete, while the LLM is still writing the rest. Pressing the button mid-reply interrupts it.
          </div>
        </div>
      </div>
    </div>
  );
}

function blobToB64(blob: Blob): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res((r.result as string).split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}
