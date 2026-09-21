"use client";

import { useEffect, useRef, useState } from "react";
import { Voice, api, fmtMs, speak } from "@/lib/api";
import { PcmPlayer, pcmToWav } from "@/lib/audio";
import { Field, Stat, Toast } from "@/components/ui";
import { useSession } from "./layout";

const SAMPLES: Record<string, string> = {
  en: "Hello, am I speaking to Amit? Great. We found a few job profiles that match your CV, and I wanted to walk you through them. Do you have two minutes?",
  hi: "नमस्ते, क्या मैं अमित से बात कर रहा हूँ? हमने आपके सीवी के आधार पर कई नौकरियाँ पाई हैं। क्या आपके पास दो मिनट हैं?",
};

const LANGS = [
  ["auto", "Auto-detect"], ["en", "English"], ["hi", "Hindi"], ["es", "Spanish"], ["fr", "French"], ["de", "German"], ["ar", "Arabic"], ["pt", "Portuguese"],
  ["it", "Italian"], ["ja", "Japanese"], ["ko", "Korean"], ["zh", "Chinese"], ["ru", "Russian"], ["tr", "Turkish"], ["nl", "Dutch"], ["pl", "Polish"], ["sv", "Swedish"],
  ["da", "Danish"], ["fi", "Finnish"], ["no", "Norwegian"], ["el", "Greek"], ["he", "Hebrew"], ["ms", "Malay"], ["sw", "Swahili"],
];

export default function Playground() {
  const { health } = useSession();
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("default");
  const [lang, setLang] = useState("auto");
  const [text, setText] = useState(SAMPLES.en);
  const [speed, setSpeed] = useState(1);
  const [exag, setExag] = useState(0.5);
  const [temp, setTemp] = useState(0.8);
  const [cfg, setCfg] = useState(0.5);
  const [seed, setSeed] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [ttfb, setTtfb] = useState<number | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [secs, setSecs] = useState<number | null>(null);
  const [wavUrl, setWavUrl] = useState<string | null>(null);
  const [toast, setToast] = useState<{ msg: string; tone?: "ok" | "err" } | null>(null);
  const [history, setHistory] = useState<{ text: string; voice: string; ttfb: number; secs: number; url: string }[]>([]);
  const player = useRef<PcmPlayer | null>(null);
  const abort = useRef<AbortController | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const raf = useRef<number>(0);

  useEffect(() => {
    api<{ voices: Voice[] }>("/v1/voices").then((r) => setVoices(r.voices)).catch(() => {});
    return () => {
      cancelAnimationFrame(raf.current);
      player.current?.close();
    };
  }, []);

  function draw() {
    const p = player.current;
    const c = canvas.current;
    if (!p || !c) return;
    const ctx = c.getContext("2d")!;
    const data = new Uint8Array(p.analyser.frequencyBinCount);
    const loop = () => {
      p.analyser.getByteFrequencyData(data);
      const w = c.width;
      const h = c.height;
      ctx.clearRect(0, 0, w, h);
      const bars = 48;
      const step = Math.floor(data.length / bars);
      const g = ctx.createLinearGradient(0, h, 0, 0);
      g.addColorStop(0, "#7c5cff");
      g.addColorStop(1, "#38d6c4");
      ctx.fillStyle = g;
      for (let i = 0; i < bars; i++) {
        const v = data[i * step] / 255;
        const bh = Math.max(2, v * h);
        const bw = w / bars - 3;
        ctx.beginPath();
        ctx.roundRect(i * (w / bars), h - bh, bw, bh, 2);
        ctx.fill();
      }
      raf.current = requestAnimationFrame(loop);
    };
    cancelAnimationFrame(raf.current);
    loop();
  }

  async function run() {
    if (!text.trim()) return;
    setBusy(true);
    setTtfb(null);
    setTotal(null);
    setSecs(null);
    abort.current?.abort();
    abort.current = new AbortController();
    player.current?.close();
    player.current = new PcmPlayer(24000);
    await player.current.resume();
    draw();
    try {
      const res = await speak(
        {
          input: text,
          voice,
          language: lang === "auto" ? undefined : lang,
          speed,
          exaggeration: exag,
          temperature: temp,
          cfg_weight: cfg,
          seed: seed.trim() ? Number(seed) : undefined,
        },
        (chunk, t) => {
          player.current?.push(chunk);
          if (t !== null) setTtfb((v) => v ?? t);
        },
        abort.current.signal
      );
      const s = res.bytes.length / 2 / res.sampleRate;
      setTotal(res.totalMs);
      setSecs(s);
      const url = URL.createObjectURL(pcmToWav(res.bytes, res.sampleRate));
      setWavUrl(url);
      setHistory((h) => [{ text, voice, ttfb: res.ttfbMs, secs: s, url }, ...h].slice(0, 8));
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setToast({ msg: e instanceof Error ? e.message : "request failed", tone: "err" });
    } finally {
      setBusy(false);
    }
  }

  function stop() {
    abort.current?.abort();
    player.current?.stop();
    setBusy(false);
  }

  const v = voices.find((x) => x.voice_id === voice);
  const rtf = total && secs ? total / 1000 / secs : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Playground</h1>
          <p className="text-sm text-fg-2">Type, press speak, hear it stream. Numbers below are measured from this browser.</p>
        </div>
        {health && (
          <div className="pill">
            <span className="live-dot" /> {health.models.includes("turbo") ? "Turbo" : ""}{health.models.includes("multilingual") ? " + Multilingual" : ""} warm
          </div>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Time to first audio" value={fmtMs(ttfb)} tone={ttfb == null ? undefined : ttfb < 300 ? "good" : ttfb < 700 ? "warn" : "bad"} sub="from request to first byte" />
        <Stat label="Total generation" value={fmtMs(total)} sub={secs ? `${secs.toFixed(1)} s of audio` : " "} />
        <Stat label="Realtime factor" value={rtf ? `${rtf.toFixed(2)}×` : "–"} sub="< 1.0 keeps up with playback" tone={rtf == null ? undefined : rtf < 0.8 ? "good" : "warn"} />
        <Stat label="Voice" value={<span className="truncate text-lg">{voice}</span>} sub={v?.builtin ? "built-in" : v?.description || "cloned"} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="card p-5">
          <textarea className="textarea min-h-[160px] resize-y text-base" value={text} onChange={(e) => setText(e.target.value)} maxLength={4096} />
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-fg-3">
            <span>{text.length} / 4096</span>
            <span>·</span>
            <button className="underline" onClick={() => setText(SAMPLES.en)}>English sample</button>
            <button className="underline" onClick={() => setText(SAMPLES.hi)}>Hindi sample</button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            {!busy ? (
              <button className="btn btn-primary !px-5" onClick={run} disabled={!text.trim()}>
                ▶ Speak
              </button>
            ) : (
              <button className="btn btn-danger !px-5" onClick={stop}>
                ■ Stop
              </button>
            )}
            {busy && (
              <span className="eq" aria-hidden>
                <span /><span /><span /><span />
              </span>
            )}
            {wavUrl && !busy && (
              <a className="btn btn-ghost" href={wavUrl} download={`voiceagent-${voice}.wav`}>
                ↓ Download wav
              </a>
            )}
          </div>
          <canvas ref={canvas} width={960} height={90} className="mt-5 h-[90px] w-full rounded-lg bg-bg-elev" />
        </div>

        <div className="card space-y-4 p-5">
          <Field label="Voice">
            <select className="select" value={voice} onChange={(e) => setVoice(e.target.value)}>
              {voices.map((x) => (
                <option key={x.voice_id} value={x.voice_id}>
                  {x.voice_id}{x.builtin ? " (built-in)" : ""}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Language" hint="auto = detect script">
            <select className="select" value={lang} onChange={(e) => setLang(e.target.value)}>
              {LANGS.map(([c, n]) => (
                <option key={c} value={c}>{n}</option>
              ))}
            </select>
          </Field>
          <Field label={`Speed · ${speed.toFixed(2)}×`}>
            <input type="range" min={0.7} max={1.4} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label={`Variation · ${temp.toFixed(2)}`} hint="lower = steadier, higher = livelier">
            <input type="range" min={0.3} max={1.2} step={0.05} value={temp} onChange={(e) => setTemp(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label={`Emotion · ${exag.toFixed(2)}`} hint="Hindi & other languages">
            <input type="range" min={0.2} max={1.2} step={0.05} value={exag} onChange={(e) => setExag(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label={`Guidance · ${cfg.toFixed(2)}`} hint="Hindi & other languages">
            <input type="range" min={0.1} max={0.9} step={0.05} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label="Seed" hint="same seed = same take">
            <input className="input mono" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))} />
          </Field>
          <div className="rounded-lg border border-dashed p-3 text-xs leading-relaxed text-fg-3">
            English → <b>Turbo</b> (~150 ms): speed, variation, seed apply. Other languages → <b>Multilingual</b> (~450 ms): all sliders apply. Emotion 0.5 is neutral; guidance 0.5 is the default — lower it if a voice sounds strained.
          </div>
        </div>
      </div>

      {history.length > 0 && (
        <div className="card overflow-hidden">
          <div className="border-b px-5 py-3 text-sm font-medium">Recent</div>
          <table className="w-full text-sm">
            <tbody>
              {history.map((h, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="max-w-[420px] truncate px-5 py-2 text-fg-2">{h.text}</td>
                  <td className="px-3 py-2 mono text-xs">{h.voice}</td>
                  <td className="px-3 py-2 tabular-nums text-xs">{fmtMs(h.ttfb)}</td>
                  <td className="px-3 py-2 tabular-nums text-xs">{h.secs.toFixed(1)} s</td>
                  <td className="px-3 py-2">
                    <audio src={h.url} controls className="h-8" preload="none" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {toast && <Toast msg={toast.msg} tone={toast.tone} onDone={() => setToast(null)} />}
    </div>
  );
}
