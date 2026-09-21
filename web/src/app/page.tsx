"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { API_BASE, Health, api } from "@/lib/api";
import { Logo } from "@/components/ui";

const features = [
  { t: "First audio in ~150 ms", d: "Token-level streaming with CUDA-graphed decoders. Your agent starts talking before the sentence is finished generating.", k: "latency" },
  { t: "Clone a voice from 6 seconds", d: "Upload one clean clip per emotional register. Calm, upbeat, urgent — pick per request with a voice_id.", k: "clone" },
  { t: "Drop-in for ElevenLabs & OpenAI", d: "Same request shapes, same output formats (pcm, µ-law 8k, mp3, opus). Change the base URL, keep your code.", k: "compat" },
  { t: "Hindi + 22 languages", d: "Devanagari is auto-detected and routed to the multilingual model. Hinglish agents just work.", k: "lang" },
  { t: "Telephony-ready", d: "µ-law / A-law at 8 kHz straight out of the API for Twilio, Exotel, Plivo and SIP media streams.", k: "tel" },
  { t: "Your data stays with you", d: "Self-hosted on your GPUs or ours in-region. No transcripts leave the box. Outputs are watermarked.", k: "priv" },
];

const plans = [
  { name: "Free", price: "₹0", unit: "forever", chars: "50k chars / mo", extras: ["1 cloned voice", "1 concurrent stream", "Community support"], cta: "Start free" },
  { name: "Starter", price: "₹1,999", unit: "/ month", chars: "2M chars / mo (~37 h)", extras: ["10 cloned voices", "3 concurrent streams", "Email support", "≈ ₹0.9 / min of audio"], cta: "Get Starter", hot: true },
  { name: "Scale", price: "₹7,999", unit: "/ month", chars: "10M chars / mo (~185 h)", extras: ["50 cloned voices", "8 concurrent streams", "Priority + Slack support", "≈ ₹0.7 / min of audio"], cta: "Talk to us" },
];

export default function Landing() {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    api<Health>("/health").then(setHealth).catch(() => setHealth(null));
  }, []);
  return (
    <main className="relative">
      <div className="grid-bg pointer-events-none absolute inset-0 h-[720px]" />
      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <Logo size={26} />
        <nav className="flex items-center gap-2 text-sm">
          <a href="#pricing" className="btn btn-ghost">Pricing</a>
          <Link href="/app/docs/" className="btn btn-ghost">Docs</Link>
          <Link href="/app/" className="btn btn-primary">Open console →</Link>
        </nav>
      </header>

      <section className="relative mx-auto max-w-6xl px-6 pb-16 pt-14 md:pt-24">
        <div className="pill mb-6">
          <span className="live-dot" /> {health ? `Live · ${health.gpu?.name ?? "GPU"} · ${health.models.join(" + ")}` : "Beta"}
        </div>
        <h1 className="max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tight md:text-6xl">
          The voice for your voice agents. <span className="gradient-text">Your voice.</span> Your servers.
        </h1>
        <p className="mt-6 max-w-2xl text-lg text-fg-2">
          Streaming text-to-speech with zero-shot voice cloning, built for real-time conversations. OpenAI- and ElevenLabs-compatible, so your stack swaps over with one base-URL change — at a fraction of the price.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link href="/app/" className="btn btn-primary !px-5 !py-3 text-base">Try the playground</Link>
          <a href="#quickstart" className="btn btn-ghost !px-5 !py-3 text-base">See the one-line swap</a>
        </div>
        <div className="mt-12 grid max-w-3xl grid-cols-2 gap-3 md:grid-cols-4">
          {[
            ["~150 ms", "first audio (English)"],
            ["~450 ms", "first audio (Hindi)"],
            ["0.3×", "realtime factor"],
            ["23", "languages"],
          ].map(([v, l]) => (
            <div key={l} className="card p-4">
              <div className="text-2xl font-semibold tabular-nums">{v}</div>
              <div className="text-xs text-fg-2">{l}</div>
            </div>
          ))}
        </div>
      </section>

      <section id="quickstart" className="mx-auto max-w-6xl px-6 py-12">
        <h2 className="text-2xl font-semibold tracking-tight">Swap ElevenLabs in one line</h2>
        <p className="mt-2 max-w-2xl text-fg-2">Point your existing client at this server. Bolna, Vapi, LiveKit Agents, Pipecat and the OpenAI SDK all work unchanged.</p>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <pre className="code">{`# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="${API_BASE || "https://tts.yourdomain.com"}/v1",
                api_key="va_...")

with client.audio.speech.with_streaming_response.create(
    model="chatterbox", voice="ishan_calm",
    input="Hello, am I speaking to Amit?",
    response_format="pcm",
) as r:
    for chunk in r.iter_bytes():
        play(chunk)`}</pre>
          <pre className="code">{`# ElevenLabs-shaped request (Bolna, custom stacks)
curl -N ${API_BASE || "https://tts.yourdomain.com"}/v1/text-to-speech/ishan_calm/stream \\
  -H "xi-api-key: va_..." \\
  -H "content-type: application/json" \\
  -d '{"text":"Your appointment is confirmed.",
       "model_id":"eleven_turbo_v2_5"}' \\
  "?output_format=ulaw_8000" > call.ulaw`}</pre>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid gap-4 md:grid-cols-3">
          {features.map((f) => (
            <div key={f.k} className="card p-5">
              <div className="mb-2 h-1.5 w-10 rounded-full" style={{ background: "linear-gradient(90deg,var(--accent),var(--accent-2))" }} />
              <div className="font-medium">{f.t}</div>
              <div className="mt-1 text-sm text-fg-2">{f.d}</div>
            </div>
          ))}
        </div>
      </section>

      <section id="pricing" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-2xl font-semibold tracking-tight">Simple, character-based pricing</h2>
        <p className="mt-2 max-w-2xl text-fg-2">Roughly 900 characters per minute of speech. ElevenLabs' conversational tiers land around ₹5–8 per minute; we're under ₹1.</p>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {plans.map((p) => (
            <div key={p.name} className={`card p-6 ${p.hot ? "glow border-accent/60" : ""}`}>
              <div className="flex items-center justify-between">
                <div className="font-medium">{p.name}</div>
                {p.hot && <span className="pill border-accent/50 text-accent">Most popular</span>}
              </div>
              <div className="mt-3 text-3xl font-semibold">
                {p.price} <span className="text-sm font-normal text-fg-2">{p.unit}</span>
              </div>
              <div className="mt-1 text-sm text-fg-2">{p.chars}</div>
              <ul className="mt-4 space-y-1.5 text-sm">
                {p.extras.map((e) => (
                  <li key={e} className="flex gap-2">
                    <span className="text-accent-2">✓</span>
                    {e}
                  </li>
                ))}
              </ul>
              <Link href="/app/" className={`btn mt-6 w-full justify-center ${p.hot ? "btn-primary" : "btn-ghost"}`}>{p.cta}</Link>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-fg-3">Need on-prem, dedicated GPUs, or an SLA? Enterprise plans start at ₹49,999 / month.</p>
      </section>

      <footer className="mx-auto max-w-6xl px-6 py-10 text-xs text-fg-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Logo size={18} />
          <div>Built on Chatterbox (MIT) · outputs carry an inaudible watermark · clone only voices you have permission to use.</div>
        </div>
      </footer>
    </main>
  );
}
