"use client";

import { useState } from "react";
import { API_BASE, getKey } from "@/lib/api";
import { Copy } from "@/components/ui";

function Block({ title, code, note }: { title: string; code: string; note?: string }) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between border-b px-5 py-3">
        <div className="text-sm font-medium">{title}</div>
        <Copy text={code} />
      </div>
      {note && <div className="border-b px-5 py-2 text-xs text-fg-2">{note}</div>}
      <pre className="code !rounded-none !border-0">{code}</pre>
    </div>
  );
}

export default function Docs() {
  const [showKey, setShowKey] = useState(false);
  const base = API_BASE || "https://tts.yourdomain.com";
  const key = showKey ? getKey() || "va_YOUR_KEY" : "va_YOUR_KEY";

  const tabs: { id: string; label: string; blocks: { title: string; code: string; note?: string }[] }[] = [
    {
      id: "openai",
      label: "OpenAI SDK",
      blocks: [
        {
          title: "Python — streaming PCM",
          note: "response_format=pcm is 16-bit signed little-endian, 24 kHz mono. Lowest latency. Ask for a different rate with sample_rate.",
          code: `from openai import OpenAI

client = OpenAI(base_url="${base}/v1", api_key="${key}")

with client.audio.speech.with_streaming_response.create(
    model="chatterbox",
    voice="ishan_calm",          # any voice_id from /v1/voices
    input="Hello, am I speaking to Amit?",
    response_format="pcm",
) as response:
    for chunk in response.iter_bytes(chunk_size=4096):
        player.write(chunk)`,
        },
        {
          title: "Node — mp3 to file",
          code: `import OpenAI from "openai";
import fs from "node:fs";

const openai = new OpenAI({ baseURL: "${base}/v1", apiKey: "${key}" });
const res = await openai.audio.speech.create({
  model: "chatterbox", voice: "ishan_calm",
  input: "Your order is on its way.", response_format: "mp3",
});
fs.writeFileSync("out.mp3", Buffer.from(await res.arrayBuffer()));`,
        },
        {
          title: "Extra fields (optional)",
          note: "All optional. language forces routing (auto-detects Devanagari otherwise).",
          code: `{
  "input": "...", "voice": "ishan_calm",
  "response_format": "pcm | wav | mp3 | opus | aac | flac | mulaw | alaw",
  "sample_rate": 16000,        // resample (mulaw/alaw default to 8000)
  "speed": 1.1,                // 0.5 – 2.0, pitch preserved
  "language": "hi",            // ISO-639-1
  "exaggeration": 0.6,         // expressiveness (multilingual model)
  "temperature": 0.8,
  "seed": 42,                  // reproducible output
  "stream": true               // false = buffered response with Content-Length
}`,
        },
      ],
    },
    {
      id: "eleven",
      label: "ElevenLabs shape",
      blocks: [
        {
          title: "Drop-in for clients that speak the ElevenLabs API",
          note: "voice_settings.stability / style map onto expressiveness; output_format uses ElevenLabs names.",
          code: `curl -N "${base}/v1/text-to-speech/ishan_calm/stream?output_format=pcm_16000" \\
  -H "xi-api-key: ${key}" \\
  -H "content-type: application/json" \\
  -d '{
    "text": "Your appointment is confirmed for Tuesday at 4 PM.",
    "model_id": "eleven_turbo_v2_5",
    "voice_settings": {"stability": 0.5, "style": 0.3}
  }' > out.pcm

# output_format: mp3_44100_128, pcm_16000, pcm_24000, ulaw_8000, alaw_8000, opus_48000`,
        },
        {
          title: "List voices (same shape as ElevenLabs)",
          code: `curl ${base}/v1/voices -H "xi-api-key: ${key}"`,
        },
      ],
    },
    {
      id: "bolna",
      label: "Bolna / Vapi / LiveKit",
      blocks: [
        {
          title: "Bolna — custom ElevenLabs endpoint",
          note: "Bolna's ElevenLabs provider takes a base URL override. Point it here and keep the voice_id equal to your cloned voice name.",
          code: `# In the agent's synthesizer config (or the ELEVENLABS_* env of a self-hosted Bolna):
{
  "provider": "elevenlabs",
  "provider_config": {
    "voice": "ishan_calm",
    "voice_id": "ishan_calm",
    "model": "eleven_turbo_v2_5",
    "base_url": "${base}",        # <- the only change
    "api_key": "${key}"
  },
  "stream": true,
  "audio_format": "pcm"
}`,
        },
        {
          title: "Vapi — custom voice provider",
          code: `{
  "voice": {
    "provider": "custom-voice",
    "server": {
      "url": "${base}/v1/audio/speech",
      "headers": { "Authorization": "Bearer ${key}" }
    },
    "voiceId": "ishan_calm"
  }
}`,
        },
        {
          title: "LiveKit Agents / Pipecat — OpenAI TTS plugin",
          code: `# LiveKit
from livekit.plugins import openai
tts = openai.TTS(base_url="${base}/v1", api_key="${key}",
                 model="chatterbox", voice="ishan_calm", response_format="pcm")

# Pipecat
from pipecat.services.openai.tts import OpenAITTSService
tts = OpenAITTSService(base_url="${base}/v1", api_key="${key}",
                       model="chatterbox", voice="ishan_calm")`,
        },
        {
          title: "Twilio media streams — µ-law 8 kHz",
          code: `curl -N ${base}/v1/audio/speech \\
  -H "Authorization: Bearer ${key}" -H "content-type: application/json" \\
  -d '{"input":"Thanks for calling.","voice":"ishan_calm","response_format":"mulaw"}' \\
  | base64 | ...   # feed as {"event":"media","media":{"payload":...}} frames`,
        },
      ],
    },
    {
      id: "voices",
      label: "Voices API",
      blocks: [
        {
          title: "Clone from a clip",
          note: "6–15 s, one speaker, no music. wav/flac/mp3/ogg/m4a. Returns when the voice is ready on every loaded model (~0.3 s).",
          code: `curl -X POST ${base}/v1/voices \\
  -H "Authorization: Bearer ${key}" \\
  -F name=ishan_calm -F language=en -F description="Neutral support tone" \\
  -F file=@ishan_calm.wav`,
        },
        {
          title: "Multiple registers = multiple voices",
          note: "Record one clip per mood and pick per request. The LLM can choose the voice_id from its own sentiment.",
          code: `voices = {"neutral": "ishan_calm", "happy": "ishan_upbeat", "urgent": "ishan_firm"}
voice = voices[classify(reply_text)]
client.audio.speech.create(model="chatterbox", voice=voice, input=reply_text, ...)`,
        },
        { title: "Delete", code: `curl -X DELETE ${base}/v1/voices/ishan_calm -H "Authorization: Bearer ${key}"` },
      ],
    },
    {
      id: "errors",
      label: "Limits & errors",
      blocks: [
        {
          title: "Error envelope (OpenAI style)",
          code: `HTTP 429
{"detail": {"message": "concurrency limit: 1 simultaneous streams on the free plan",
            "type": "invalid_request_error", "code": "concurrency_limit"}}

codes: missing_api_key · invalid_api_key · rate_limit_exceeded · insufficient_quota · concurrency_limit`,
        },
        {
          title: "Good to know",
          code: `• input is capped at 4096 characters per request; split long scripts by paragraph
• keep the HTTP connection alive - a fresh TCP connection adds ~100-200 ms to first audio
• pcm/wav/mulaw stream with zero encoder delay; mp3/opus add ~100 ms
• first audio: ~150 ms (English/Turbo), ~450 ms (other languages/Multilingual)
• GET /health for readiness, GET /v1/usage for your metering`,
        },
      ],
    },
  ];
  const [tab, setTab] = useState(tabs[0].id);
  const cur = tabs.find((t) => t.id === tab)!;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Docs</h1>
          <p className="text-sm text-fg-2">
            Base URL <span className="mono">{base}</span>. Everything below is copy-paste ready.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-fg-2">
          <input type="checkbox" checked={showKey} onChange={(e) => setShowKey(e.target.checked)} /> insert my key into examples
        </label>
      </div>
      <div className="flex flex-wrap gap-1">
        {tabs.map((t) => (
          <button key={t.id} className={`btn !py-1.5 !px-3 text-xs ${tab === t.id ? "btn-primary" : "btn-ghost"}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="space-y-4">
        {cur.blocks.map((b) => (
          <Block key={b.title} {...b} />
        ))}
      </div>
    </div>
  );
}
