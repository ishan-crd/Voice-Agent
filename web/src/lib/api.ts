"use client";

// Thin client for the Voice-Agent API. The key lives in localStorage for the
// beta (single-user dashboards); swap for a session cookie when accounts get
// passwords.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  (typeof window !== "undefined" ? window.location.origin : "");

const KEY = "va_api_key";

export function getKey(): string {
  try {
    return localStorage.getItem(KEY) || "";
  } catch {
    return "";
  }
}
export function setKey(k: string) {
  try {
    if (k) localStorage.setItem(KEY, k);
    else localStorage.removeItem(KEY);
  } catch {}
}

export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(status: number, message: string, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const k = getKey();
  return k ? { Authorization: `Bearer ${k}`, ...extra } : extra;
}

async function parseError(r: Response): Promise<ApiError> {
  let msg = r.statusText;
  let code: string | undefined;
  try {
    const j = await r.json();
    const d = j.detail ?? j.error ?? j;
    msg = typeof d === "string" ? d : d.message || JSON.stringify(d);
    code = typeof d === "object" ? d.code : undefined;
  } catch {}
  return new ApiError(r.status, msg, code);
}

export async function api<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: headers({ ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...(init.headers as Record<string, string>) }),
  });
  if (!r.ok) throw await parseError(r);
  return (await r.json()) as T;
}

export type Health = {
  status: string;
  models: string[];
  voices: string[];
  queue_depth: number;
  gpu?: { name?: string; vram_used_gb?: number };
};
export type Me = {
  account_id: string;
  name: string;
  email?: string;
  plan: string;
  admin: boolean;
  limits?: { chars_per_month: number; concurrency: number; rpm: number; voices: number };
  month_chars?: number;
};
export type Voice = {
  voice_id: string;
  name: string;
  description: string;
  language: string;
  exaggeration: number;
  builtin: boolean;
  ready_for: string[];
  created_at: number;
};
export type Usage = {
  days: number;
  requests: number;
  errors: number;
  chars: number;
  audio_seconds: number;
  ttfa_ms: { avg: number | null; p50: number | null; p95: number | null };
  daily: { day: string; requests: number; chars: number; audio_s: number; ttfa_avg: number | null }[];
  by_model: { model: string; requests: number; audio_s: number }[];
  month_chars: number | null;
};
export type UsageRow = {
  ts: number;
  endpoint: string;
  model: string;
  voice: string;
  language: string;
  format: string;
  chars: number;
  audio_s: number;
  ttfa_ms: number | null;
  total_ms: number | null;
  status: number;
};
export type ApiKey = { id: string; name: string; prefix: string; created_at: number; last_used_at: number | null; revoked_at: number | null; key?: string };

export type SpeechParams = {
  input: string;
  voice: string;
  language?: string;
  response_format?: string;
  speed?: number;
  exaggeration?: number;
  temperature?: number;
  cfg_weight?: number;
  model?: string;
};

/** Stream speech; resolves TTFB as soon as the first byte lands and returns the
 *  full PCM buffer at the end. Callers can play `onChunk` blocks live. */
export async function speak(
  params: SpeechParams,
  onChunk?: (chunk: Uint8Array, ttfbMs: number | null) => void,
  signal?: AbortSignal
): Promise<{ bytes: Uint8Array; ttfbMs: number; totalMs: number; sampleRate: number }> {
  const t0 = performance.now();
  const r = await fetch(`${API_BASE}/v1/audio/speech`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ model: "chatterbox", response_format: "pcm", ...params }),
    signal,
  });
  if (!r.ok || !r.body) throw await parseError(r);
  const sampleRate = Number(r.headers.get("x-sample-rate") || 24000);
  const reader = r.body.getReader();
  const parts: Uint8Array[] = [];
  let ttfb: number | null = null;
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value && value.length) {
      if (ttfb === null) ttfb = performance.now() - t0;
      parts.push(value);
      total += value.length;
      onChunk?.(value, ttfb);
    }
  }
  const bytes = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    bytes.set(p, o);
    o += p.length;
  }
  return { bytes, ttfbMs: ttfb ?? performance.now() - t0, totalMs: performance.now() - t0, sampleRate };
}

export function fmtInt(n: number | null | undefined) {
  return n == null ? "–" : Math.round(n).toLocaleString();
}
export function fmtMs(n: number | null | undefined) {
  return n == null ? "–" : `${Math.round(n)} ms`;
}
export function fmtDur(s: number | null | undefined) {
  if (s == null) return "–";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = s / 60;
  return m < 60 ? `${m.toFixed(1)} min` : `${(m / 60).toFixed(1)} h`;
}
export function fmtDate(ts: number | null | undefined) {
  return ts ? new Date(ts * 1000).toLocaleString() : "–";
}
