"use client";

import { useEffect, useState } from "react";

export function Logo({ size = 22 }: { size?: number }) {
  return (
    <span className="inline-flex items-center gap-2 font-semibold tracking-tight">
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#c4b5ff" />
            <stop offset="0.5" stopColor="#7c5cff" />
            <stop offset="1" stopColor="#38d6c4" />
          </linearGradient>
        </defs>
        <rect x="3" y="9" width="3" height="6" rx="1.5" fill="url(#lg)" />
        <rect x="8" y="5" width="3" height="14" rx="1.5" fill="url(#lg)" />
        <rect x="13" y="2" width="3" height="20" rx="1.5" fill="url(#lg)" />
        <rect x="18" y="7" width="3" height="10" rx="1.5" fill="url(#lg)" />
      </svg>
      <span>
        Voice<span className="gradient-text">Agent</span>
      </span>
    </span>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "good" | "warn" | "bad" }) {
  const color = tone === "good" ? "text-good" : tone === "warn" ? "text-warn" : tone === "bad" ? "text-bad" : "";
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wider text-fg-3">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-fg-2">{sub}</div>}
    </div>
  );
}

export function Copy({ text, label = "Copy" }: { text: string; label?: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      className="btn btn-ghost !py-1 !px-2 text-xs"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setOk(true);
          setTimeout(() => setOk(false), 1200);
        } catch {}
      }}
    >
      {ok ? "Copied" : label}
    </button>
  );
}

export function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs font-medium text-fg-2">{label}</span>
        {hint && <span className="text-[11px] text-fg-3">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

export function Toast({ msg, tone = "ok", onDone }: { msg: string; tone?: "ok" | "err"; onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 3500);
    return () => clearTimeout(t);
  }, [onDone]);
  return (
    <div className={`fixed bottom-5 right-5 z-50 rounded-xl border px-4 py-3 text-sm shadow-xl ${tone === "err" ? "border-bad/40 bg-card text-bad" : "border-good/40 bg-card text-good"}`}>
      {msg}
    </div>
  );
}

export function Empty({ title, body }: { title: string; body?: string }) {
  return (
    <div className="card flex flex-col items-center justify-center p-10 text-center">
      <div className="text-base font-medium">{title}</div>
      {body && <div className="mt-1 max-w-sm text-sm text-fg-2">{body}</div>}
    </div>
  );
}

/** Single-series bar chart: recessive grid, thin marks, hover tooltip. */
export function BarChart({ data, yLabel, format }: { data: { x: string; y: number }[]; yLabel: string; format?: (y: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 640;
  const H = 180;
  const padL = 40;
  const padB = 22;
  const padT = 10;
  const max = Math.max(1, ...data.map((d) => d.y));
  const n = Math.max(1, data.length);
  const slot = (W - padL) / n;
  const bw = Math.max(2, Math.min(28, slot - 2));
  const fmt = format ?? ((y: number) => y.toLocaleString());
  const ticks = [0, 0.5, 1].map((t) => t * max);
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={yLabel}>
        {ticks.map((t, i) => {
          const y = padT + (H - padT - padB) * (1 - t / max);
          return (
            <g key={i}>
              <line x1={padL} x2={W} y1={y} y2={y} stroke="var(--border)" strokeWidth="1" />
              <text x={padL - 6} y={y + 4} textAnchor="end" fontSize="10" fill="var(--text-3)">
                {fmt(t)}
              </text>
            </g>
          );
        })}
        {data.map((d, i) => {
          const h = (H - padT - padB) * (d.y / max);
          const x = padL + i * slot + (slot - bw) / 2;
          const y = H - padB - h;
          return (
            <g key={d.x} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={padL + i * slot} y={padT} width={slot} height={H - padT - padB} fill="transparent" />
              <rect x={x} y={y} width={bw} height={Math.max(0, h)} rx="3" fill="var(--series-1)" opacity={hover === null || hover === i ? 1 : 0.55} />
              {(i === 0 || i === n - 1 || (n <= 12 && i % 2 === 0)) && (
                <text x={padL + i * slot + slot / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-3)">
                  {d.x.slice(5)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover !== null && data[hover] && (
        <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 rounded-lg border bg-bg-elev px-3 py-2 text-xs shadow-lg">
          <div className="text-fg-3">{data[hover].x}</div>
          <div className="font-semibold tabular-nums">
            {fmt(data[hover].y)} <span className="font-normal text-fg-2">{yLabel}</span>
          </div>
        </div>
      )}
    </div>
  );
}
