"use client";

import { useEffect, useState } from "react";
import { Usage, UsageRow, api, fmtDate, fmtDur, fmtInt, fmtMs } from "@/lib/api";
import { BarChart, Empty, Stat } from "@/components/ui";
import { useSession } from "../layout";

export default function UsagePage() {
  const { me } = useSession();
  const [days, setDays] = useState(30);
  const [metric, setMetric] = useState<"requests" | "audio_s" | "chars">("audio_s");
  const [u, setU] = useState<Usage | null>(null);
  const [rows, setRows] = useState<UsageRow[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    setErr("");
    api<Usage>(`/v1/usage?days=${days}`).then(setU).catch((e) => setErr(e.message));
    api<{ rows: UsageRow[] }>("/v1/usage/recent?limit=50").then((r) => setRows(r.rows)).catch(() => {});
  }, [days]);

  if (err) return <Empty title="Usage is unavailable" body={err} />;
  if (!u) return <div className="text-fg-3">Loading…</div>;

  const limit = me?.limits?.chars_per_month ?? 0;
  const used = u.month_chars ?? 0;
  const pct = limit ? Math.min(100, (used / limit) * 100) : 0;
  const series = u.daily.map((d) => ({ x: d.day, y: metric === "audio_s" ? d.audio_s / 60 : d[metric] }));
  const label = metric === "audio_s" ? "minutes" : metric;
  const fmt = (y: number) => (metric === "audio_s" ? y.toFixed(1) : Math.round(y).toLocaleString());

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Usage</h1>
          <p className="text-sm text-fg-2">Metered per request: characters in, seconds of audio out, time to first audio.</p>
        </div>
        <div className="flex gap-1">
          {[7, 30, 90].map((d) => (
            <button key={d} className={`btn !py-1 !px-3 text-xs ${days === d ? "btn-primary" : "btn-ghost"}`} onClick={() => setDays(d)}>
              {d}d
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Stat label="Requests" value={fmtInt(u.requests)} sub={u.errors ? `${u.errors} errors` : "no errors"} tone={u.errors ? "warn" : undefined} />
        <Stat label="Audio generated" value={fmtDur(u.audio_seconds)} sub={`${fmtInt(u.chars)} characters`} />
        <Stat label="First audio p50" value={fmtMs(u.ttfa_ms.p50)} sub={`p95 ${fmtMs(u.ttfa_ms.p95)}`} tone={u.ttfa_ms.p50 == null ? undefined : u.ttfa_ms.p50 < 300 ? "good" : "warn"} />
        <div className="card p-4">
          <div className="text-xs uppercase tracking-wider text-fg-3">This month</div>
          <div className="mt-1 text-2xl font-semibold tabular-nums">{fmtInt(used)}</div>
          <div className="mt-1 text-xs text-fg-2">{limit ? `of ${fmtInt(limit)} chars on ${me?.plan}` : "unlimited"}</div>
          {limit > 0 && (
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-bg-elev">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, background: pct > 90 ? "var(--bad)" : pct > 70 ? "var(--warn)" : "var(--accent)" }} />
            </div>
          )}
        </div>
      </div>

      <div className="card p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="font-medium">Daily {label}</div>
          <div className="flex gap-1">
            {(["audio_s", "requests", "chars"] as const).map((m) => (
              <button key={m} className={`btn !py-1 !px-3 text-xs ${metric === m ? "btn-primary" : "btn-ghost"}`} onClick={() => setMetric(m)}>
                {m === "audio_s" ? "minutes" : m}
              </button>
            ))}
          </div>
        </div>
        {series.length ? <BarChart data={series} yLabel={label} format={fmt} /> : <div className="py-10 text-center text-sm text-fg-3">Nothing in this window yet.</div>}
        {u.by_model.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-fg-2">
            {u.by_model.map((m) => (
              <span key={m.model} className="pill">
                {m.model}: {fmtInt(m.requests)} req · {fmtDur(m.audio_s)}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="card overflow-hidden">
        <div className="border-b px-5 py-3 text-sm font-medium">Recent requests</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-bg-elev text-left text-xs uppercase tracking-wider text-fg-3">
              <tr>
                <th className="px-5 py-2">When</th>
                <th className="px-3 py-2">Endpoint</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Voice</th>
                <th className="px-3 py-2">Lang</th>
                <th className="px-3 py-2">Fmt</th>
                <th className="px-3 py-2 text-right">Chars</th>
                <th className="px-3 py-2 text-right">Audio</th>
                <th className="px-3 py-2 text-right">First audio</th>
                <th className="px-3 py-2 text-right">Total</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t text-xs">
                  <td className="px-5 py-2 text-fg-2">{fmtDate(r.ts)}</td>
                  <td className="px-3 py-2 mono">{r.endpoint}</td>
                  <td className="px-3 py-2">{r.model}</td>
                  <td className="px-3 py-2 mono">{r.voice}</td>
                  <td className="px-3 py-2 uppercase">{r.language}</td>
                  <td className="px-3 py-2">{r.format}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtInt(r.chars)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.audio_s.toFixed(1)} s</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtMs(r.ttfa_ms)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtMs(r.total_ms)}</td>
                  <td className="px-3 py-2">
                    <span className={`pill ${r.status < 400 ? "text-good" : r.status === 499 ? "text-warn" : "text-bad"}`}>{r.status === 499 ? "cancelled" : r.status}</span>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={11} className="px-5 py-6 text-center text-fg-3">
                    No requests yet — try the playground.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
