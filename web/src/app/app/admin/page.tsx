"use client";

import { useEffect, useState } from "react";
import { ApiKey, Usage, api, fmtDate, fmtDur, fmtInt, fmtMs } from "@/lib/api";
import { Copy, Empty, Stat, Toast } from "@/components/ui";
import { useSession } from "../layout";

type Account = {
  id: string;
  name: string;
  email: string | null;
  plan: string;
  created_at: number;
  char_limit: number;
  max_concurrency: number;
  rpm: number;
  max_voices: number;
  month_chars: number;
  keys: ApiKey[];
  voices: string[];
};

export default function Admin() {
  const { me } = useSession();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [plans, setPlans] = useState<Record<string, Record<string, number>>>({});
  const [usage, setUsage] = useState<Usage | null>(null);
  const [form, setForm] = useState({ name: "", email: "", plan: "free" });
  const [fresh, setFresh] = useState<{ account: string; key: string } | null>(null);
  const [toast, setToast] = useState<{ msg: string; tone?: "ok" | "err" } | null>(null);

  const load = () => {
    api<{ accounts: Account[]; plans: Record<string, Record<string, number>> }>("/admin/accounts")
      .then((r) => {
        setAccounts(r.accounts);
        setPlans(r.plans);
      })
      .catch((e) => setToast({ msg: e.message, tone: "err" }));
    api<Usage>("/admin/usage?days=30").then(setUsage).catch(() => {});
  };
  useEffect(() => {
    if (me?.admin) load();
  }, [me]);

  if (!me?.admin) return <Empty title="Admin key required" body="Sign in with TTS_ADMIN_KEY to manage accounts." />;

  async function create() {
    try {
      const r = await api<{ account: Account; key: ApiKey }>("/admin/accounts", { method: "POST", body: JSON.stringify({ name: form.name, email: form.email || null, plan: form.plan }) });
      setFresh({ account: r.account.name, key: r.key.key || "" });
      setForm({ name: "", email: "", plan: "free" });
      load();
    } catch (e) {
      setToast({ msg: e instanceof Error ? e.message : "failed", tone: "err" });
    }
  }
  async function setPlan(a: Account, plan: string) {
    await api(`/admin/accounts/${a.id}/plan`, { method: "PUT", body: JSON.stringify({ plan }) });
    load();
  }
  async function newKey(a: Account) {
    const r = await api<ApiKey>(`/admin/accounts/${a.id}/keys`, { method: "POST", body: JSON.stringify({ name: "issued-" + new Date().toISOString().slice(0, 10) }) });
    setFresh({ account: a.name, key: r.key || "" });
    load();
  }
  async function revoke(k: ApiKey) {
    await api(`/v1/keys/${k.id}`, { method: "DELETE" });
    load();
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="text-sm text-fg-2">All accounts on this server. Plans: {Object.keys(plans).join(", ")}.</p>
      </div>

      {usage && (
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Requests · 30d" value={fmtInt(usage.requests)} sub={`${usage.errors} errors`} />
          <Stat label="Audio · 30d" value={fmtDur(usage.audio_seconds)} sub={`${fmtInt(usage.chars)} chars`} />
          <Stat label="First audio p50 / p95" value={fmtMs(usage.ttfa_ms.p50)} sub={fmtMs(usage.ttfa_ms.p95)} />
          <Stat label="Accounts" value={accounts.length} sub={`${accounts.reduce((n, a) => n + a.keys.filter((k) => !k.revoked_at).length, 0)} active keys`} />
        </div>
      )}

      {fresh && (
        <div className="card glow border-accent/50 p-5">
          <div className="font-medium">Key for {fresh.account} — copy now, shown once</div>
          <div className="mt-3 flex items-center gap-2">
            <code className="mono flex-1 truncate rounded-lg bg-bg-elev px-3 py-2 text-sm">{fresh.key}</code>
            <Copy text={fresh.key} />
            <button className="btn btn-ghost" onClick={() => setFresh(null)}>
              Done
            </button>
          </div>
        </div>
      )}

      <div className="card p-5">
        <div className="mb-3 font-medium">New account</div>
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_160px_auto]">
          <input className="input" placeholder="Company / person" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <input className="input" placeholder="email (optional)" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          <select className="select" value={form.plan} onChange={(e) => setForm({ ...form, plan: e.target.value })}>
            {Object.keys(plans).map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={create} disabled={!form.name.trim()}>
            Create + issue key
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {accounts.map((a) => (
          <div key={a.id} className="card p-5">
            <div className="flex flex-wrap items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{a.name}</span>
                  <span className="mono text-xs text-fg-3">{a.id}</span>
                </div>
                <div className="text-xs text-fg-2">
                  {a.email || "no email"} · since {fmtDate(a.created_at)} · voices: {a.voices.join(", ") || "none"}
                </div>
              </div>
              <div className="text-right text-xs">
                <div className="tabular-nums">
                  {fmtInt(a.month_chars)} / {a.char_limit ? fmtInt(a.char_limit) : "∞"} chars
                </div>
                <div className="text-fg-3">
                  {a.max_concurrency} streams · {a.rpm || "∞"} rpm · {a.max_voices || "∞"} voices
                </div>
              </div>
              <select className="select !w-32" value={a.plan} onChange={(e) => setPlan(a, e.target.value)}>
                {Object.keys(plans).map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>
              <button className="btn btn-ghost" onClick={() => newKey(a)}>
                + key
              </button>
            </div>
            {a.keys.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {a.keys.map((k) => (
                  <span key={k.id} className={`pill ${k.revoked_at ? "line-through opacity-50" : ""}`}>
                    <span className="mono">{k.prefix}…</span> {k.name}
                    {!k.revoked_at && (
                      <button className="ml-1 text-bad" title="revoke" onClick={() => revoke(k)}>
                        ×
                      </button>
                    )}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      {toast && <Toast msg={toast.msg} tone={toast.tone} onDone={() => setToast(null)} />}
    </div>
  );
}
