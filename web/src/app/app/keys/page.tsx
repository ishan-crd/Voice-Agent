"use client";

import { useEffect, useState } from "react";
import { ApiKey, api, fmtDate } from "@/lib/api";
import { Copy, Empty, Toast } from "@/components/ui";
import { useSession } from "../layout";

export default function Keys() {
  const { me } = useSession();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [name, setName] = useState("");
  const [fresh, setFresh] = useState<ApiKey | null>(null);
  const [toast, setToast] = useState<{ msg: string; tone?: "ok" | "err" } | null>(null);

  const load = () => api<{ keys: ApiKey[] }>("/v1/keys").then((r) => setKeys(r.keys)).catch(() => {});
  useEffect(() => {
    if (!me?.admin) load();
  }, [me]);

  if (me?.admin) {
    return <Empty title="You are signed in with the admin key" body="Issue customer keys from the Admin page; admin keys are not listed here." />;
  }

  async function create() {
    try {
      const k = await api<ApiKey>("/v1/keys", { method: "POST", body: JSON.stringify({ name: name.trim() || "default" }) });
      setFresh(k);
      setName("");
      load();
    } catch (e) {
      setToast({ msg: e instanceof Error ? e.message : "failed", tone: "err" });
    }
  }
  async function revoke(k: ApiKey) {
    if (!confirm(`Revoke "${k.name}" (${k.prefix}…)? Anything using it stops working immediately.`)) return;
    await api(`/v1/keys/${k.id}`, { method: "DELETE" });
    load();
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">API keys</h1>
        <p className="text-sm text-fg-2">Send as <span className="mono">Authorization: Bearer va_…</span> (or <span className="mono">xi-api-key</span> for ElevenLabs-shaped clients).</p>
      </div>

      {fresh && (
        <div className="card glow border-accent/50 p-5">
          <div className="font-medium">New key created — copy it now, it won&apos;t be shown again</div>
          <div className="mt-3 flex items-center gap-2">
            <code className="mono flex-1 truncate rounded-lg bg-bg-elev px-3 py-2 text-sm">{fresh.key}</code>
            <Copy text={fresh.key || ""} />
            <button className="btn btn-ghost" onClick={() => setFresh(null)}>
              Done
            </button>
          </div>
        </div>
      )}

      <div className="card p-5">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex-1">
            <div className="mb-1 text-xs font-medium text-fg-2">Key name</div>
            <input className="input" placeholder="production" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <button className="btn btn-primary" onClick={create}>
            + Create key
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-bg-elev text-left text-xs uppercase tracking-wider text-fg-3">
            <tr>
              <th className="px-5 py-2">Name</th>
              <th className="px-3 py-2">Key</th>
              <th className="px-3 py-2">Created</th>
              <th className="px-3 py-2">Last used</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id} className="border-t">
                <td className="px-5 py-3 font-medium">{k.name}</td>
                <td className="px-3 py-3 mono text-xs">{k.prefix}…</td>
                <td className="px-3 py-3 text-xs text-fg-2">{fmtDate(k.created_at)}</td>
                <td className="px-3 py-3 text-xs text-fg-2">{k.last_used_at ? fmtDate(k.last_used_at) : "never"}</td>
                <td className="px-3 py-3">{k.revoked_at ? <span className="pill text-bad">revoked</span> : <span className="pill text-good">active</span>}</td>
                <td className="px-3 py-3 text-right">
                  {!k.revoked_at && (
                    <button className="btn btn-danger !py-1 !px-2 text-xs" onClick={() => revoke(k)}>
                      Revoke
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {keys.length === 0 && (
              <tr>
                <td colSpan={6} className="px-5 py-6 text-center text-sm text-fg-3">
                  No keys yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {toast && <Toast msg={toast.msg} tone={toast.tone} onDone={() => setToast(null)} />}
    </div>
  );
}
