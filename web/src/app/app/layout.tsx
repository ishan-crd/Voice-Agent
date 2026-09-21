"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ApiError, Health, Me, api, getKey, setKey } from "@/lib/api";
import { Logo } from "@/components/ui";

type Session = { me: Me | null; health: Health | null; refresh: () => void; signOut: () => void };
const Ctx = createContext<Session>({ me: null, health: null, refresh: () => {}, signOut: () => {} });
export const useSession = () => useContext(Ctx);

const nav = [
  { href: "/app/", label: "Playground", icon: "▶" },
  { href: "/app/voices/", label: "Voices", icon: "◉" },
  { href: "/app/keys/", label: "API keys", icon: "⚿" },
  { href: "/app/usage/", label: "Usage", icon: "▤" },
  { href: "/app/docs/", label: "Docs", icon: "❯" },
];

export default function ConsoleLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [me, setMe] = useState<Me | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [state, setState] = useState<"loading" | "anon" | "ok">("loading");
  const [err, setErr] = useState("");
  const [draft, setDraft] = useState("");

  const load = useCallback(async () => {
    api<Health>("/health").then(setHealth).catch(() => {});
    try {
      const m = await api<Me>("/v1/me");
      setMe(m);
      setState("ok");
    } catch (e) {
      setMe(null);
      setState("anon");
      if (e instanceof ApiError && e.status !== 401) setErr(e.message);
    }
  }, []);

  useEffect(() => {
    // onboarding links: /app/?key=va_... signs the browser in and strips the param
    try {
      const u = new URL(window.location.href);
      const k = u.searchParams.get("key");
      if (k) {
        setKey(k);
        u.searchParams.delete("key");
        window.history.replaceState({}, "", u.toString());
      }
    } catch {}
    load();
  }, [load]);

  const signOut = () => {
    setKey("");
    setMe(null);
    setState("anon");
  };

  if (state === "loading") {
    return <div className="flex flex-1 items-center justify-center text-fg-3">Connecting…</div>;
  }

  if (state === "anon") {
    return (
      <div className="relative flex flex-1 items-center justify-center px-6">
        <div className="grid-bg pointer-events-none absolute inset-0" />
        <form
          className="card glow relative w-full max-w-md p-8"
          onSubmit={async (e) => {
            e.preventDefault();
            setErr("");
            setKey(draft.trim());
            try {
              await api<Me>("/v1/me");
              await load();
            } catch (e2) {
              setKey("");
              setErr(e2 instanceof Error ? e2.message : "sign-in failed");
            }
          }}
        >
          <Logo size={26} />
          <h1 className="mt-6 text-xl font-semibold">Sign in with your API key</h1>
          <p className="mt-1 text-sm text-fg-2">Paste the <span className="mono">va_…</span> key you were given. It stays in this browser only.</p>
          <input className="input mono mt-5" placeholder="va_…" value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus />
          {err && <div className="mt-3 text-sm text-bad">{err}</div>}
          <button className="btn btn-primary mt-4 w-full justify-center" type="submit" disabled={!draft.trim()}>
            Continue
          </button>
          <p className="mt-4 text-xs text-fg-3">
            No key yet? Ask for one at <a className="underline" href="mailto:hello@voiceagent.dev">hello@voiceagent.dev</a>, or if you run this server, create one with <span className="mono">scripts/admin.py create-account</span>.
          </p>
          {health && (
            <p className="mt-3 flex items-center gap-2 text-xs text-fg-3">
              <span className="live-dot" /> {health.gpu?.name ?? "server"} online · {health.models.join(", ")}
            </p>
          )}
        </form>
      </div>
    );
  }

  return (
    <Ctx.Provider value={{ me, health, refresh: load, signOut }}>
      <div className="flex flex-1">
        <aside className="hidden w-60 shrink-0 flex-col border-r bg-bg-elev md:flex">
          <div className="px-5 py-5">
            <Link href="/">
              <Logo size={22} />
            </Link>
          </div>
          <nav className="flex flex-col gap-1 px-3">
            {nav.map((n) => {
              const active = path === n.href || (n.href !== "/app/" && path?.startsWith(n.href));
              return (
                <Link key={n.href} href={n.href} className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${active ? "bg-card font-medium text-fg" : "text-fg-2 hover:bg-card hover:text-fg"}`}>
                  <span className="w-4 text-center text-fg-3">{n.icon}</span>
                  {n.label}
                </Link>
              );
            })}
            {me?.admin && (
              <Link href="/app/admin/" className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${path?.startsWith("/app/admin") ? "bg-card font-medium" : "text-fg-2 hover:bg-card hover:text-fg"}`}>
                <span className="w-4 text-center text-fg-3">★</span>
                Admin
              </Link>
            )}
          </nav>
          <div className="mt-auto space-y-3 border-t p-4 text-xs">
            {health && (
              <div className="flex items-center gap-2 text-fg-2">
                <span className="live-dot" />
                <span className="truncate">{health.gpu?.name?.replace("NVIDIA ", "") ?? "server"}</span>
                <span className="ml-auto text-fg-3">q{health.queue_depth}</span>
              </div>
            )}
            <div>
              <div className="truncate font-medium">{me?.name}</div>
              <div className="flex items-center justify-between text-fg-3">
                <span className="capitalize">{me?.plan} plan</span>
                <button className="underline" onClick={signOut}>
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </aside>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-2 border-b px-4 py-3 md:hidden">
            <Logo size={20} />
            <div className="ml-auto flex gap-1 overflow-x-auto text-xs">
              {nav.map((n) => (
                <Link key={n.href} href={n.href} className={`rounded-md px-2 py-1 ${path === n.href ? "bg-card" : "text-fg-2"}`}>
                  {n.label}
                </Link>
              ))}
            </div>
          </div>
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 md:px-8 md:py-8">{children}</main>
        </div>
      </div>
    </Ctx.Provider>
  );
}
