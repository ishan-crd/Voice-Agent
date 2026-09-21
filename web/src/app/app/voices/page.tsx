"use client";

import { useEffect, useRef, useState } from "react";
import { Voice, api, fmtDate, speak } from "@/lib/api";
import { PcmPlayer } from "@/lib/audio";
import { Empty, Field, Toast } from "@/components/ui";
import { useSession } from "../layout";

export default function Voices() {
  const { me, refresh } = useSession();
  const [voices, setVoices] = useState<Voice[]>([]);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [lang, setLang] = useState("en");
  const [mode, setMode] = useState<"quick" | "pro">("quick");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<{ msg: string; tone?: "ok" | "err" } | null>(null);
  const [previewing, setPreviewing] = useState<string | null>(null);
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const [recSecs, setRecSecs] = useState(0);
  const chunks = useRef<Blob[]>([]);
  const player = useRef<PcmPlayer | null>(null);

  const load = () => api<{ voices: Voice[] }>("/v1/voices").then((r) => setVoices(r.voices)).catch(() => {});
  useEffect(() => {
    load();
    return () => player.current?.close();
  }, []);

  async function upload() {
    if (!file || !name.trim()) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("file", file, file.name || "clip.webm");
      fd.append("description", desc);
      fd.append("language", lang);
      fd.append("mode", mode);
      const v = await api<Voice>("/v1/voices", { method: "POST", body: fd });
      setToast({ msg: `Voice "${v.voice_id}" is ready for ${v.ready_for.join(" + ")}` });
      setName("");
      setDesc("");
      setFile(null);
      load();
      refresh();
    } catch (e) {
      setToast({ msg: e instanceof Error ? e.message : "upload failed", tone: "err" });
    } finally {
      setBusy(false);
    }
  }

  async function preview(v: Voice) {
    setPreviewing(v.voice_id);
    player.current?.close();
    player.current = new PcmPlayer(24000);
    await player.current.resume();
    const text = v.language === "hi" ? "नमस्ते! यह मेरी क्लोन की हुई आवाज़ है।" : "Hi there! This is what my cloned voice sounds like, streaming straight from the server.";
    try {
      await speak({ input: text, voice: v.voice_id }, (c) => player.current?.push(c));
    } catch (e) {
      setToast({ msg: e instanceof Error ? e.message : "preview failed", tone: "err" });
    } finally {
      setPreviewing(null);
    }
  }

  async function remove(v: Voice) {
    if (!confirm(`Delete voice "${v.voice_id}"? Requests using it will start failing.`)) return;
    try {
      await api(`/v1/voices/${v.voice_id}`, { method: "DELETE" });
      load();
      refresh();
    } catch (e) {
      setToast({ msg: e instanceof Error ? e.message : "delete failed", tone: "err" });
    }
  }

  async function startRec() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const mr = new MediaRecorder(stream);
      chunks.current = [];
      mr.ondataavailable = (e) => chunks.current.push(e.data);
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: mr.mimeType || "audio/webm" });
        setFile(new File([blob], "recording.webm", { type: blob.type }));
      };
      mr.start();
      setRec(mr);
      setRecSecs(0);
      const t0 = Date.now();
      const iv = setInterval(() => {
        setRecSecs((Date.now() - t0) / 1000);
        if (mr.state !== "recording") clearInterval(iv);
      }, 200);
    } catch {
      setToast({ msg: "microphone access denied", tone: "err" });
    }
  }
  function stopRec() {
    rec?.stop();
    setRec(null);
  }

  const mine = voices.filter((v) => !v.builtin);
  const limit = me?.limits?.voices ?? 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Voices</h1>
        <p className="text-sm text-fg-2">
          One clip per emotional register. 6–15 s, one speaker, no music. {limit ? `${mine.length} / ${limit} used on the ${me?.plan} plan.` : ""}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="card space-y-4 p-5">
          <div className="font-medium">Clone a new voice</div>
          <Field label="Voice ID" hint="letters, digits, - _">
            <input className="input mono" placeholder="ishan_calm" value={name} onChange={(e) => setName(e.target.value.replace(/[^\w-]/g, "_").toLowerCase())} />
          </Field>
          <Field label="Description">
            <input className="input" placeholder="Neutral support tone" value={desc} onChange={(e) => setDesc(e.target.value)} />
          </Field>
          <Field label="Primary language" hint="used when auto-detect is ambiguous">
            <select className="select" value={lang} onChange={(e) => setLang(e.target.value)}>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="ar">Arabic</option>
              <option value="pt">Portuguese</option>
            </select>
          </Field>
          <Field label="Clone mode">
            <div className="grid grid-cols-2 gap-2">
              {(["quick", "pro"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setMode(m)} className={`rounded-lg border px-3 py-2 text-left text-xs ${mode === m ? "border-accent bg-card" : "border-border text-fg-2"}`}>
                  <div className="font-semibold">{m === "quick" ? "Quick" : "Pro"}</div>
                  <div className="mt-0.5 text-[11px] leading-snug text-fg-3">{m === "quick" ? "One 6–15 s clip. Instant." : "1–10 min recording. Averages your identity across it and picks the cleanest stretch as the prompt."}</div>
                </button>
              ))}
            </div>
          </Field>
          <Field label={mode === "pro" ? "Long recording" : "Reference clip"} hint={mode === "pro" ? "wav / mp3 / m4a, up to 200 MB" : "6–15 s"}>
            <div className="flex flex-col gap-2">
              <input type="file" accept="audio/*" className="text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <div className="flex items-center gap-2 text-xs text-fg-3">
                <span>or</span>
                {!rec ? (
                  <button className="btn btn-ghost !py-1 !px-2 text-xs" onClick={startRec} type="button">
                    ● Record in browser
                  </button>
                ) : (
                  <button className="btn btn-danger !py-1 !px-2 text-xs" onClick={stopRec} type="button">
                    ■ Stop ({recSecs.toFixed(0)} s)
                  </button>
                )}
              </div>
              {file && (
                <div className="text-xs text-fg-2">
                  {file.name} · {(file.size / 1024).toFixed(0)} KB
                </div>
              )}
            </div>
          </Field>
          <button className="btn btn-primary w-full justify-center" disabled={busy || !file || !name.trim()} onClick={upload}>
            {busy ? (mode === "pro" ? "Analysing recording…" : "Preparing voice…") : "Clone voice"}
          </button>
          {mode === "pro" && (
            <p className="rounded-lg border border-dashed p-2 text-[11px] leading-relaxed text-fg-3">
              Read naturally for a few minutes in the mood you want — a story, an email, a support call. Vary your sentences; avoid music, other speakers and long silences. The model still conditions on ~10 s of audio, so the gain is a steadier, more typical identity rather than a different engine.
            </p>
          )}
          <p className="text-[11px] leading-relaxed text-fg-3">By uploading you confirm you own this voice or have the speaker&apos;s permission. Generated audio carries an inaudible watermark.</p>
        </div>

        <div className="space-y-3">
          {voices.length === 0 && <Empty title="No voices yet" body="Clone one on the left, or use the built-in voice from the playground." />}
          {voices.map((v) => (
            <div key={v.voice_id} className="card flex flex-wrap items-center gap-4 p-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full text-sm font-semibold" style={{ background: "linear-gradient(135deg,var(--accent),var(--accent-2))" }}>
                {v.voice_id.slice(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="mono font-medium">{v.voice_id}</span>
                  {v.builtin && <span className="pill">built-in</span>}
                  {!v.builtin && v.mode === "pro" && <span className="pill border-accent/50 text-accent">pro</span>}
                  <span className="pill uppercase">{v.language}</span>
                </div>
                <div className="truncate text-xs text-fg-2">{v.description || (v.builtin ? "Chatterbox reference voice" : "—")}</div>
                <div className="text-[11px] text-fg-3">
                  ready for {v.ready_for.join(", ")}{!v.builtin && ` · added ${fmtDate(v.created_at)}`}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button className="btn btn-ghost" onClick={() => preview(v)} disabled={previewing !== null}>
                  {previewing === v.voice_id ? "Playing…" : "▶ Preview"}
                </button>
                {!v.builtin && (
                  <button className="btn btn-danger" onClick={() => remove(v)}>
                    Delete
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      {toast && <Toast msg={toast.msg} tone={toast.tone} onDone={() => setToast(null)} />}
    </div>
  );
}
