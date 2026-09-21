"use client";

/** Plays 16-bit PCM chunks as they arrive, gapless, via Web Audio. */
export class PcmPlayer {
  private ctx: AudioContext;
  private nextTime = 0;
  private sampleRate: number;
  private sources: AudioBufferSourceNode[] = [];
  private carry = new Uint8Array(0);
  analyser: AnalyserNode;

  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate;
    this.ctx = new AudioContext({ sampleRate });
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.connect(this.ctx.destination);
  }

  async resume() {
    if (this.ctx.state !== "running") await this.ctx.resume();
  }

  push(chunk: Uint8Array) {
    // keep an odd trailing byte for the next chunk
    let data = chunk;
    if (this.carry.length) {
      data = new Uint8Array(this.carry.length + chunk.length);
      data.set(this.carry);
      data.set(chunk, this.carry.length);
    }
    const even = data.length - (data.length % 2);
    this.carry = data.slice(even);
    if (even === 0) return;
    const i16 = new Int16Array(data.buffer.slice(data.byteOffset, data.byteOffset + even));
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = this.ctx.createBuffer(1, f32.length, this.sampleRate);
    buf.copyToChannel(f32, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser);
    const now = this.ctx.currentTime;
    // small initial cushion so scheduling jitter never causes a gap
    if (this.nextTime < now + 0.015) this.nextTime = now + 0.035;
    src.start(this.nextTime);
    this.nextTime += buf.duration;
    this.sources.push(src);
  }

  /** seconds of audio queued but not yet played */
  buffered(): number {
    return Math.max(0, this.nextTime - this.ctx.currentTime);
  }

  stop() {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {}
    }
    this.sources = [];
    this.nextTime = 0;
  }

  close() {
    this.stop();
    this.ctx.close().catch(() => {});
  }
}

/** PCM16 -> WAV blob for download / <audio>. */
export function pcmToWav(pcm: Uint8Array, sampleRate = 24000): Blob {
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + pcm.length, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, pcm.length, true);
  return new Blob([header, pcm as BlobPart], { type: "audio/wav" });
}

/** Captures the microphone as 16 kHz int16 frames while `capturing` is true.
 *  The stream stays open between turns so a press never waits on getUserMedia. */
export class MicCapture {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private capturing = false;
  onFrame: ((frame: ArrayBuffer) => void) | null = null;

  async open(): Promise<void> {
    if (this.ctx) return;
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    const ctx = new AudioContext({ sampleRate: 16000 });
    const worklet = `
      class Cap extends AudioWorkletProcessor {
        constructor() { super(); this.on = false; this.buf = []; this.n = 0;
          this.port.onmessage = (e) => { this.on = e.data; if (!this.on) this.flush(); }; }
        flush() { if (!this.n) return; const out = new Int16Array(this.n); let o = 0;
          for (const c of this.buf) { for (let i = 0; i < c.length; i++) { const v = Math.max(-1, Math.min(1, c[i])); out[o++] = v < 0 ? v * 32768 : v * 32767; } }
          this.port.postMessage(out.buffer, [out.buffer]); this.buf = []; this.n = 0; }
        process(inputs) { const ch = inputs[0] && inputs[0][0]; if (this.on && ch) { this.buf.push(Float32Array.from(ch)); this.n += ch.length; if (this.n >= 640) this.flush(); } return true; }
      }
      registerProcessor("cap", Cap);`;
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([worklet], { type: "application/javascript" })));
    const src = ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(ctx, "cap");
    node.port.onmessage = (e) => this.onFrame?.(e.data as ArrayBuffer);
    src.connect(node);
    // keep the graph alive without routing the mic to the speakers
    const sink = ctx.createGain();
    sink.gain.value = 0;
    node.connect(sink).connect(ctx.destination);
    this.ctx = ctx;
    this.node = node;
  }

  get isOpen() {
    return !!this.ctx;
  }

  async start() {
    await this.open();
    await this.ctx!.resume();
    this.capturing = true;
    this.node!.port.postMessage(true);
  }

  stop() {
    if (!this.capturing) return;
    this.capturing = false;
    this.node?.port.postMessage(false);
  }

  close() {
    this.stop();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.ctx = null;
    this.node = null;
    this.stream = null;
  }
}
