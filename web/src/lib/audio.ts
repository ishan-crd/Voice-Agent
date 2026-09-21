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
    if (this.nextTime < now + 0.02) this.nextTime = now + 0.06;
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
