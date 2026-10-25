// Zoomable timeline drawn on <canvas>: time ruler, waveform, speaker lanes and error lanes,
// all sharing one time axis, plus an overview strip of the whole file.
// Mouse: wheel = zoom, shift+wheel / drag = pan, click = seek, shift+drag = zoom to region,
// double-click = fit. Rows are plain objects built by app.js:
//   { kind: 'header' | 'ref' | 'hyp' | 'err', label, color | colorVar, segs: [[start, end], ...],
//     probs?, step?, onset?, offset?, unscored?, band?, muted? }

export const PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];

const GUTTER = 150;      // left column for lane labels (px)
const PAD_RIGHT = 12;
const MIN_SPAN = 0.05;   // deepest zoom: 50 ms across the plot
const TICK_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];
const HEIGHT = { ruler: 22, wave: 100, header: 26, ref: 16, hyp: 28, err: 16 };

export function fmtTime(t, digits = 2) {
  const scale = 10 ** digits;
  t = Math.round(Math.max(0, t) * scale) / scale;
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(digits).padStart(digits ? digits + 3 : 2, '0')}`;
}

export class Timeline {
  constructor({ canvas, overview, tooltip, player, describe }) {
    Object.assign(this, { canvas, overview, tooltip, player, describe });
    this.layer = document.createElement('canvas');  // cached static drawing
    this.rows = [];
    this.marks = [];
    this.duration = 0;
    this.view = [0, 1];
    this.wave = null;
    this.waveNote = '';
    this.hoverX = null;
    this.drag = null;
    this.sel = null;
    this.stopAt = null;
    this.width = 0;
    this.dirty = true;   // static layer must be redrawn
    this.redraw = true;  // composite (cursor, hover, overview) must be redrawn
    this._bind();
    new ResizeObserver(() => { if (this.canvas.clientWidth !== this.width) this._layout(); }).observe(this.canvas);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.invalidate());
    const loop = () => { this._frame(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  setData({ duration, rows, marks, keepView }) {
    const sameLength = Math.abs(duration - this.duration) < 1e-3;
    this.duration = duration;
    this.rows = [{ kind: 'ruler' }, { kind: 'wave', label: 'waveform' }, ...rows]
      .map((row) => ({ h: HEIGHT[row.kind], ...row }));
    this.marks = marks || [];
    if (!(keepView && sameLength)) this.view = [0, duration];
    this._layout();
  }

  setWave(samples, sampleRate, note = '') {
    this.wave = samples ? buildPeaks(samples, sampleRate) : null;
    this.waveNote = note;
    this.ovPeaks = null;
    this.invalidate();
  }

  invalidate() { this.dirty = true; }

  // ---- view -------------------------------------------------------------------------------

  setView(a, b) {
    const d = this.duration || 1;
    const span = Math.min(Math.max(b - a, Math.min(MIN_SPAN, d)), d);
    const start = Math.min(Math.max(a, 0), d - span);
    this.view = [start, start + span];
    this.invalidate();
  }

  zoom(factor, at) {
    const [a, b] = this.view;
    const t = at ?? (a + b) / 2;
    const span = Math.min(Math.max((b - a) * factor, MIN_SPAN), this.duration);
    const start = t - ((t - a) / (b - a)) * span;
    this.setView(start, start + span);
  }

  pan(fraction) {
    const [a, b] = this.view;
    const d = (b - a) * fraction;
    this.setView(a + d, b + d);
  }

  fit() { this.setView(0, this.duration); }

  zoomTo(a, b) { this.setView(a, b); }

  // ---- playback ---------------------------------------------------------------------------

  seek(t) {
    this.stopAt = null;
    if (this.player.src) this.player.currentTime = Math.min(Math.max(t, 0), this.duration);
    this.redraw = true;
  }

  playRange(a, b) {
    if (!this.player.src) return;
    this.player.currentTime = Math.max(0, a);
    this.player.play();
    this.stopAt = b;
  }

  // ---- geometry ---------------------------------------------------------------------------

  get plotWidth() { return Math.max(1, this.width - GUTTER - PAD_RIGHT); }

  timeToX(t) {
    const [a, b] = this.view;
    return GUTTER + ((t - a) / (b - a)) * this.plotWidth;
  }

  xToTime(x) {
    const [a, b] = this.view;
    return a + ((x - GUTTER) / this.plotWidth) * (b - a);
  }

  _layout() {
    const dpr = window.devicePixelRatio || 1;
    let y = 0;
    for (const row of this.rows) { row.y = y; y += row.h; }
    this.dpr = dpr;
    this.width = this.canvas.clientWidth;
    this.height = y + 4;
    this.canvas.style.height = `${this.height}px`;
    for (const c of [this.canvas, this.layer]) {
      c.width = Math.round(this.width * dpr);
      c.height = Math.round(this.height * dpr);
    }
    this.overview.width = Math.round(this.overview.clientWidth * dpr);
    this.overview.height = Math.round(this.overview.clientHeight * dpr);
    this.ovPeaks = null;
    this.invalidate();
  }

  // ---- drawing ----------------------------------------------------------------------------

  _frame() {
    const p = this.player;
    if (!p.paused) {
      if (this.stopAt != null && p.currentTime >= this.stopAt) p.pause();
      const [a, b] = this.view;
      if (!this.drag && (p.currentTime > b || p.currentTime < a)) {  // follow the cursor page by page
        const span = b - a;
        this.setView(p.currentTime - span * 0.05, p.currentTime + span * 0.95);
      }
    }
    if (p.currentTime !== this.lastCursor) { this.lastCursor = p.currentTime; this.redraw = true; }
    if (this.dirty) { this._drawLayer(); this.dirty = false; this.redraw = true; }
    if (this.redraw) { this._composite(); this._drawOverview(); this.redraw = false; }
  }

  _theme() {
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    return {
      v, text: v('--text'), muted: v('--muted'), grid: v('--grid'), wave: v('--wave'), accent: v('--accent'),
      stripe: v('--stripe'), unscored: v('--unscored'), miss: v('--miss'), font: v('--font'),
    };
  }

  _drawLayer() {
    const ctx = this.layer.getContext('2d');
    const th = (this.theme = this._theme());
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    if (!this.duration) return;
    ctx.font = `12px ${th.font}`;
    ctx.textBaseline = 'middle';

    for (const row of this.rows) {
      if (row.band) { ctx.fillStyle = th.stripe; ctx.fillRect(0, row.y, this.width, row.h); }
    }

    // ruler + vertical grid
    const [a, b] = this.view;
    const pxPerSec = this.plotWidth / (b - a);
    const step = TICK_STEPS.find((s) => s * pxPerSec >= 90) ?? 3600;
    const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
    const ruler = this.rows[0];
    ctx.lineWidth = 1;
    for (let k = Math.ceil(a / step); k * step <= b; k++) {
      const x = Math.round(this.timeToX(k * step)) + 0.5;
      ctx.strokeStyle = th.grid;
      ctx.beginPath(); ctx.moveTo(x, ruler.y + 14); ctx.lineTo(x, this.height); ctx.stroke();
      ctx.fillStyle = th.muted;
      ctx.fillText(fmtTime(k * step, digits), x + 3, ruler.y + 8);
    }

    for (const row of this.rows) {
      if (row.kind !== 'ruler') this._drawLabel(ctx, row, th);
      ctx.save();
      ctx.beginPath(); ctx.rect(GUTTER, row.y, this.plotWidth, row.h); ctx.clip();
      if (row.kind === 'wave') this._drawWave(ctx, row, th);
      else if (row.kind === 'ref') this._drawSegs(ctx, row.segs, row.y + 3, row.h - 6, row.color, 0.9);
      else if (row.kind === 'hyp') this._drawHyp(ctx, row, th);
      else if (row.kind === 'err') {
        if (row.unscored) this._drawSegs(ctx, row.unscored, row.y, row.h, th.unscored, 1);
        this._drawSegs(ctx, row.segs, row.y + 2, row.h - 4, th.v(row.colorVar), 0.9);
      }
      ctx.restore();
    }
  }

  _drawLabel(ctx, row, th) {
    const mid = row.y + row.h / 2;
    if (row.kind === 'header') {
      ctx.font = `600 11px ${th.font}`;
      ctx.fillStyle = th.muted;
      ctx.fillText(row.label.toUpperCase(), 10, mid + 3);
      ctx.strokeStyle = th.grid;
      ctx.beginPath(); ctx.moveTo(10, row.y + row.h - 1.5); ctx.lineTo(this.width - PAD_RIGHT, row.y + row.h - 1.5); ctx.stroke();
      ctx.font = `12px ${th.font}`;
      return;
    }
    let x = 10;
    if (row.color || row.colorVar) {
      ctx.fillStyle = row.color || th.v(row.colorVar);
      ctx.fillRect(x, mid - 4, 8, 8);
      x += 14;
    }
    ctx.fillStyle = row.muted ? th.muted : th.text;
    ctx.fillText(ellipsize(ctx, row.label, GUTTER - x - 8), x, mid);
  }

  _drawSegs(ctx, segs, y, h, color, alpha) {
    const [a, b] = this.view;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    for (const [s, e] of segs) {
      if (e < a || s > b) continue;
      const x0 = this.timeToX(Math.max(s, a));
      ctx.fillRect(x0, y, Math.max(1, this.timeToX(Math.min(e, b)) - x0), h);
    }
    ctx.globalAlpha = 1;
  }

  _drawHyp(ctx, row, th) {
    if (!row.probs) { this._drawSegs(ctx, row.segs, row.y + 3, row.h - 6, row.color, 0.9); return; }
    const top = row.y + 3, h = row.h - 6;
    const yOf = (p) => top + (1 - p) * h;
    this._drawSegs(ctx, row.segs, top, h, row.color, 0.28);

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = th.muted;
    ctx.lineWidth = 1;
    for (const thr of new Set([row.onset, row.offset])) {
      const y = Math.round(yOf(thr)) + 0.5;
      ctx.beginPath(); ctx.moveTo(GUTTER, y); ctx.lineTo(GUTTER + this.plotWidth, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // speaker probability, one step per model frame
    const [a, b] = this.view;
    const { probs, step } = row;
    const i0 = Math.max(0, Math.floor(a / step));
    const i1 = Math.min(probs.length - 1, Math.ceil(b / step));
    ctx.beginPath();
    for (let i = i0; i <= i1; i++) {
      const y = yOf(probs[i]);
      if (i === i0) ctx.moveTo(this.timeToX(i * step), y); else ctx.lineTo(this.timeToX(i * step), y);
      ctx.lineTo(this.timeToX((i + 1) * step), y);
    }
    ctx.strokeStyle = row.color;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }

  _drawWave(ctx, row, th) {
    const mid = row.y + row.h / 2;
    ctx.strokeStyle = th.grid;
    ctx.beginPath(); ctx.moveTo(GUTTER, mid + 0.5); ctx.lineTo(GUTTER + this.plotWidth, mid + 0.5); ctx.stroke();
    const w = this.wave;
    if (!w) {
      ctx.fillStyle = th.muted;
      ctx.fillText(this.waveNote, GUTTER + 8, mid - 12);
      return;
    }
    const [a, b] = this.view;
    const spp = ((b - a) * w.sr) / this.plotWidth;  // samples per pixel
    const amp = (row.h / 2 - 4) / w.peak;
    ctx.fillStyle = ctx.strokeStyle = th.wave;
    if (spp < 2) {  // deep zoom: draw the samples themselves
      const i0 = Math.max(0, Math.floor(a * w.sr) - 1);
      const i1 = Math.min(w.samples.length - 1, Math.ceil(b * w.sr) + 1);
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) {
        const x = this.timeToX(i / w.sr), y = mid - w.samples[i] * amp;
        if (i === i0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.lineWidth = 1;
      ctx.stroke();
      return;
    }
    let level = w.levels[0];
    for (const l of w.levels) if (l.block <= spp) level = l;
    const start = a * w.sr;
    for (let px = 0; px < this.plotWidth; px++) {
      const j0 = Math.floor((start + px * spp) / level.block);
      const j1 = Math.min(Math.ceil((start + (px + 1) * spp) / level.block), level.min.length);
      let lo = Infinity, hi = -Infinity;
      for (let j = j0; j < j1; j++) {
        if (level.min[j] < lo) lo = level.min[j];
        if (level.max[j] > hi) hi = level.max[j];
      }
      if (hi >= lo) ctx.fillRect(GUTTER + px, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
    }
  }

  _composite() {
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.duration) return;
    ctx.drawImage(this.layer, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const th = this.theme;
    const line = (x, color, width) => {  // a vertical line through every row except the headers
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath();
      for (const row of this.rows) {
        if (row.kind === 'header') continue;
        ctx.moveTo(x, row.kind === 'ruler' ? row.y + 14 : row.y);
        ctx.lineTo(x, row.y + row.h);
      }
      ctx.stroke();
    };
    if (this.sel) {
      const x0 = this.timeToX(Math.min(...this.sel)), x1 = this.timeToX(Math.max(...this.sel));
      ctx.globalAlpha = 0.15; ctx.fillStyle = th.accent;
      ctx.fillRect(x0, 0, x1 - x0, this.height);
      ctx.globalAlpha = 1;
    }
    if (this.hoverX != null) line(Math.round(this.hoverX) + 0.5, th.muted, 1);
    const t = this.player.currentTime;
    if (t >= this.view[0] && t <= this.view[1]) line(this.timeToX(t), th.accent, 2);
  }

  _drawOverview() {
    const c = this.overview;
    const ctx = c.getContext('2d');
    const w = c.clientWidth, h = c.clientHeight;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.duration || !this.theme) return;
    const th = this.theme;
    const x = (t) => (t / this.duration) * w;

    if (this.wave) {
      if (!this.ovPeaks) this.ovPeaks = overviewPeaks(this.wave, this.duration, w);
      ctx.fillStyle = th.wave;
      ctx.globalAlpha = 0.6;
      const mid = (h - 8) / 2, amp = (h - 10) / 2 / this.wave.peak;
      this.ovPeaks.forEach(([lo, hi], px) => ctx.fillRect(px, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp)));
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = th.miss;
    for (const [s, e] of this.marks) ctx.fillRect(x(s), h - 6, Math.max(1, x(e) - x(s)), 6);

    const [a, b] = this.view;
    ctx.globalAlpha = 0.12; ctx.fillStyle = th.accent;
    ctx.fillRect(x(a), 0, Math.max(2, x(b) - x(a)), h);
    ctx.globalAlpha = 1; ctx.strokeStyle = th.accent; ctx.lineWidth = 1;
    ctx.strokeRect(x(a) + 0.5, 0.5, Math.max(2, x(b) - x(a)) - 1, h - 1);
    ctx.fillStyle = th.accent;
    ctx.fillRect(x(this.player.currentTime), 0, 1.5, h);
  }

  // ---- interaction ------------------------------------------------------------------------

  _bind() {
    const c = this.canvas;
    c.addEventListener('wheel', (e) => {
      if (!this.duration) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 30 : e.deltaMode === 2 ? 300 : 1;
      const dx = e.deltaX * unit, dy = e.deltaY * unit;
      if (e.shiftKey || Math.abs(dx) > Math.abs(dy)) this.pan((dx || dy) / this.plotWidth);
      else this.zoom(Math.exp(dy * 0.002), e.offsetX > GUTTER ? this.xToTime(e.offsetX) : undefined);
    }, { passive: false });

    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !this.duration) return;
      c.setPointerCapture(e.pointerId);
      this.drag = { x: e.offsetX, t: this.xToTime(e.offsetX), view: [...this.view], shift: e.shiftKey, moved: false };
    });
    c.addEventListener('pointermove', (e) => {
      const d = this.drag;
      if (d) {
        if (Math.abs(e.offsetX - d.x) > 3) d.moved = true;
        if (d.moved && d.shift) this.sel = [d.t, this.xToTime(e.offsetX)];
        else if (d.moved) {
          const dt = ((e.offsetX - d.x) / this.plotWidth) * (d.view[1] - d.view[0]);
          this.setView(d.view[0] - dt, d.view[1] - dt);
          c.style.cursor = 'grabbing';
        }
      }
      this.hoverX = e.offsetX >= GUTTER ? e.offsetX : null;
      this.redraw = true;
      this._showTooltip(e);
    });
    c.addEventListener('pointerup', (e) => {
      const d = this.drag;
      this.drag = null;
      c.style.cursor = '';
      if (!d) return;
      if (this.sel) {
        const s = Math.min(...this.sel), t = Math.max(...this.sel);
        this.sel = null;
        if (t - s > 0.01) this.setView(s, t);
      } else if (!d.moved && e.offsetX >= GUTTER) {
        this.seek(this.xToTime(e.offsetX));
      }
      this.redraw = true;
    });
    c.addEventListener('pointerleave', () => {
      this.hoverX = null;
      this.tooltip.hidden = true;
      this.redraw = true;
    });
    c.addEventListener('dblclick', () => this.fit());

    const ov = this.overview;
    const centerAt = (e) => {
      const t = (e.offsetX / ov.clientWidth) * this.duration;
      const span = this.view[1] - this.view[0];
      this.setView(t - span / 2, t + span / 2);
    };
    ov.addEventListener('pointerdown', (e) => {
      if (!this.duration) return;
      ov.setPointerCapture(e.pointerId);
      this.ovDrag = true;
      centerAt(e);
    });
    ov.addEventListener('pointermove', (e) => { if (this.ovDrag) centerAt(e); });
    ov.addEventListener('pointerup', () => { this.ovDrag = false; });

    this.player.addEventListener('pause', () => { this.stopAt = null; });
  }

  _showTooltip(e) {
    const tip = this.tooltip;
    const t = this.hoverX == null ? -1 : this.xToTime(this.hoverX);
    if (t < 0 || t > this.duration || this.drag?.moved || !this.describe) { tip.hidden = true; return; }
    tip.innerHTML = this.describe(t);
    tip.hidden = false;
    const right = e.offsetX + 16 + tip.offsetWidth <= this.width;
    tip.style.left = `${right ? e.offsetX + 16 : e.offsetX - 16 - tip.offsetWidth}px`;
    tip.style.top = `${Math.max(0, Math.min(e.offsetY + 12, this.height - tip.offsetHeight - 4))}px`;
  }
}

// Min/max pyramid (1, 32, 1024 samples per entry) so any zoom level draws quickly.
function buildPeaks(samples, sr) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  const levels = [{ block: 1, min: samples, max: samples }];
  for (let block = 32; block <= 1024; block *= 32) {
    const prev = levels[levels.length - 1];
    const k = block / prev.block;
    const n = Math.ceil(prev.min.length / k);
    const min = new Float32Array(n), max = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let lo = Infinity, hi = -Infinity;
      for (let j = i * k, end = Math.min(j + k, prev.min.length); j < end; j++) {
        if (prev.min[j] < lo) lo = prev.min[j];
        if (prev.max[j] > hi) hi = prev.max[j];
      }
      min[i] = lo;
      max[i] = hi;
    }
    levels.push({ block, min, max });
  }
  return { samples, sr, peak: peak || 1, levels };
}

function overviewPeaks(wave, duration, width) {
  const level = wave.levels[wave.levels.length - 1];
  const perPx = (duration * wave.sr) / level.block / width;
  return Array.from({ length: Math.floor(width) }, (_, px) => {
    let lo = Infinity, hi = -Infinity;
    for (let j = Math.floor(px * perPx), end = Math.min(Math.ceil((px + 1) * perPx), level.min.length); j < end; j++) {
      if (level.min[j] < lo) lo = level.min[j];
      if (level.max[j] > hi) hi = level.max[j];
    }
    return hi >= lo ? [lo, hi] : [0, 0];
  });
}

function ellipsize(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  while (text.length > 1 && ctx.measureText(`${text}…`).width > maxWidth) text = text.slice(0, -1);
  return `${text}…`;
}
