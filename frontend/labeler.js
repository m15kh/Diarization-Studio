// Label editor (the Label tab): the Timeline's ruler, waveform, zoom and playback, with one editable lane per speaker.
// Mouse: drag a segment to move it (onto another lane: it becomes that speaker's; onto "+ new speaker": a new
// speaker's), drag its edges to change where it starts / ends, drag on an empty part of a lane to add a segment,
// click to select it (the playback cursor goes to its start; a click on the selected segment puts the cursor where
// it is clicked; during playback, the segment plays), double-click to play it. A dragged edge snaps to the other
// segments' boundaries and to the playback cursor (hold ctrl to place it freely). Ruler and waveform: drag = pan,
// shift+drag = zoom to a region; wheel = zoom, shift+wheel = pan, like the Inspect tab.
// A segment that ends up overlapping another segment of its speaker merges with it. Every change can be undone;
// `onChange(note)` runs after each one and after the selection changes (`note` says what an edit did, or '').

import { Timeline, PALETTE, GUTTER, fmtTime } from './timeline.js';

const LANE_H = 34;       // height of a speaker lane (px)
const EDGE_PX = 6;       // a segment edge can be grabbed this close (px)
const SNAP_PX = 8;       // a dragged edge snaps to a boundary this close (px)
const DRAG_PX = 3;       // a press that moves less than this is a click
const MIN_LEN = 0.02;    // shortest segment (s)
const UNDO_LEVELS = 200;

const round3 = (x) => Math.round(x * 1000) / 1000;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const bySegment = (x, y) => x.s - y.s || x.e - y.e || x.k - y.k;
// Text inside a segment: dark on the light speaker colors, white on the dark ones.
const textOn = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return ((n >> 16) * 299 + ((n >> 8) & 255) * 587 + (n & 255) * 114) / 1000 > 150 ? '#1f2937' : '#fff';
};

export class LabelEditor extends Timeline {
  constructor(opts) {
    super(opts);
    this.onChange = opts.onChange;
    this.speakers = [];    // [{ name }], one lane each
    this.segs = [];        // [{ s, e, k }]: start and end (s), speaker index; sorted by start
    this.selected = null;  // one of `segs`, or null
    this.undoStack = [];
    this.redoStack = [];
    this.snap = true;
    this.snapAt = null;    // the boundary a dragged edge snapped to, drawn as a dashed line
  }

  color(k) { return PALETTE[k % PALETTE.length]; }

  name(k) { return this.speakers[k]?.name ?? 'new speaker'; }

  // Show other labels: the speakers' `names` and `segments` as [[start, end, speaker index], ...]. Clears undo.
  setLabels({ duration, names, segments }) {
    this._load(names, segments);
    this.undoStack = [];
    this.redoStack = [];
    this._rows(false, duration);
  }

  // Replace the labels as one edit: undo brings back the ones before.
  applyLabels({ names, segments }) {
    this._checkpoint();
    this._load(names, segments);
    this._rows();
    this._changed();
  }

  _load(names, segments) {
    this.speakers = names.map((name) => ({ name }));
    this.segs = segments.map(([s, e, k]) => ({ s, e, k }));
    this._sort();
    this.selected = null;
  }

  // The labels as { names, segments: [[start, end, speaker index], ...] }, sorted by start.
  labels() {
    return {
      names: this.speakers.map((sp) => sp.name),
      segments: [...this.segs].sort(bySegment).map((g) => [round3(g.s), round3(g.e), g.k]),
    };
  }

  // One lane per speaker, then an empty one: a segment dropped or drawn there gets a new speaker.
  _rows(keepView = true, duration = this.duration) {
    const rows = [{ kind: 'header', label: 'Speakers · keys 1–9 give the selected segment to that speaker' }];
    this.speakers.forEach((sp, k) => rows.push({
      kind: 'lane', k, h: LANE_H, color: this.color(k), band: k % 2 === 1, label: k < 9 ? `${k + 1} · ${sp.name}` : sp.name,
    }));
    rows.push({ kind: 'lane', k: this.speakers.length, h: LANE_H, label: '+ new speaker', muted: true, add: true });
    this.setData({ duration, rows, keepView });
  }

  _sort() { this.segs.sort(bySegment); }

  select(g) {
    if (g === this.selected) return;
    this.selected = g;
    this._changed();
  }

  // A segment just selected: the playback cursor goes to its start, so play plays all of it; during playback it
  // plays now (just that segment).
  cue(g) {
    if (this.player.paused) this.seek(g.s);
    else this.playRange(g.s, g.e);
  }

  // ---- edits ------------------------------------------------------------------------------
  // Each edit calls _checkpoint() before it changes anything (for undo) and ends with _changed().

  _checkpoint() {
    this.undoStack.push(this._snapshot());
    if (this.undoStack.length > UNDO_LEVELS) this.undoStack.shift();
    this.redoStack = [];
  }

  _snapshot() {
    return { speakers: this.speakers.map((sp) => ({ ...sp })), segs: this.segs.map((g) => ({ ...g })),
             sel: this.segs.indexOf(this.selected) };
  }

  _restore(snap) {
    Object.assign(this, { speakers: snap.speakers, segs: snap.segs, selected: snap.segs[snap.sel] || null });
    this._rows();
    this._changed();
  }

  undo() {
    if (!this.undoStack.length) return false;
    this.redoStack.push(this._snapshot());
    this._restore(this.undoStack.pop());
    return true;
  }

  redo() {
    if (!this.redoStack.length) return false;
    this.undoStack.push(this._snapshot());
    this._restore(this.redoStack.pop());
    return true;
  }

  _changed(note = '') {
    this.invalidate();
    this.onChange?.(note);
  }

  // After `g` moved, changed length or speaker: merge the segments of its speaker it now overlaps into it.
  // `what` says what the edit did, for onChange.
  _settle(g, what) {
    let merged = 0;
    for (;;) {
      const over = this.segs.filter((x) => x !== g && x.k === g.k && x.s < g.e && x.e > g.s);
      if (!over.length) break;
      for (const x of over) { g.s = Math.min(g.s, x.s); g.e = Math.max(g.e, x.e); }
      this.segs = this.segs.filter((x) => !over.includes(x));
      merged += over.length;
    }
    this._sort();
    this._changed([what, merged && `merged with ${plural(merged, 'overlapping segment')} of ${this.name(g.k)}`]
      .filter(Boolean).join(' · '));
  }

  deleteSelected() {
    const g = this.selected;
    if (!g) return false;
    this._checkpoint();
    this.segs = this.segs.filter((x) => x !== g);
    this.selected = null;
    this._changed(`Deleted a segment of ${this.name(g.k)} (${fmtTime(g.s)} – ${fmtTime(g.e)})`);
    return true;
  }

  // Cut the selected segment in two at `t`; the second part gets selected (to give it to another speaker).
  splitSelected(t) {
    const g = this.selected;
    if (!g || t < g.s + MIN_LEN || t > g.e - MIN_LEN) return false;
    this._checkpoint();
    const right = { s: round3(t), e: g.e, k: g.k };
    g.e = right.s;
    this.segs.push(right);
    this.selected = right;
    this._sort();
    this._changed(`Split at ${fmtTime(t)}`);
    return true;
  }

  // Join the selected segment with the next segment of its speaker: the gap between them becomes speech.
  joinSelected() {
    const g = this.selected;
    const next = g && this.segs.filter((x) => x !== g && x.k === g.k && x.s >= g.s).sort(bySegment)[0];
    if (!next) return false;
    this._checkpoint();
    g.e = Math.max(g.e, next.e);
    this.segs = this.segs.filter((x) => x !== next);
    this._settle(g, 'Joined with the next segment');
    return true;
  }

  // Give the selected segment to speaker k (speakers.length: a new speaker).
  setSpeaker(k) {
    const g = this.selected;
    if (!g || k < 0 || k > this.speakers.length || k === g.k) return false;
    this._checkpoint();
    if (k === this.speakers.length) this._newSpeaker();
    g.k = k;
    this._settle(g, `Now ${this.name(k)}`);
    return true;
  }

  // Set where the selected segment starts and ends (s); false when that leaves it too short.
  setTimes(s, e) {
    const g = this.selected;
    if (!g || !Number.isFinite(s) || !Number.isFinite(e)) return false;
    [s, e] = [round3(Math.max(0, s)), round3(Math.min(this.duration, e))];
    if (e - s < MIN_LEN) return false;
    if (s !== g.s || e !== g.e) {
      this._checkpoint();
      Object.assign(g, { s, e });
      this._settle(g, '');
    }
    return true;
  }

  addSpeaker() {
    this._checkpoint();
    const k = this._newSpeaker();
    this._changed(`Added ${this.name(k)}`);
    return k;
  }

  _newSpeaker() {
    const names = new Set(this.speakers.map((sp) => sp.name));
    let n = 0;
    while (names.has(`speaker_${n}`)) n++;
    this.speakers.push({ name: `speaker_${n}` });
    this._rows();
    return this.speakers.length - 1;
  }

  // Rename speaker k; false when the name is empty or another speaker has it. RTTM fields are separated by
  // whitespace, so spaces in a name become underscores.
  renameSpeaker(k, name) {
    name = name.trim().split(/\s+/).join('_');
    if (!name || this.speakers.some((sp, j) => j !== k && sp.name === name)) return false;
    if (name !== this.speakers[k].name) {
      this._checkpoint();
      this.speakers[k].name = name;
      this._rows();
      this._changed();
    }
    return true;
  }

  // Delete speaker k and its segments.
  removeSpeaker(k) {
    const n = this.segs.filter((g) => g.k === k).length, name = this.name(k);
    this._checkpoint();
    this._dropSpeaker(k);
    this._changed(`Deleted ${name}${n ? ` and ${plural(n, 'segment')}` : ''}`);
  }

  // Give every segment of speaker `from` to `to` and delete `from`; where both talked at once, the segments merge.
  mergeSpeaker(from, to) {
    if (from === to) return;
    const [a, b] = [this.name(from), this.name(to)];
    this._checkpoint();
    for (const g of this.segs) if (g.k === from) g.k = to;
    const keep = [];
    for (const g of this.segs.filter((x) => x.k === to).sort(bySegment)) {
      const last = keep[keep.length - 1];
      if (last && g.s < last.e) {
        last.e = Math.max(last.e, g.e);
        if (this.selected === g) this.selected = last;
      } else keep.push(g);
    }
    this.segs = [...this.segs.filter((x) => x.k !== to), ...keep];
    this._dropSpeaker(from);
    this._sort();
    this._changed(`Merged ${a} into ${b}`);
  }

  _dropSpeaker(k) {
    this.segs = this.segs.filter((g) => g.k !== k);
    for (const g of this.segs) if (g.k > k) g.k--;
    this.speakers.splice(k, 1);
    if (!this.segs.includes(this.selected)) this.selected = null;
    this._rows();
  }

  // Select the next (dir 1) or previous (-1) segment in time, of any speaker, move the playback cursor to its start
  // and bring it into view.
  step(dir) {
    const g = this.selected, t = this.player.currentTime;
    const i = g ? this.segs.indexOf(g) + dir
      : dir > 0 ? this.segs.findIndex((x) => x.s > t + 1e-3) : this.segs.findLastIndex((x) => x.s < t - 1e-3);
    const next = this.segs[i];
    if (!next) return false;
    this.select(next);
    this.cue(next);
    this.reveal(next);
    return true;
  }

  // Scroll the view to `g` when it is not all in view.
  reveal(g) {
    const [a, b] = this.view, span = b - a;
    if (g.s >= a && g.e <= b) return;
    if (g.e - g.s > span * 0.9) this.setView(g.s - span * 0.05, g.s + span * 0.95);
    else this.setView((g.s + g.e) / 2 - span / 2, (g.s + g.e) / 2 + span / 2);
  }

  // Esc while dragging: put everything back as it was.
  cancelDrag() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    this.snapAt = this.sel = null;
    if (d.mode === 'pan') this.setView(...d.view);
    else if (d.moved && d.seg) this._restore(this.undoStack.pop());
    this.invalidate();
  }

  // ---- mouse ------------------------------------------------------------------------------

  _bindPointer() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !this.duration) return;
      c.setPointerCapture(e.pointerId);
      const hit = this._hit(e.offsetX, e.offsetY);
      this.drag = {
        mode: hit?.seg ? hit.part : hit ? 'create' : e.shiftKey ? 'zoom' : 'pan',
        x: e.offsetX, y: e.offsetY, t: this.xToTime(e.offsetX), view: [...this.view], moved: false,
        seg: hit?.seg, orig: hit?.seg && { ...hit.seg }, lane: hit?.row.k,
        again: Boolean(hit?.seg) && hit.seg === this.selected,  // a press on the segment already selected
      };
      if (hit?.seg) this.select(hit.seg);
    });
    c.addEventListener('pointermove', (e) => {
      const d = this.drag;
      this.hoverX = e.offsetX >= GUTTER ? e.offsetX : null;
      if (d && !d.moved && Math.abs(e.offsetX - d.x) + Math.abs(e.offsetY - d.y) > DRAG_PX) this._startDrag(d, e);
      if (d?.moved) this._dragTo(d, e);
      else if (!d) c.style.cursor = this._cursorAt(e.offsetX, e.offsetY);
      this.redraw = true;
      this._tip(e);
    });
    c.addEventListener('pointerup', (e) => {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      this.snapAt = null;
      if (d.mode === 'zoom' && this.sel) {
        const s = Math.min(...this.sel), t = Math.max(...this.sel);
        this.sel = null;
        if (t - s > 0.01) this.setView(s, t);
      } else if (!d.moved) {  // a click
        if (d.seg && !d.again) this.cue(d.seg);  // selected on press
        else {  // the selected segment (to split it here, say), the ruler or the waveform: seek; an empty lane deselects
          if (d.lane != null && !d.seg) this.select(null);
          if (e.offsetX >= GUTTER) this.seek(this.xToTime(e.offsetX));
        }
      } else if (d.seg) this._drop(d);
      c.style.cursor = this._cursorAt(e.offsetX, e.offsetY);
      this.invalidate();
    });
    c.addEventListener('pointerleave', () => {
      this.hoverX = null;
      this.tooltip.hidden = true;
      this.redraw = true;
    });
    c.addEventListener('dblclick', (e) => {
      const g = this._hit(e.offsetX, e.offsetY)?.seg;
      if (g) this.playRange(g.s, g.e); else this.fit();
    });
  }

  _startDrag(d, e) {
    d.moved = true;
    if (d.mode === 'pan' || d.mode === 'zoom') return;
    this._checkpoint();
    if (d.mode === 'create') {
      if (this.snap && !(e.ctrlKey || e.metaKey)) d.t = this._snapTime(d.t, null);
      d.seg = { s: d.t, e: d.t, k: d.lane };
      this.segs.push(d.seg);
      this.selected = d.seg;
    }
  }

  _dragTo(d, e) {
    const t = Math.min(Math.max(this.xToTime(e.offsetX), 0), this.duration);
    const snap = this.snap && !(e.ctrlKey || e.metaKey), g = d.seg;
    this.snapAt = null;
    if (d.mode === 'pan') {
      const dt = ((e.offsetX - d.x) / this.plotWidth) * (d.view[1] - d.view[0]);
      this.setView(d.view[0] - dt, d.view[1] - dt);
      this.canvas.style.cursor = 'grabbing';
      return;
    }
    if (d.mode === 'zoom') { this.sel = [d.t, t]; return; }
    if (d.mode === 'body') {
      // the times follow the pointer once it has gone sideways, so a segment dragged straight up or down keeps them
      if (Math.abs(e.offsetX - d.x) > EDGE_PX) d.slide = true;
      const len = d.orig.e - d.orig.s;
      let s = d.slide ? Math.min(Math.max(d.orig.s + t - d.t, 0), this.duration - len) : d.orig.s;
      if (d.slide && snap) s = this._snapMove(s, len, g);
      Object.assign(g, { s, e: s + len, k: this._laneAt(e.offsetY)?.k ?? g.k });
      this.canvas.style.cursor = 'grabbing';
    } else if (d.mode === 'start') {
      g.s = Math.min(snap ? this._snapTime(t, g) : t, g.e - MIN_LEN);
    } else if (d.mode === 'end') {
      g.e = Math.max(snap ? this._snapTime(t, g) : t, g.s + MIN_LEN);
    } else {  // create: from where the drag started to the pointer
      const x = snap ? this._snapTime(t, g) : t;
      Object.assign(g, { s: Math.min(d.t, x), e: Math.max(d.t, x) });
    }
    this.invalidate();
  }

  // The end of a drag that changed a segment: keep the change, or drop a new segment too short to keep.
  _drop(d) {
    const g = d.seg, o = d.orig;
    if (g.e - g.s < MIN_LEN || (o && g.s === o.s && g.e === o.e && g.k === o.k)) {  // nothing to keep
      if (!o) this.segs = this.segs.filter((x) => x !== g);
      this.selected = o ? g : null;
      this.undoStack.pop();
      return this._changed();
    }
    Object.assign(g, { s: round3(g.s), e: round3(g.e) });
    if (g.k === this.speakers.length) this._newSpeaker();
    this._settle(g, !o ? `Added a segment of ${this.name(g.k)}` : g.k !== o.k ? `Now ${this.name(g.k)}` : '');
  }

  _laneAt(y) {
    return this.rows.find((r) => r.kind === 'lane' && y >= r.y && y < r.y + r.h) || null;
  }

  // What is at a point: null off the lanes, { row } on an empty part of a lane, { row, seg, part } on a segment,
  // `part` being 'start' / 'end' near its edges and 'body' in between. The selected segment wins where they overlap.
  _hit(x, y) {
    const row = this._laneAt(y);
    if (!row || x < GUTTER || !this.duration) return null;
    const t = this.xToTime(x), px = (this.view[1] - this.view[0]) / this.plotWidth;  // seconds per pixel
    const near = this.segs.filter((g) => g.k === row.k && t >= g.s - EDGE_PX * px && t <= g.e + EDGE_PX * px);
    if (!near.length) return { row };
    const inside = near.filter((g) => t >= g.s && t <= g.e);
    const gap = (g) => Math.min(Math.abs(t - g.s), Math.abs(t - g.e));
    const seg = inside.includes(this.selected) ? this.selected : inside[0] || near.sort((p, q) => gap(p) - gap(q))[0];
    const grab = Math.min(EDGE_PX, (seg.e - seg.s) / px / 3) * px;  // a narrow segment keeps a middle to move it by
    return { row, seg, part: t <= seg.s + grab ? 'start' : t >= seg.e - grab ? 'end' : 'body' };
  }

  _cursorAt(x, y) {
    const hit = this._hit(x, y);
    return !hit ? '' : !hit.seg ? 'crosshair' : hit.part === 'body' ? 'grab' : 'ew-resize';
  }

  // The boundary within SNAP_PX of `t` nearest to it, or null: the start or end of a segment other than `self`, or
  // the playback cursor.
  _nearest(t, self) {
    let best = null, dist = (SNAP_PX * (this.view[1] - this.view[0])) / this.plotWidth;
    const near = (x) => { if (Math.abs(x - t) < dist) { dist = Math.abs(x - t); best = x; } };
    for (const g of this.segs) if (g !== self) { near(g.s); near(g.e); }
    near(this.player.currentTime);
    return best;
  }

  _snapTime(t, self) {
    this.snapAt = this._nearest(t, self);
    return this.snapAt ?? t;
  }

  // A segment `len` long moved to start at `s`: snap whichever of its edges is closer to a boundary.
  _snapMove(s, len, self) {
    const a = this._nearest(s, self), b = this._nearest(s + len, self);
    if (a == null && b == null) return s;
    const useA = b == null || (a != null && Math.abs(a - s) <= Math.abs(b - s - len));
    this.snapAt = useA ? a : b;
    return Math.min(Math.max(useA ? a : b - len, 0), this.duration - len);
  }

  _tip(e) {
    const tip = this.tooltip, d = this.drag, t = this.xToTime(e.offsetX);
    const g = d ? d.seg : this._hit(e.offsetX, e.offsetY)?.seg;
    if (e.offsetX < GUTTER || t < 0 || t > this.duration || (d?.moved && !g)) { tip.hidden = true; return; }
    if (g) tip.innerHTML = `<b>${esc(this.name(g.k))}</b> · ${(g.e - g.s).toFixed(2)} s<br>${fmtTime(g.s, 3)} – ${fmtTime(g.e, 3)}`;
    else {
      const talking = this.speakers.filter((_, k) => this.segs.some((x) => x.k === k && t >= x.s && t < x.e));
      tip.innerHTML = `<b>${fmtTime(t, 3)}</b><br>${talking.map((sp) => esc(sp.name)).join(', ') || '<i>silence</i>'}`;
    }
    tip.hidden = false;
    const right = e.offsetX + 16 + tip.offsetWidth <= this.width;
    tip.style.left = `${right ? e.offsetX + 16 : e.offsetX - 16 - tip.offsetWidth}px`;
    tip.style.top = `${Math.max(0, Math.min(e.offsetY + 12, this.height - tip.offsetHeight - 4))}px`;
  }

  // ---- drawing ----------------------------------------------------------------------------

  _drawLayer() {
    super._drawLayer();
    if (!this.duration) return;
    const ctx = this.layer.getContext('2d'), th = this.theme;
    const lanes = this.rows.filter((r) => r.kind === 'lane');
    ctx.font = `11px ${th.mono}`;
    for (const row of lanes) this._drawLane(ctx, row, th);
    ctx.font = `12px ${th.font}`;
    if (this.snapAt != null && lanes.length) {  // where a dragged edge snapped to
      const x = Math.round(this.timeToX(this.snapAt)) + 0.5, last = lanes[lanes.length - 1];
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = th.accent;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, lanes[0].y); ctx.lineTo(x, last.y + last.h); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  _drawLane(ctx, row, th) {
    const [a, b] = this.view, top = row.y + 4, h = row.h - 8;
    ctx.save();
    ctx.beginPath(); ctx.rect(GUTTER, row.y, this.plotWidth, row.h); ctx.clip();
    if (row.add) {  // dashed: a segment dropped or drawn here gets a new speaker
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = th.muted;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.strokeRect(GUTTER + 0.5, top + 0.5, this.plotWidth - 1, h - 1);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = th.lane;
      ctx.fillRect(GUTTER, top, this.plotWidth, h);
    }
    for (const g of this.segs) {
      if (g.k === row.k && g !== this.selected && g.e >= a && g.s <= b) this._drawSeg(ctx, g, top, h, th, false);
    }
    if (this.selected?.k === row.k) this._drawSeg(ctx, this.selected, top, h, th, true);
    ctx.restore();
  }

  _drawSeg(ctx, g, top, h, th, selected) {
    const x0 = this.timeToX(g.s), w = Math.max(2, this.timeToX(g.e) - x0), color = this.color(g.k);
    ctx.globalAlpha = selected ? 1 : 0.8;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.roundRect(x0, top, w, h, Math.min(4, w / 2)); ctx.fill();
    ctx.globalAlpha = 1;
    if (selected) {  // outlined, with a grip on each edge
      ctx.strokeStyle = ctx.fillStyle = th.text;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.roundRect(x0 - 1, top - 1, w + 2, h + 2, 5); ctx.stroke();
      for (const x of [x0, x0 + w]) ctx.fillRect(x - 2, top + h / 2 - 6, 4, 12);
    }
    // its length, in the middle of the part in view, when it fits
    const v0 = Math.max(x0, GUTTER), v1 = Math.min(x0 + w, GUTTER + this.plotWidth), text = `${(g.e - g.s).toFixed(2)} s`;
    if (v1 - v0 >= ctx.measureText(text).width + 14) {
      ctx.fillStyle = textOn(color);
      ctx.textAlign = 'center';
      ctx.fillText(text, (v0 + v1) / 2, top + h / 2 + 0.5);
      ctx.textAlign = 'left';
    }
  }

  // The overview also shows where the labels are: a thin strip per segment, in its speaker's color.
  _drawOverview() {
    super._drawOverview();
    if (!this.duration || !this.theme) return;
    const ctx = this.overview.getContext('2d'), w = this.overview.clientWidth, h = this.overview.clientHeight;
    for (const g of this.segs) {
      const x0 = (g.s / this.duration) * w;
      ctx.fillStyle = this.color(g.k);
      ctx.fillRect(x0, h - 5, Math.max(1, (g.e / this.duration) * w - x0), 4);
    }
  }
}
