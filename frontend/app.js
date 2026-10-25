import { Timeline, PALETTE, fmtTime } from './timeline.js';

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (x, digits = 2) => (x == null ? '–' : `${(x * 100).toFixed(digits)}%`);
const secs = (x) => `${x.toFixed(2)} s`;
const basename = (p) => p.split(/[\\/]/).pop();
const dot = (color) => `<span class="dot" style="background:${color}"></span>`;
const isFolder = (path) => Boolean(path) && !/\.(wav|flac|mp3|ogg|opus|m4a|webm)$/i.test(path);
const card = (label, value, sub = '', cls = '') => `<div class="metric ${cls}"><div class="m-label">${label}</div>
  <div class="m-value">${value}</div><div class="m-sub" title="${sub}">${sub}</div></div>`;

// NeMo PostProcessingParams (same names as the postprocessing YAML) + scoring options
const SETTINGS = [
  { key: 'onset', def: 0.5, max: 1, step: 0.01, slider: true, help: 'A segment starts when the probability rises above this' },
  { key: 'offset', def: 0.5, max: 1, step: 0.01, slider: true, help: 'A segment ends when the probability falls below this' },
  { key: 'pad_onset', def: 0, max: 2, step: 0.01, help: 'Seconds added before every segment' },
  { key: 'pad_offset', def: 0, max: 2, step: 0.01, help: 'Seconds added after every segment' },
  { key: 'min_duration_on', def: 0, max: 5, step: 0.01, help: 'Drop segments shorter than this many seconds' },
  { key: 'min_duration_off', def: 0, max: 5, step: 0.01, help: 'Fill gaps shorter than this many seconds' },
  { key: 'collar', label: 'collar ± (s)', def: 0.25, max: 2, step: 0.05, help: 'Not scored: this many seconds on each side of every reference boundary (NIST / NeMo convention; pyannote collar = 2 × this)' },
  { key: 'ignore_overlap', label: 'ignore overlap', def: false, help: 'Do not score regions where two or more reference speakers talk' },
];
const PP_KEYS = SETTINGS.slice(0, 6).map((s) => s.key);

const FIELDS = {
  checkpoint: { kind: 'checkpoint', title: 'a checkpoint (.nemo / .ckpt)' },
  audio: { kind: 'audio', title: 'an audio file or folder' },
  ref: { kind: 'rttm', title: 'the reference RTTM (file or folder)' },
  hyp: { kind: 'rttm', title: 'a prediction RTTM (file or folder)' },
};

const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(`diar.${key}`)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`diar.${key}`, JSON.stringify(value)); } catch { /* storage unavailable */ }
  },
};

const player = $('#player');
const timeline = new Timeline({
  canvas: $('#timeline'), overview: $('#overview'), tooltip: $('#tooltip'), player, describe: describeTime,
});

let result = null;      // last /api/run response
let view = null;        // colors and per-speaker segments derived from `result`
let loadedAudio = '';   // audio path currently in the player
let lastAudio = '';     // audio path the RTTM fields were last paired with
let pairing = Promise.resolve();
let modelPath = '';
let runSeq = 0;
let batch = null;       // folder run: { path, ref_dir, hyp_dir, fields, checkpoint, rows: [{ audio, ref, hyp, res?, error? }] }
let batchSeq = 0;
let folderSort = 'name';
let errorSort = 'time';
let picker = null;
let rescoreTimer = null;
let statusTimer = null;

renderSettings();
restore();
bindInputs();
bindResults();
bindKeys();
initServer();

// ---- server -------------------------------------------------------------------------------

async function api(url, body) {
  const opts = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({ detail: `${res.status} ${res.statusText}` }));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join('; ') : data.detail;
    throw Object.assign(new Error(detail || res.statusText), { trace: data.trace });
  }
  return data;
}

async function initServer() {
  try {
    const [info, startup] = await Promise.all([api('/api/model'), api('/api/startup')]);
    // paths given on the server command line: applied once per server start,
    // so reloading the page keeps what was changed in the UI since
    const paths = startup.launch !== store.get('launch') ? startup.paths : {};
    store.set('launch', startup.launch);
    applyPaths(paths);
    if (info) {
      if (!val('checkpoint')) setVal('checkpoint', info.path);
      showModel(info);
    }
    if (paths.audio && (val('checkpoint') || val('hyp'))) run();
    else if (Object.keys(paths).length) setStatus('Paths set from the terminal');
  } catch { /* server not reachable yet */ }
}

function applyPaths(paths) {
  if (!Object.keys(paths).length) return;
  if (paths.audio) {
    // a new audio file: RTTMs not given in the terminal must not stay from the previous file
    for (const k of ['audio', 'ref', 'hyp']) setVal(k, paths[k] || '');
    lastAudio = paths.audio;
  } else {
    for (const k of ['ref', 'hyp']) if (paths[k]) setVal(k, paths[k]);
  }
  if (paths.checkpoint) setVal('checkpoint', paths.checkpoint);
  save();
}

async function loadModel() {
  const path = val('checkpoint');
  if (!path) return showError('Choose a checkpoint first.');
  const seq = ++runSeq;
  setStatus(`Loading ${basename(path)}…`, true);
  try {
    showModel(await api('/api/model', { path }));
    if (seq === runSeq) setStatus('Model loaded');
    hideError();
  } catch (e) {
    if (seq === runSeq) { setStatus(''); showError(e); }
  }
}

function run() {
  return isFolder(val('audio')) ? runFolder() : runFile();
}

async function runFile() {
  clearTimeout(rescoreTimer);
  const body = { audio: val('audio'), checkpoint: val('checkpoint'), ref_rttm: val('ref'), hyp_rttm: val('hyp'), ...getSettings() };
  if (!body.audio) return showError('Choose an audio file.');
  if (!body.checkpoint && !body.hyp_rttm) return showError('Choose a checkpoint, or a prediction RTTM to inspect.');
  const seq = ++runSeq;
  const loading = !body.hyp_rttm && body.checkpoint !== modelPath;
  setStatus(loading ? 'Loading model and running…' : 'Running…', true);
  try {
    const res = await api('/api/run', body);
    if (seq !== runSeq) return;  // a newer request replaced this one
    showResult(res);
    setStatus(res.source === 'model'
      ? `Done · inference ${res.infer_time.toFixed(2)} s${res.cached ? ' (cached)' : ''}`
      : 'Done · prediction from RTTM');
  } catch (e) {
    if (seq !== runSeq) return;
    setStatus('');
    showError(e);
  }
}

function scheduleRescore() {
  clearTimeout(rescoreTimer);
  rescoreTimer = setTimeout(() => {
    if (result && !isFolder(val('audio'))) runFile();
    if (batch?.rows.some((r) => r.res)) scoreFolder(++batchSeq, !batch.running);  // a stopped run stays stopped
  }, 250);
}

// ---- folder -------------------------------------------------------------------------------

// Run every audio file of a folder, one after the other; RTTMs are paired by file name (see /api/folder).
async function runFolder() {
  const fields = { audio: val('audio'), ref: val('ref'), hyp: val('hyp') };
  const checkpoint = val('checkpoint');
  if (!checkpoint && !fields.hyp) return showError('Choose a checkpoint, or a prediction RTTM folder to inspect.');
  const seq = ++batchSeq;
  ++runSeq;  // drop a single-file run still in flight
  setStatus('');
  try {
    const list = await api(`/api/folder?${new URLSearchParams(fields)}`);
    if (seq !== batchSeq) return;
    hideError();
    batch = { ...list, fields, checkpoint, rows: list.files.map((f) => ({ ...f })) };
    result = null;
    $('#results').hidden = true;
    scoreFolder(seq);
  } catch (e) {
    if (seq === batchSeq) showError(e);
  }
}

// (Re-)score the files with the current settings; `onlyDone`: just the ones that have a result.
// Model outputs are cached, so a re-score is quick.
async function scoreFolder(seq, onlyDone = false) {
  const b = batch;
  const todo = onlyDone ? b.rows.filter((r) => r.res) : b.rows;
  Object.assign(b, { running: true, stopped: false, done: 0, total: todo.length });
  for (const row of todo) {
    b.current = row;
    renderFolder();
    try {
      if (b.hyp_dir && !row.hyp) throw new Error(`No ${basename(row.audio).replace(/\.[^.]*$/, '')}.rttm in the prediction folder`);
      const res = await api('/api/run', {
        audio: row.audio, checkpoint: b.checkpoint, ref_rttm: row.ref, hyp_rttm: row.hyp, ...getSettings(), brief: true,
      });
      if (seq !== batchSeq) return;
      Object.assign(row, { res, error: '' });
      if (res.model) showModel(res.model);
    } catch (e) {
      if (seq !== batchSeq) return;
      Object.assign(row, { res: null, error: e.message });
    }
    b.done++;
  }
  Object.assign(b, { running: false, current: null });
  renderFolder();
}

function stopFolder() {
  ++batchSeq;
  Object.assign(batch, { running: false, stopped: true, current: null });
  renderFolder();
}

function renderFolder() {
  const b = batch;
  $('#folder').hidden = !b;
  if (!b) return;
  const n = b.rows.length;
  const done = b.rows.filter((r) => r.res);
  const scored = done.filter((r) => r.res.score);
  const failed = b.rows.filter((r) => r.error).length;
  $('#folder-count').textContent = `${n}`;
  $('#folder-progress').innerHTML = `${b.running ? '<span class="spinner"></span>' : ''}${esc(b.path)}${
    b.running ? ` · ${b.done} / ${b.total}` : b.stopped ? ` · stopped at ${b.done} / ${b.total}` : ''}`;
  $('#folder-stop').hidden = !b.running;

  // totals: DER over all scored speech (time-weighted), like NeMo / pyannote report it
  const sum = (rows, f) => rows.reduce((acc, r) => acc + f(r.res), 0);
  const speech = sum(scored, (res) => res.score.ref_speech);
  const rate = (x) => (speech ? x / speech : null);
  const model = done.filter((r) => r.res.source === 'model');
  const cards = [card('Files', `${done.length} / ${n}`,
    failed ? `${failed} failed` : `${scored.length} with reference`, failed ? 'warn' : '')];
  if (scored.length) {
    const [m, f, c] = ['miss_s', 'fa_s', 'conf_s'].map((k) => sum(scored, (res) => res.score[k]));
    cards.push(card('DER', pct(rate(m + f + c)), `all files · ${speech.toFixed(0)} s scored speech`, 'main'));
    cards.push(card('Missed', pct(rate(m)), secs(m), 'miss'));
    cards.push(card('False alarm', pct(rate(f)), secs(f), 'fa'));
    cards.push(card('Confusion', pct(rate(c)), secs(c), 'conf'));
  }
  cards.push(card('Audio', fmtTime(sum(done, (res) => res.duration), 0), `${done.length} files`));
  if (model.length) {
    const infer = sum(model, (res) => res.infer_time);
    cards.push(card('Inference', `${infer.toFixed(1)} s`, `RTF ${(infer / sum(model, (res) => res.duration)).toFixed(4)}`));
  }
  $('#folder-metrics').innerHTML = cards.join('');

  const hasRef = b.rows.some((r) => r.ref);
  const order = b.rows.map((_, i) => i);
  if (folderSort === 'der') order.sort((i, j) => (b.rows[j].res?.score?.der ?? -1) - (b.rows[i].res?.score?.der ?? -1));
  const head = ['file', 'length', 'speakers', ...(hasRef ? ['DER', 'missed', 'false alarm', 'confusion'] : ['speech', 'overlap']), 'inference'];
  const body = order.map((i) => {
    const r = b.rows[i], res = r.res, s = res?.score;
    const cls = i === b.active ? ' class="active"' : '';
    const first = `<td title="${esc(r.audio)}">${esc(basename(r.audio))}</td>`;
    if (r.error) return `<tr data-f="${i}"${cls}>${first}<td class="fail" colspan="${head.length - 1}">${esc(r.error)}</td></tr>`;
    if (!res) {
      const wait = r === b.current ? '<span class="spinner"></span>' : '';
      return `<tr data-f="${i}"${cls}>${first}${'<td class="num">–</td>'.repeat(head.length - 2)}<td class="num">${wait}</td></tr>`;
    }
    const spk = res.ref
      ? `<span class="${res.ref.speakers !== res.hyp.speakers ? 'warn' : ''}">${res.ref.speakers} → ${res.hyp.speakers}</span>`
      : res.hyp.speakers;
    const nums = hasRef
      ? (s ? [pct(s.der), pct(s.miss), pct(s.fa), pct(s.conf)] : ['–', '–', '–', '–'])
      : [secs(res.hyp.speech), pct(res.hyp.speech ? res.hyp.overlap / res.hyp.speech : null, 1)];
    const infer = r === b.current ? '<span class="spinner"></span>'
      : res.source === 'model' ? `${res.infer_time.toFixed(2)} s` : 'RTTM';
    return `<tr data-f="${i}"${cls}>${first}<td class="num">${fmtTime(res.duration, 1)}</td><td class="num">${spk}</td>
      ${nums.map((x) => `<td class="num">${x}</td>`).join('')}<td class="num">${infer}</td></tr>`;
  });
  $('#folder-table').innerHTML = `<table class="table">
    <thead><tr>${head.map((h, k) => `<th${k ? ' class="num"' : ''}>${h}</th>`).join('')}</tr></thead>
    <tbody>${body.join('')}</tbody></table>`;
}

// Open one file of the folder in the inspector below.
async function openFolderFile(i) {
  const r = batch.rows[i];
  batch.active = i;
  renderFolder();
  for (const k of ['audio', 'ref', 'hyp']) setVal(k, r[k]);
  lastAudio = r.audio;
  save();
  await runFile();
  if (result?.audio === r.audio) $('#results').scrollIntoView({ behavior: 'smooth' });
}

function rerunFolder() {
  for (const k of ['audio', 'ref', 'hyp']) setVal(k, batch.fields[k]);
  lastAudio = batch.fields.audio;
  save();
  runFolder();
}

function downloadFolder(kind) {
  const name = basename(batch.path) || 'folder';
  if (kind === 'rttm') {
    download(`${name}.rttm`, batch.rows.map((r) => r.res?.rttm || '').join(''), 'text/plain');
    return;
  }
  const cols = ['file', 'duration_s', 'ref_speakers', 'pred_speakers', 'der', 'miss', 'fa', 'conf', 'ref_speech_s', 'infer_s', 'error'];
  const cell = (x) => (x == null ? '' : /[",\n]/.test(String(x)) ? `"${String(x).replace(/"/g, '""')}"` : String(x));
  const lines = batch.rows.map(({ audio, res, error }) => [
    basename(audio), res?.duration, res?.ref?.speakers, res?.hyp.speakers, res?.score?.der, res?.score?.miss,
    res?.score?.fa, res?.score?.conf, res?.score?.ref_speech, res?.source === 'model' ? res.infer_time : null, error,
  ].map(cell).join(','));
  download(`${name}.csv`, [cols.join(','), ...lines].join('\n') + '\n', 'text/csv');
}

// ---- results ------------------------------------------------------------------------------

function showResult(res) {
  const sameAudio = result?.audio === res.audio;
  result = res;
  view = buildView(res);
  $('#results').hidden = false;
  hideError();
  if (res.model) showModel(res.model);
  $('#warnings').innerHTML = res.warnings.map((w) => `<div class="warning">${esc(w)}</div>`).join('');
  renderMetrics();
  renderTimeline(sameAudio);
  renderErrors();
  renderSpeakers();
  if (res.audio !== loadedAudio) loadAudio(res.audio);
  updateTime();
}

function buildView(res) {
  const group = (segs, n) => {
    const out = Array.from({ length: n }, () => []);
    for (const [s, e, k] of segs) out[k].push([s, e]);
    return out;
  };
  const nHyp = res.hyp.names.length;
  const v = {
    refSegs: res.ref ? group(res.ref.segments, res.ref.names.length) : [],
    hypSegs: group(res.hyp.segments, nHyp),
    probs: res.probs ? res.probs.map((p) => Float32Array.from(p)) : null,
    refColor: (res.ref?.names || []).map((_, r) => PALETTE[r % PALETTE.length]),
    hypColor: [],
    hypVisible: [],
  };
  v.maxProb = v.probs?.map((p) => p.reduce((m, x) => (x > m ? x : m), 0));
  v.hypOverlap = overlapRegions(v.hypSegs);
  // A predicted speaker takes the color of the reference speaker it is mapped to.
  const map = res.score?.hyp_to_ref || new Array(nHyp).fill(null);
  let extra = v.refColor.length;
  for (let h = 0; h < nHyp; h++) {
    v.hypVisible[h] = v.hypSegs[h].length > 0 || (v.maxProb?.[h] ?? 0) >= 0.1;
    v.hypColor[h] = map[h] != null ? v.refColor[map[h]] : PALETTE[(res.ref ? extra++ : h) % PALETTE.length];
  }
  return v;
}

// Where two or more speakers talk at once: [[start, end], ...]. `segsBySpeaker` holds [[start, end], ...] per speaker.
function overlapRegions(segsBySpeaker) {
  const events = segsBySpeaker.flatMap((segs, k) => segs.flatMap(([s, e]) => [[s, 1, k], [e, -1, k]]));
  events.sort((x, y) => x[0] - y[0] || x[1] - y[1]);  // at the same time, ends first
  const active = new Array(segsBySpeaker.length).fill(0);  // per speaker, so self-overlap counts once
  const out = [];
  let talking = 0, start = 0;
  for (const [t, d, k] of events) {
    const before = talking;
    active[k] += d;
    if (d > 0 && active[k] === 1) talking++;
    if (d < 0 && active[k] === 0) talking--;
    if (before < 2 && talking >= 2) start = t;
    if (before >= 2 && talking < 2 && t > start) {
      const last = out[out.length - 1];
      if (last && last[1] >= start) last[1] = t; else out.push([start, t]);
    }
  }
  return out;
}

function renderMetrics() {
  const res = result, s = res.score;
  const overlap = (st) => (st.speech ? st.overlap / st.speech : null);
  const cards = [];
  if (s) {
    cards.push(card('DER', pct(s.der), `scored speech ${s.ref_speech.toFixed(1)} s`, 'main'));
    cards.push(card('Missed', pct(s.miss), secs(s.miss_s), 'miss'));
    cards.push(card('False alarm', pct(s.fa), secs(s.fa_s), 'fa'));
    cards.push(card('Confusion', pct(s.conf), secs(s.conf_s), 'conf'));
    const nr = res.ref.speakers, nh = res.hyp.speakers;
    cards.push(card('Speakers', `${nr} → ${nh}`, 'reference → predicted', nr !== nh ? 'warn' : ''));
    const [ro, ho] = [overlap(res.ref), overlap(res.hyp)].map((x) => (x == null ? '–' : (x * 100).toFixed(1)));
    cards.push(card('Overlap', `${ro} → ${ho}%`, 'share of speech, ref → pred'));
  } else {
    cards.push(card('Speakers', res.hyp.speakers, 'predicted'));
    cards.push(card('Speech', secs(res.hyp.speech), `of ${fmtTime(res.duration)}`));
    cards.push(card('Overlap', pct(overlap(res.hyp), 1), 'share of speech'));
  }
  if (res.source === 'model') {
    cards.push(card('Inference', `${res.infer_time.toFixed(2)} s`,
      res.cached ? 'cached · re-scoring only' : `RTF ${(res.infer_time / res.duration).toFixed(4)}`));
  } else {
    cards.push(card('Prediction', 'RTTM', esc(basename(res.hyp_path))));
  }
  const a = res.audio_info;
  const audioSub = a.sample_rate ? `${a.sample_rate / 1000} kHz · ${a.channels} ch · ${esc(a.format)}` : '';
  cards.push(card('Audio', fmtTime(res.duration, 1), audioSub));
  $('#metrics').innerHTML = cards.join('');
}

function renderTimeline(keepView) {
  const res = result, v = view;
  const showAll = $('#show-all').checked;
  const { onset, offset } = getSettings();
  const rows = [];
  let hidden = 0;
  const hypRow = (h, label, band) => ({
    kind: 'hyp', label, band, color: v.hypColor[h], segs: v.hypSegs[h],
    probs: v.probs?.[h], step: res.frame_step, onset, offset, muted: !v.hypSegs[h].length,
  });

  // who speaks when: one lane per predicted speaker, overlapped speech highlighted across all lanes
  rows.push({ kind: 'header', label: 'Diarization · who speaks when · highlighted: 2+ speakers at once' });
  res.hyp.names.forEach((name, h) => {
    if (!(showAll || v.hypVisible[h] || res.score?.hyp_to_ref[h] != null)) return;
    rows.push({ kind: 'seg', label: name, color: v.hypColor[h], segs: v.hypSegs[h], highlight: v.hypOverlap,
                muted: !v.hypSegs[h].length });
  });

  if (res.score) {
    const map = res.score.hyp_to_ref;
    rows.push({ kind: 'header', label: 'Speakers · reference ▸ prediction' });
    res.ref.names.forEach((name, r) => {
      const band = r % 2 === 1;
      rows.push({ kind: 'ref', label: name, color: v.refColor[r], segs: v.refSegs[r], band });
      const h = map.indexOf(r);
      rows.push(h >= 0
        ? hypRow(h, `↳ ${res.hyp.names[h]}`, band)
        : { kind: 'hyp', label: '↳ not detected', color: v.refColor[r], segs: [], band, muted: true });
    });
    res.hyp.names.forEach((name, h) => {
      if (map[h] != null) return;
      if (showAll || v.hypVisible[h]) rows.push(hypRow(h, `${name} (extra)`, false));
      else hidden++;
    });
    const lanes = res.score.lanes;
    const scoring = `collar ±${res.score.collar} s${res.score.ignore_overlap ? ' · overlap not scored' : ''}`;
    rows.push({ kind: 'header', label: `Errors · ${scoring}` });
    rows.push({ kind: 'err', label: 'missed', colorVar: '--miss', segs: lanes.miss, unscored: lanes.unscored });
    rows.push({ kind: 'err', label: 'false alarm', colorVar: '--fa', segs: lanes.fa, unscored: lanes.unscored });
    rows.push({ kind: 'err', label: 'confusion', colorVar: '--conf', segs: lanes.conf, unscored: lanes.unscored });
    rows.push({ kind: 'err', label: 'overlap (ref)', colorVar: '--overlap', segs: lanes.overlap });
  } else if (res.probs) {  // without probabilities this would repeat the diarization lanes
    rows.push({ kind: 'header', label: 'Speaker probability · dashed: onset / offset' });
    res.hyp.names.forEach((name, h) => {
      if (showAll || v.hypVisible[h]) rows.push(hypRow(h, name, h % 2 === 1));
      else hidden++;
    });
  }
  $('#show-all-wrap').hidden = !res.probs;
  $('#show-all-label').textContent = `silent outputs${hidden ? ` (${hidden} hidden)` : ''}`;
  $('#prev-err').disabled = $('#next-err').disabled = !res.score?.regions.length;
  timeline.setData({
    duration: res.duration, rows, keepView,
    marks: res.score ? res.score.regions.map((g) => [g.start, g.end]) : [],
  });
}

function renderErrors() {
  const box = $('#errors');
  const s = result.score;
  $('#error-count').textContent = s ? `${s.regions.length}` : '';
  $('#error-sort').hidden = !s?.regions.length;
  if (!s) {
    box.innerHTML = '<p class="empty">Add a reference RTTM to see where the model makes mistakes.</p>';
    return;
  }
  if (!s.regions.length) {
    box.innerHTML = '<p class="empty">No errors in the scored region.</p>';
    return;
  }
  const regions = s.regions.map((g, i) => ({ ...g, i, total: g.miss + g.fa + g.conf }));
  if (errorSort === 'size') regions.sort((a, b) => b.total - a.total);
  const shown = regions.slice(0, 500);
  const names = (list, ids, colors) => ids.map((k) => `${dot(colors[k])}${esc(list[k])}`).join(' ');
  const tag = (cls, label, x) => (x > 0 ? `<span class="tag ${cls}">${label} ${x.toFixed(2)}</span>` : '');
  box.innerHTML = `
    <table class="table">
      <thead><tr><th></th><th>start</th><th>dur (s)</th><th title="speaker-seconds">error</th><th>reference</th><th>prediction</th></tr></thead>
      <tbody>${shown.map((g) => `
        <tr data-i="${g.i}">
          <td><button class="btn icon play" title="Play this region">▶</button></td>
          <td class="num">${fmtTime(g.start)}</td>
          <td class="num">${(g.end - g.start).toFixed(2)}</td>
          <td>${tag('miss', 'miss', g.miss)}${tag('fa', 'FA', g.fa)}${tag('conf', 'conf', g.conf)}</td>
          <td>${names(result.ref.names, g.ref, view.refColor)}</td>
          <td>${names(result.hyp.names, g.hyp, view.hypColor)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
    ${regions.length > shown.length ? `<p class="muted small">Showing ${shown.length} of ${regions.length}.</p>` : ''}`;
}

function renderSpeakers() {
  const res = result, v = view;
  const box = $('#speakers');
  if (res.score) {
    const rows = res.score.speakers.map((sp) => `
      <tr>
        <td>${sp.ref != null ? dot(v.refColor[sp.ref]) + esc(res.ref.names[sp.ref]) : '<span class="muted">—</span>'}</td>
        <td>${sp.hyp != null ? dot(v.hypColor[sp.hyp]) + esc(res.hyp.names[sp.hyp]) : '<span class="muted">not detected</span>'}</td>
        <td class="num">${sp.ref_time.toFixed(1)}</td>
        <td class="num">${sp.hyp_time.toFixed(1)}</td>
        <td class="num">${pct(sp.recall, 1)}</td>
        <td class="num">${pct(sp.precision, 1)}</td>
      </tr>`).join('');
    box.innerHTML = `
      <table class="table">
        <thead><tr><th>reference</th><th>prediction</th><th>ref (s)</th><th>pred (s)</th>
          <th title="Scored time of the reference speaker that its mapped prediction also covers">recall</th>
          <th title="Scored time of the prediction during which its mapped reference speaker talks">precision</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } else {
    const rows = res.hyp.names.map((name, h) => (!v.hypVisible[h] ? '' : `
      <tr>
        <td>${dot(v.hypColor[h])}${esc(name)}</td>
        <td class="num">${res.hyp.per_speaker[h].toFixed(1)}</td>
        <td class="num">${v.hypSegs[h].length}</td>
        <td class="num">${v.maxProb ? v.maxProb[h].toFixed(2) : '–'}</td>
      </tr>`)).join('');
    box.innerHTML = `
      <table class="table">
        <thead><tr><th>speaker</th><th>speech (s)</th><th>segments</th><th>max prob</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }
}

function describeTime(t) {
  const res = result, v = view;
  if (!res) return '';
  const on = (segs) => segs.some(([s, e]) => t >= s && t < e);
  const lines = [`<b>${fmtTime(t, 3)}</b>`];
  if (res.ref) {
    const active = res.ref.names.map((n, r) => (on(v.refSegs[r]) ? `${dot(v.refColor[r])}${esc(n)}` : '')).filter(Boolean);
    lines.push(`<div><span class="k">ref</span>${active.join(' ') || '<i>silence</i>'}</div>`);
  }
  const frame = v.probs ? Math.min(Math.floor(t / res.frame_step), v.probs[0].length - 1) : -1;
  const pred = res.hyp.names.map((name, h) => {
    const p = frame >= 0 ? v.probs[h][frame] : null;
    if (!on(v.hypSegs[h]) && !(p >= 0.05)) return '';
    return `${dot(v.hypColor[h])}${esc(name)}${p != null ? ` <span class="muted">${p.toFixed(2)}</span>` : ''}`;
  }).filter(Boolean);
  lines.push(`<div><span class="k">pred</span>${pred.join(' ') || '<i>silence</i>'}</div>`);
  if (res.score) {
    const L = res.score.lanes;
    const errors = [['missed', L.miss], ['false alarm', L.fa], ['confusion', L.conf]].filter(([, s]) => on(s)).map(([n]) => n);
    if (errors.length) lines.push(`<div><span class="k">error</span>${errors.join(', ')}</div>`);
    if (on(L.unscored)) lines.push('<div class="muted">inside collar · not scored</div>');
  }
  return lines.join('');
}

function focusRegion(g, play) {
  const pad = Math.max(0.75, (g.end - g.start) * 0.6);
  timeline.zoomTo(g.start - pad, g.end + pad);
  if (play) timeline.playRange(Math.max(0, g.start - 0.3), g.end + 0.3);
  else timeline.seek(g.start);
  for (const tr of document.querySelectorAll('#errors tr.active')) tr.classList.remove('active');
  const row = document.querySelector(`#errors tr[data-i="${result.score.regions.indexOf(g)}"]`);
  row?.classList.add('active');
  row?.scrollIntoView({ block: 'nearest' });
}

function jumpError(dir) {
  const regions = result?.score?.regions;
  if (!regions?.length) return;
  const t = player.currentTime;
  const g = dir > 0 ? regions.find((r) => r.start > t + 0.01) : [...regions].reverse().find((r) => r.start < t - 0.01);
  if (g) focusRegion(g, false);
}

// ---- audio --------------------------------------------------------------------------------

async function loadAudio(path) {
  loadedAudio = path;
  player.pause();
  timeline.setWave(null, 0, 'loading audio…');
  try {
    const res = await fetch(`/api/audio?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const bytes = await res.arrayBuffer();
    if (loadedAudio !== path) return;
    if (player.src.startsWith('blob:')) URL.revokeObjectURL(player.src);
    player.src = URL.createObjectURL(new Blob([bytes], { type: res.headers.get('content-type') || '' }));
    // decode at 16 kHz, the rate the model sees
    const decoded = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(bytes);
    if (loadedAudio === path) timeline.setWave(mixDown(decoded), decoded.sampleRate);
  } catch (e) {
    if (loadedAudio === path) timeline.setWave(null, 0, `waveform unavailable: ${e.message}`);
  }
}

function mixDown(buffer) {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += data[i] / buffer.numberOfChannels;
  }
  return out;
}

function togglePlay() {
  if (!player.src) return;
  if (player.paused) player.play(); else player.pause();
}

function updateTime() {
  $('#time').textContent = `${fmtTime(player.currentTime)} / ${fmtTime(result?.duration || 0)}`;
}

// ---- inputs -------------------------------------------------------------------------------

function val(field) { return $(`#${field}`).value.trim(); }
function setVal(field, value) { $(`#${field}`).value = value; }

// When the audio changes, RTTMs that belonged to the previous file follow the new one.
// Returns the pending lookup, so callers (Run, Enter, blur) all wait for the same one.
function audioChanged() {
  const audio = val('audio');
  if (isFolder(audio)) {
    lastAudio = audio;
    save();
  } else if (audio && audio !== lastAudio) {
    const prev = lastAudio;
    lastAudio = audio;
    pairing = (async () => {
      const sent = { ref: val('ref'), hyp: val('hyp') };
      try {
        const pair = await api(`/api/pair?${new URLSearchParams({ audio, prev_audio: prev, ...sent })}`);
        for (const k of ['ref', 'hyp']) if (val(k) === sent[k]) setVal(k, pair[k]);  // unless edited meanwhile
      } catch { /* keep the fields as they are */ }
      save();
    })();
  }
  return pairing;
}

async function stepAudio(step) {
  const current = val('audio');
  if (!current) return;
  try {
    const { path } = await api(`/api/neighbor?${new URLSearchParams({ path: current, step })}`);
    if (!path) return setStatus(step > 0 ? 'Last file in this folder' : 'First file in this folder');
    setVal('audio', path);
    await audioChanged();
    run();
  } catch (e) { showError(e); }
}

async function browseFor(field) {
  const path = await pickFile(field);
  if (!path) return;
  setVal(field, path);
  if (field === 'audio') await audioChanged();
  save();
}

async function uploadFile(field, file) {
  setStatus(`Uploading ${file.name}…`, true);
  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    setVal(field, data.path);
    if (field === 'audio') await audioChanged();
    save();
    setStatus(`Uploaded ${file.name}`);
  } catch (e) {
    setStatus('');
    showError(e);
  }
}

function fieldForFile(name, droppedOn) {
  const ext = name.slice(name.lastIndexOf('.')).toLowerCase();
  if (ext === '.rttm') return droppedOn === 'hyp' ? 'hyp' : 'ref';
  if (ext === '.nemo' || ext === '.ckpt') return 'checkpoint';
  return 'audio';
}

function bindInputs() {
  for (const btn of document.querySelectorAll('[data-browse]')) {
    btn.addEventListener('click', () => browseFor(btn.dataset.browse));
  }
  const fileInput = $('#file-input');
  for (const btn of document.querySelectorAll('[data-upload]')) {
    btn.addEventListener('click', () => {
      fileInput.dataset.field = btn.dataset.upload;
      fileInput.accept = btn.dataset.upload === 'audio' ? 'audio/*,.wav,.flac,.mp3,.ogg,.opus,.m4a' : '.rttm';
      fileInput.click();
    });
  }
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) uploadFile(fileInput.dataset.field, fileInput.files[0]);
    fileInput.value = '';
  });

  // drop files anywhere: routed by extension (an RTTM dropped on the prediction row goes there)
  document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
  document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
  document.addEventListener('drop', async (e) => {
    e.preventDefault();
    document.body.classList.remove('dragging');
    const droppedOn = e.target.closest?.('[data-drop]')?.dataset.drop;
    const files = [...e.dataTransfer.files].sort((a, b) => (fieldForFile(a.name) === 'audio' ? -1 : 0) - (fieldForFile(b.name) === 'audio' ? -1 : 0));
    for (const file of files) await uploadFile(fieldForFile(file.name, droppedOn), file);
  });

  for (const field of Object.keys(FIELDS)) {
    const input = $(`#${field}`);
    input.addEventListener('change', () => (field === 'audio' ? audioChanged() : save()));
    input.addEventListener('keydown', async (e) => {
      if (e.key !== 'Enter') return;
      if (field === 'audio') await audioChanged();
      run();
    });
  }
  $('#load-btn').addEventListener('click', loadModel);
  $('#run-btn').addEventListener('click', async () => { await audioChanged(); run(); });
  $('#prev-file').addEventListener('click', () => stepAudio(-1));
  $('#next-file').addEventListener('click', () => stepAudio(1));
  $('#reset-settings').addEventListener('click', () => { setSettings({}); save(); scheduleRescore(); });
  $('#copy-yaml').addEventListener('click', copyYaml);

  $('#folder-stop').addEventListener('click', stopFolder);
  $('#folder-rerun').addEventListener('click', rerunFolder);
  $('#folder-csv').addEventListener('click', () => downloadFolder('csv'));
  $('#folder-rttm').addEventListener('click', () => downloadFolder('rttm'));
  $('#folder-table').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-f]');
    if (tr) openFolderFile(Number(tr.dataset.f));
  });
  for (const btn of document.querySelectorAll('[data-fsort]')) {
    btn.addEventListener('click', () => {
      folderSort = btn.dataset.fsort;
      for (const b of document.querySelectorAll('[data-fsort]')) b.classList.toggle('on', b === btn);
      renderFolder();
    });
  }

  // file browser dialog
  $('#br-list').addEventListener('click', (e) => {
    const li = e.target.closest('li');
    if (li?.dataset.dir) listDir(li.dataset.dir);
    else if (li?.dataset.file) closePicker(li.dataset.file);
  });
  $('#br-path').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); listDir($('#br-path').value); }
  });
  $('#br-close').addEventListener('click', () => closePicker(null));
  $('#br-use-dir').addEventListener('click', () => closePicker($('#br-path').value, true));
  $('#browser').addEventListener('close', () => { if (picker) closePicker(null); });
}

function pickFile(field) {
  return new Promise((resolve) => {
    picker = { field, kind: FIELDS[field].kind, resolve };
    $('#br-title').textContent = `Select ${FIELDS[field].title}`;
    $('#br-use-dir').hidden = field === 'checkpoint';
    $('#browser').showModal();
    listDir(val(field) || store.get(`dir.${field}`, '') || store.get('dir.audio', ''));
  });
}

async function listDir(path) {
  const list = $('#br-list');
  try {
    const d = await api(`/api/browse?${new URLSearchParams({ path, kind: picker.kind })}`);
    $('#br-path').value = d.path;
    const items = [];
    if (d.parent) items.push(`<li class="dir" data-dir="${esc(d.parent)}">..</li>`);
    for (const x of d.dirs) items.push(`<li class="dir" data-dir="${esc(x.path)}">${esc(x.name)}</li>`);
    for (const f of d.files) {
      items.push(`<li class="file" data-file="${esc(f.path)}"><span>${esc(f.name)}</span>
        <span class="meta">${fmtSize(f.size)} · ${new Date(f.mtime * 1000).toLocaleString()}</span></li>`);
    }
    if (!d.dirs.length && !d.files.length) items.push('<li class="empty">Nothing here</li>');
    list.innerHTML = items.join('');
    list.scrollTop = 0;
  } catch (e) {
    list.innerHTML = `<li class="empty">${esc(e.message)}</li>`;
  }
}

function closePicker(path, isDir = false) {
  const p = picker;
  picker = null;
  if ($('#browser').open) $('#browser').close();
  if (path && p) store.set(`dir.${p.field}`, isDir ? path : path.replace(/[\\/][^\\/]*$/, ''));
  p?.resolve(path);
}

function fmtSize(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
  return `${bytes.toFixed(i ? 1 : 0)} ${units[i]}`;
}

// ---- settings -----------------------------------------------------------------------------

function renderSettings() {
  $('#settings').innerHTML = SETTINGS.map((s) => {
    const label = s.label || s.key;
    if (typeof s.def === 'boolean') {
      return `<label class="setting check" title="${esc(s.help)}"><input type="checkbox" data-key="${s.key}"> ${label}</label>`;
    }
    const slider = s.slider ? `<input type="range" min="0" max="${s.max}" step="${s.step}" data-key="${s.key}">` : '';
    return `<label class="setting${s.slider ? ' with-slider' : ''}" title="${esc(s.help)}"><span>${label}</span>${slider}
      <input type="number" min="0" max="${s.max}" step="${s.step}" data-key="${s.key}"></label>`;
  }).join('');
  $('#settings').addEventListener('input', (e) => {
    const key = e.target.dataset.key;
    if (!key) return;
    for (const el of document.querySelectorAll(`#settings [data-key="${key}"]`)) {
      if (el !== e.target && el.type !== 'checkbox') el.value = e.target.value;
    }
    save();
    scheduleRescore();
  });
}

function getSettings() {
  const out = {};
  for (const s of SETTINGS) {
    if (typeof s.def === 'boolean') {
      out[s.key] = $(`#settings input[type="checkbox"][data-key="${s.key}"]`).checked;
    } else {
      const x = parseFloat($(`#settings input[type="number"][data-key="${s.key}"]`).value);
      out[s.key] = Number.isFinite(x) ? x : s.def;
    }
  }
  return out;
}

function setSettings(values) {
  for (const s of SETTINGS) {
    const value = values?.[s.key] ?? s.def;
    for (const el of document.querySelectorAll(`#settings [data-key="${s.key}"]`)) {
      if (el.type === 'checkbox') el.checked = value; else el.value = value;
    }
  }
}

async function copyYaml() {
  const s = getSettings();
  const yaml = `parameters:\n${PP_KEYS.map((k) => `  ${k}: ${s[k]}`).join('\n')}\n`;
  try {
    await navigator.clipboard.writeText(yaml);
  } catch {  // clipboard API needs https or localhost
    const area = Object.assign(document.createElement('textarea'), { value: yaml });
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
  setStatus('Post-processing YAML copied');
}

function save() {
  store.set('fields', Object.fromEntries(Object.keys(FIELDS).map((k) => [k, val(k)])));
  store.set('settings', getSettings());
}

function restore() {
  const fields = store.get('fields', {});
  for (const k of Object.keys(FIELDS)) setVal(k, fields[k] || '');
  setSettings(store.get('settings', {}));
  lastAudio = val('audio');
}

// ---- status / errors ----------------------------------------------------------------------

function showModel(info) {
  modelPath = info.path;
  $('#model-info').textContent = [
    info.arch, `${info.num_speakers} speaker outputs`, `${info.sample_rate / 1000} kHz`,
    `${Math.round(info.frame_step * 1000)} ms frames`, `${info.params_m}M params`,
    info.device_name, `loaded in ${info.load_time} s`,
  ].join(' · ');
  const badge = $('#model-badge');
  badge.textContent = `${info.name} · ${info.device}`;
  badge.title = info.path;
  badge.classList.add('ok');
}

function setStatus(text, spinning = false) {
  clearInterval(statusTimer);
  const el = $('#status');
  if (!spinning) { el.textContent = text; return; }
  const started = performance.now();
  const tick = () => {
    el.innerHTML = `<span class="spinner"></span>${esc(text)} ${((performance.now() - started) / 1000).toFixed(1)} s`;
  };
  tick();
  statusTimer = setInterval(tick, 100);
}

function showError(e) {
  const box = $('#error');
  const message = typeof e === 'string' ? e : e.message;
  box.innerHTML = `<b>Error:</b> ${esc(message)}${e.trace ? `<details><summary>traceback</summary><pre>${esc(e.trace)}</pre></details>` : ''}`;
  box.hidden = false;
}

function hideError() { $('#error').hidden = true; }

// ---- results toolbar & keys ---------------------------------------------------------------

function bindResults() {
  $('#play-btn').addEventListener('click', togglePlay);
  player.addEventListener('play', () => { $('#play-btn').textContent = '❚❚'; });
  player.addEventListener('pause', () => { $('#play-btn').textContent = '▶'; });
  player.addEventListener('timeupdate', updateTime);
  player.addEventListener('seeked', updateTime);
  $('#speed').addEventListener('change', (e) => { player.playbackRate = parseFloat(e.target.value); });
  $('#zoom-in').addEventListener('click', () => timeline.zoom(0.5));
  $('#zoom-out').addEventListener('click', () => timeline.zoom(2));
  $('#zoom-fit').addEventListener('click', () => timeline.fit());
  $('#prev-err').addEventListener('click', () => jumpError(-1));
  $('#next-err').addEventListener('click', () => jumpError(1));
  $('#show-all').addEventListener('change', () => renderTimeline(true));
  $('#dl-rttm').addEventListener('click', () => download(`${result.file_id}.rttm`, result.rttm, 'text/plain'));
  $('#dl-json').addEventListener('click', () => download(`${result.file_id}.json`, JSON.stringify(result), 'application/json'));
  for (const btn of document.querySelectorAll('[data-sort]')) {
    btn.addEventListener('click', () => {
      errorSort = btn.dataset.sort;
      for (const b of document.querySelectorAll('[data-sort]')) b.classList.toggle('on', b === btn);
      renderErrors();
    });
  }
  $('#errors').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-i]');
    if (tr) focusRegion(result.score.regions[Number(tr.dataset.i)], Boolean(e.target.closest('.play')));
  });
}

function bindKeys() {
  document.addEventListener('keydown', (e) => {
    if (!result || e.ctrlKey || e.metaKey || e.altKey || e.target.closest('input, select, textarea, dialog')) return;
    const actions = {
      ' ': togglePlay,
      ArrowLeft: () => timeline.pan(-0.25),
      ArrowRight: () => timeline.pan(0.25),
      '+': () => timeline.zoom(0.5),
      '=': () => timeline.zoom(0.5),
      '-': () => timeline.zoom(2),
      0: () => timeline.fit(),
      n: () => jumpError(1),
      p: () => jumpError(-1),
    };
    const action = actions[e.key];
    if (action) { e.preventDefault(); action(); }
  });
}

function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
