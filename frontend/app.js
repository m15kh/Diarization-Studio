import { Timeline, PALETTE, fmtTime } from './timeline.js';

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (x, digits = 2) => (x == null ? '–' : `${(x * 100).toFixed(digits)}%`);
const secs = (x) => `${x.toFixed(2)} s`;
const basename = (p) => p.split(/[\\/]/).pop();
const dot = (color) => `<span class="dot" style="background:${color}"></span>`;
const isFolder = (path) => Boolean(path) && !/\.(wav|flac|mp3|ogg|opus|m4a|webm)$/i.test(path);
// Reference → predicted speaker count; the prediction is green when right, red (▼) with too few, amber (▲) with too many.
const speakerCount = (ref, hyp, digits = 0) => {
  const d = hyp - ref, same = Math.abs(d) < (digits ? 0.05 : 0.5);
  const [cls, mark, tip] = same ? ['spk-ok', '', 'right number of speakers']
    : d < 0 ? ['spk-few', ` ▼${Math.abs(d).toFixed(digits)}`, 'too few speakers predicted (missed)']
      : ['spk-many', ` ▲${d.toFixed(digits)}`, 'too many speakers predicted (extra)'];
  return `<span title="${tip}">${ref.toFixed(digits)} → <span class="${cls}">${hyp.toFixed(digits)}${mark}</span></span>`;
};
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
  audio: { kind: 'audio', title: 'an audio file' },
  ref: { kind: 'rttm', title: 'the reference RTTM' },
  hyp: { kind: 'rttm', title: 'a prediction RTTM' },
};

// What the input fields say in each mode ('file': one audio file, 'folder': a folder of them)
const MODES = {
  file: {
    audio: 'Audio', 'audio-hint': 'one file · server path, browse, upload, or drop a file',
    ref: 'Reference RTTM', 'ref-hint': 'optional · ground truth, shows the errors · found next to the audio or in ../rttm/',
    hyp: 'Prediction RTTM', 'hyp-hint': 'optional · inspect an existing output instead of running the model',
    placeholders: { audio: '/path/to/audio.wav', ref: 'found automatically next to the audio, or ../rttm/', hyp: 'leave empty to run the checkpoint' },
  },
  folder: {
    audio: 'Audio folder', 'audio-hint': 'sub-folders included: a whole test set (test/ with spk2/wav/, spk3/wav/, ...) works · opened one file at a time',
    ref: 'Reference RTTM folder', 'ref-hint': 'optional · <name>.rttm for each <name>.wav, sub-folders included · found in rttm/ next to wav/',
    hyp: 'Prediction RTTM folder', 'hyp-hint': 'optional · existing outputs to inspect instead of running the model',
    placeholders: { audio: '/path/to/test or /path/to/test/spk2', ref: 'found automatically in ../rttm/', hyp: 'leave empty to run the checkpoint' },
  },
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
  canvas: $('#timeline'), overview: $('#overview'), tooltip: $('#tooltip'), player, describe: describeTime, onLoop: showLoop,
  focusAt: errorFocus,
});

let result = null;      // last /api/run response
let view = null;        // colors and per-speaker segments derived from `result`
let mode = 'file';      // 'file' or 'folder', see setMode
let laneSize = store.get('laneSize', '1.6');     // height multiplier of the speaker / error lanes
let viewMode = store.get('viewMode', 'simple');  // timeline layout: 'simple' or 'detailed'
const hiddenSpeakers = new Set(store.get('hiddenSpeakers', []));  // speaker keys ('ref:name' / 'hyp:name') not drawn
let loadedAudio = '';   // audio path currently in the player
let lastAudio = '';     // audio path the RTTM fields were last paired with
let pairing = Promise.resolve();
let modelPath = '';
let runSeq = 0;
let shown = null;       // { audio, ref, hyp } of `result`, re-scored when the settings change
let batch = null;       // folder: { path, ref_dir, hyp_dir, fields, checkpoint, active, rows: [{ audio, name, group, ref, hyp, res?, error? }] }
let batchSeq = 0;
let folderSort = { col: 'file', desc: false };  // folder table: click a column header to sort by it, again to flip
let setView = store.get('setView', 'abs');  // per-set chart: 'abs' (% of speech) or 'share' (share of the error)
let errorSort = 'time';
const errorTypes = new Set(store.get('errorTypes', ['miss', 'fa', 'conf']));  // error types the list and ◀ ▶ show
let picker = null;
let savedItems = [];    // saved errors (see /api/saved), newest first
const clipPlayer = new Audio();
let rescoreTimer = null;
let statusTimer = null;

renderSettings();
renderYamlHistory();
restore();
bindInputs();
bindResults();
bindKeys();
bindSaved();
initServer();
loadSaved();

// ---- server -------------------------------------------------------------------------------

async function api(url, body, method) {
  const opts = body === undefined ? { method: method || 'GET' } : {
    method: method || 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
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
    // a folder reopens at the file looked at last, when nothing has to load first
    else if (mode === 'folder' && val('audio') && (val('hyp') || (info && val('checkpoint') === info.path))) run();
    else if (Object.keys(paths).length) setStatus('Paths set from the terminal');
  } catch { /* server not reachable yet */ }
}

function applyPaths(paths) {
  if (!Object.keys(paths).length) return;
  if (paths.audio) {
    setMode(isFolder(paths.audio) ? 'folder' : 'file');
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
  const audio = val('audio');
  if (audio && isFolder(audio) !== (mode === 'folder')) {
    return showError(mode === 'folder'
      ? 'This is an audio file. Choose Single audio above, or give the folder it is in.'
      : 'This is a folder. Choose Folder above to step through its files.');
  }
  return mode === 'folder' ? runFolder() : runFile();
}

// Switch between testing one audio file and a folder. Each mode keeps its own audio / RTTM paths.
function setMode(next) {
  if (next !== mode) {
    save();
    mode = next;
    const paths = store.get(`paths.${mode}`, {});
    for (const k of ['audio', 'ref', 'hyp']) setVal(k, paths[k] || '');
    lastAudio = val('audio');
    store.set('mode', mode);
    hideError();
  }
  document.body.dataset.mode = mode;
  for (const b of document.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === mode);
  for (const el of document.querySelectorAll('[data-text]')) el.textContent = MODES[mode][el.dataset.text];
  for (const [k, text] of Object.entries(MODES[mode].placeholders)) $(`#${k}`).placeholder = text;
  $('#prev-file').title = mode === 'folder' ? 'Previous file of the folder · [' : 'Previous file in this folder (runs it) · [';
  $('#next-file').title = mode === 'folder' ? 'Next file of the folder · ]' : 'Next file in this folder (runs it) · ]';
  if (batch) renderFolder();
}

// Run one file and show it; returns the result, or null when it failed or a newer run replaced it.
async function runFile(paths = { audio: val('audio'), ref: val('ref'), hyp: val('hyp') }) {
  clearTimeout(rescoreTimer);
  const body = { audio: paths.audio, checkpoint: val('checkpoint'), ref_rttm: paths.ref, hyp_rttm: paths.hyp, ...getSettings() };
  if (!body.audio) return showError('Choose an audio file.');
  if (!body.checkpoint && !body.hyp_rttm) return showError('Choose a checkpoint, or a prediction RTTM to inspect.');
  const seq = ++runSeq;
  const loading = !body.hyp_rttm && body.checkpoint !== modelPath;
  setStatus(loading ? 'Loading model and running…' : 'Running…', true);
  try {
    const res = await api('/api/run', body);
    if (seq !== runSeq) return null;  // a newer request replaced this one
    shown = paths;
    showResult(res);
    setStatus(res.source === 'model'
      ? `Done · inference ${res.infer_time.toFixed(2)} s${res.cached ? ' (cached)' : ''}`
      : 'Done · prediction from RTTM');
    return res;
  } catch (e) {
    if (seq === runSeq) { setStatus(''); showError(e); }
    return null;
  }
}

function scheduleRescore() {
  clearTimeout(rescoreTimer);
  rescoreTimer = setTimeout(() => {
    if (result && shown) runFile(shown).then((res) => { if (res) keepInFolder(res); });
    if (mode === 'folder' && batch?.rows.some((r) => r.res)) scoreFolder(++batchSeq, batch.running ? 'all' : 'done');  // a stopped run stays stopped
  }, 250);
}

// ---- folder -------------------------------------------------------------------------------

// Open a folder: list its audio files (RTTMs paired by file name, see /api/folder) and run only the file
// looked at last time in this folder, or the first one. "Run all" runs the rest for the totals.
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
    const groups = [...new Set(batch.rows.map((r) => r.group))].filter(Boolean);
    const set = store.get('folderSet', {})[batch.path] || '';
    batch.set = groups.includes(set) ? set : '';
    $('#folder-set').hidden = !groups.length;
    $('#folder-set').innerHTML = [['', `all sets (${batch.rows.length})`],
      ...groups.map((g) => [g, `${g} (${batch.rows.filter((r) => r.group === g).length})`])]
      .map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
    $('#folder-set').value = batch.set;
    const last = store.get('folderPos', {})[batch.path];
    const inSet = setRows(batch);
    const start = inSet.find((i) => batch.rows[i].name === last) ?? inSet[0];
    await openFolderFile(start);
    if (start !== inSet[0]) setStatus(`Resumed at ${last} (${inSet.indexOf(start) + 1} / ${inSet.length}) · ⏮ first starts over`);
  } catch (e) {
    if (seq === batchSeq) showError(e);
  }
}

// (Re-)score the files with the current settings: 'all', 'done' (the ones that have a result) or 'missing'
// (the ones that do not). Model outputs are cached, so a re-score is quick.
async function scoreFolder(seq, which = 'all', everySet = false) {
  const b = batch;
  const rows = which === 'done' || everySet ? b.rows : setRows(b).map((i) => b.rows[i]);  // runs cover the chosen set
  const todo = which === 'all' ? rows : rows.filter((r) => Boolean(r.res) === (which === 'done'));
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
  $('#folder').hidden = !b || mode !== 'folder';
  if (!b) return;
  const shownIdx = setRows(b), rows = shownIdx.map((i) => b.rows[i]);
  const n = rows.length;
  const done = rows.filter((r) => r.res);
  const scored = done.filter((r) => r.res.score);
  const failed = rows.filter((r) => r.error).length;
  $('#folder-count').textContent = n < b.rows.length ? `${n} / ${b.rows.length}` : `${n}`;
  const open = b.rows[b.active], pos = shownIdx.indexOf(b.active);
  $('#folder-pos').textContent = open
    ? `${pos >= 0 ? `${pos + 1} / ${n}` : 'not in this set'} · ${open.group ? `${open.group} · ` : ''}${basename(open.audio)}` : '';
  $('#folder-first').disabled = $('#folder-prev').disabled = pos === 0;
  $('#folder-next').disabled = pos >= 0 && pos >= n - 1;
  $('#folder-progress').innerHTML = `${b.running ? '<span class="spinner"></span>' : ''}${esc(b.path)}${
    b.running ? ` · running ${b.done} / ${b.total}` : b.stopped ? ` · stopped at ${b.done} / ${b.total}` : ''}`;
  $('#folder-stop').hidden = !b.running;
  $('#folder-all').hidden = b.running || done.length + failed >= n;

  // totals: DER over all scored speech (time-weighted), like NeMo / pyannote report it
  const sum = (rows, f) => rows.reduce((acc, r) => acc + f(r.res), 0);
  const speech = sum(scored, (res) => res.score.ref_speech);
  const rate = (x) => (speech ? x / speech : null);
  const model = done.filter((r) => r.res.source === 'model');
  const cards = [card('Files', `${done.length} / ${n}`,
    failed ? `${failed} failed` : `${scored.length} with reference`, failed ? 'warn' : '')];
  if (scored.length) {
    const [m, f, c] = ['miss_s', 'fa_s', 'conf_s'].map((k) => sum(scored, (res) => res.score[k]));
    cards.push(card('DER', pct(rate(m + f + c)), `${b.set || 'all files'} · ${speech.toFixed(0)} s scored speech`, 'main'));
    cards.push(card('Missed', pct(rate(m)), secs(m), 'miss'));
    cards.push(card('False alarm', pct(rate(f)), secs(f), 'fa'));
    cards.push(card('Confusion', pct(rate(c)), secs(c), 'conf'));
    const nj = sum(scored, (res) => res.score.jer_speakers ?? 0);
    cards.push(card('JER', pct(nj ? sum(scored, (res) => res.score.jer_sum ?? 0) / nj : null), `mean over ${nj} reference speakers`));
  }
  cards.push(card('Audio', fmtTime(sum(done, (res) => res.duration), 0), `${done.length} files`));
  if (model.length) {
    const infer = sum(model, (res) => res.infer_time);
    cards.push(card('Inference', `${infer.toFixed(1)} s`, `RTF ${(infer / sum(model, (res) => res.duration)).toFixed(4)}`));
  }
  $('#folder-metrics').innerHTML = cards.join('');
  renderSets();

  const hasRef = rows.some((r) => r.ref);
  const hasGroup = !b.set && rows.some((r) => r.group);  // one set: its name is in the selector
  const head = ['file', ...(hasGroup ? ['set'] : []), 'length', 'speakers', ...(hasRef ? ['DER', 'missed', 'false alarm', 'confusion', 'JER'] : ['speech', 'overlap']), 'inference'];
  const order = sortFolder(b, shownIdx, head.includes(folderSort.col) ? folderSort : { col: 'file', desc: false });
  const body = order.map((i) => {
    const r = b.rows[i], res = r.res, s = res?.score;
    const cls = i === b.active ? ' class="active"' : '';
    const lead = hasGroup ? 2 : 1;
    const grade = derGrade(res?.score?.der);
    const first = `<td title="${esc(r.audio)}"${grade ? ` class="grade ${grade}"` : ''}>${esc(basename(r.audio))}</td>${
      hasGroup ? `<td>${esc(r.group)}</td>` : ''}`;
    if (r.error) return `<tr data-f="${i}"${cls}>${first}<td class="fail" colspan="${head.length - lead}">${esc(r.error)}</td></tr>`;
    if (!res) {
      const wait = r === b.current ? '<span class="spinner"></span>' : '';
      return `<tr data-f="${i}"${cls}>${first}${'<td class="num">–</td>'.repeat(head.length - lead - 1)}<td class="num">${wait}</td></tr>`;
    }
    const spk = res.ref
      ? speakerCount(res.ref.speakers, res.hyp.speakers)
      : res.hyp.speakers;
    const nums = hasRef
      ? (s ? [heatCell(s.der, '--miss', 0.3, `der ${grade}`), heatCell(s.miss, '--miss', 0.15), heatCell(s.fa, '--fa', 0.15),
              heatCell(s.conf, '--conf', 0.15), `<td class="num">${pct(s.jer)}</td>`]
           : Array(5).fill('<td class="num">–</td>'))
      : [secs(res.hyp.speech), pct(res.hyp.speech ? res.hyp.overlap / res.hyp.speech : null, 1)].map((x) => `<td class="num">${x}</td>`);
    const infer = r === b.current ? '<span class="spinner"></span>'
      : res.source === 'model' ? `${res.infer_time.toFixed(2)} s` : 'RTTM';
    return `<tr data-f="${i}"${cls}>${first}<td class="num">${fmtTime(res.duration, 1)}</td><td class="num">${spk}</td>
      ${nums.join('')}<td class="num">${infer}</td></tr>`;
  });
  $('#folder-table').innerHTML = `<table class="table">
    <thead><tr>${head.map((h, k) => {
      const on = h === folderSort.col, arrow = on ? (folderSort.desc ? ' ▾' : ' ▴') : '';
      return `<th data-sort="${h}" class="sortable${k && h !== 'set' ? ' num' : ''}${on ? ' on' : ''}" title="Sort by ${h}">${h}${arrow}</th>`;
    }).join('')}</tr></thead>
    <tbody>${body.join('')}</tbody></table>`;
  // keep the open file in view inside the table (scrollIntoView would also scroll the page)
  const box = $('#folder-table'), tr = box.querySelector('tr.active');
  if (tr && b.inView !== b.active) {
    b.inView = b.active;
    if (tr.offsetTop < box.scrollTop + tr.offsetHeight || tr.offsetTop + tr.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTop = tr.offsetTop - box.clientHeight / 2;
    }
  }
}

// A row's value in a folder table column, for sorting (null: no value yet, sorted last).
const FOLDER_COLS = {
  file: (r) => basename(r.audio).toLowerCase(),
  set: (r) => r.group || '',
  length: (r) => r.res?.duration,
  speakers: (r) => r.res?.hyp.speakers,
  DER: (r) => r.res?.score?.der,
  missed: (r) => r.res?.score?.miss,
  'false alarm': (r) => r.res?.score?.fa,
  confusion: (r) => r.res?.score?.conf,
  JER: (r) => r.res?.score?.jer,
  speech: (r) => r.res?.hyp.speech,
  overlap: (r) => (r.res?.hyp.speech ? r.res.hyp.overlap / r.res.hyp.speech : null),
  inference: (r) => (r.res?.source === 'model' ? r.res.infer_time : null),
};

function sortFolder(b, idx, { col, desc }) {
  const get = FOLDER_COLS[col] ?? FOLDER_COLS.file, dir = desc ? -1 : 1;
  return idx.map((i) => [i, get(b.rows[i])]).sort(([i, x], [j, y]) => {
    if (x == null || y == null) return x == null && y == null ? i - j : x == null ? 1 : -1;
    const d = typeof x === 'string' ? x.localeCompare(y, undefined, { numeric: true }) : x - y;
    return d ? d * dir : i - j;
  }).map(([i]) => i);
}

function setFolderSort(col, desc) {
  folderSort = { col, desc };
  for (const btn of document.querySelectorAll('[data-fsort]')) {
    btn.classList.toggle('on', btn.dataset.fsort === (col === 'file' && !desc ? 'name' : col === 'DER' && desc ? 'der' : ''));
  }
  renderFolder();
}

const FOLDER_MEMORY = 50;  // folders whose last opened file is remembered

// Open one file of the folder in the inspector below, remember it for the next visit, and prepare the next file.
async function openFolderFile(i, scroll = false) {
  const b = batch, r = b.rows[i];
  if (!r) return;
  b.active = i;
  renderFolder();
  const positions = Object.entries(store.get('folderPos', {})).filter(([path]) => path !== b.path);
  store.set('folderPos', Object.fromEntries([[b.path, r.name], ...positions].slice(0, FOLDER_MEMORY)));
  if (b.hyp_dir && !r.hyp) return showError(`No ${basename(r.audio).replace(/\.[^.]*$/, '')}.rttm in the prediction folder`);
  const res = await runFile(r);
  if (!res || batch !== b) return;
  keepInFolder(res);
  if (scroll) $('#results').scrollIntoView({ behavior: 'smooth' });
  const inSet = setRows(b);
  prefetch(b, inSet[inSet.indexOf(i) + 1]);
}

// A file's numbers for the folder table, like the server's `brief` result.
function keepInFolder(res) {
  const row = batch?.rows.find((r) => r.audio === res.audio);
  if (!row) return;
  const without = (o, keys) => o && Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k)));
  row.res = { ...without(res, ['probs']), hyp: without(res.hyp, ['segments']), ref: without(res.ref, ['segments']),
              score: without(res.score, ['lanes', 'regions', 'speakers']) };
  row.error = '';
  renderFolder();
}

// Run the model on a file in the background, so it opens at once (its output is cached on the server).
async function prefetch(b, i) {
  const r = b.rows[i];
  if (!r || r.res || r.hyp || b.hyp_dir || b.running || !b.checkpoint) return;
  const settings = getSettings();
  try {
    const res = await api('/api/run', { audio: r.audio, checkpoint: b.checkpoint, ref_rttm: r.ref, ...settings, brief: true });
    // settings changed meanwhile: the numbers are stale, but the model output is cached all the same
    if (batch === b && !r.res && JSON.stringify(settings) === JSON.stringify(getSettings())) {
      Object.assign(r, { res, error: '' });
      renderFolder();
    }
  } catch { /* the error shows when the file is opened */ }
}

// Error totals over some files of the folder: DER and its parts over all their scored speech (time-weighted,
// like NeMo / pyannote), JER over all their reference speakers, and how often the speaker count is right.
function aggregate(rows) {
  const done = rows.filter((r) => r.res), scored = done.filter((r) => r.res.score);
  const sum = (list, f) => list.reduce((acc, r) => acc + (f(r.res) || 0), 0);
  const speech = sum(scored, (res) => res.score.ref_speech);
  const [miss, fa, conf] = ['miss_s', 'fa_s', 'conf_s'].map((k) => sum(scored, (res) => res.score[k]));
  const nj = sum(scored, (res) => res.score.jer_speakers);
  const counted = done.filter((r) => r.res.ref);
  const diff = counted.map((r) => r.res.hyp.speakers - r.res.ref.speakers);
  const refSpeech = sum(counted, (res) => res.ref.speech);
  return {
    n: rows.length, done: done.length, scored: scored.length, speech, miss, fa, conf,
    der: speech ? (miss + fa + conf) / speech : null,
    jer: nj ? sum(scored, (res) => res.score.jer_sum) / nj : null,
    counted: counted.length,
    exact: diff.filter((d) => d === 0).length, under: diff.filter((d) => d < 0).length, over: diff.filter((d) => d > 0).length,
    refSpk: counted.length ? sum(counted, (res) => res.ref.speakers) / counted.length : null,
    hypSpk: counted.length ? sum(counted, (res) => res.hyp.speakers) / counted.length : null,
    overlap: refSpeech ? sum(counted, (res) => res.ref.overlap) / refSpeech : null,
  };
}

const ERROR_PARTS = [['miss', 'missed', '--miss'], ['fa', 'false alarm', '--fa'], ['conf', 'confusion', '--conf']];

// Per set (sub-folder, e.g. spk2 ... spk8): one bar split into missed / false alarm / confusion, and a table.
// DER grade of a file: 'good' under 5%, 'fair' under 15%, 'poor' from there (colors the file name and the DER).
function derGrade(der) {
  return der == null ? '' : der < 0.05 ? 'good' : der < 0.15 ? 'fair' : 'poor';
}

// A table cell tinted in `colorVar`, the stronger the larger `x` (full strength at `full`), so bad files stand out.
function heatCell(x, colorVar, full, cls = '') {
  const heat = Math.round(Math.min(1, x / full) * 45);
  return `<td class="num heat ${cls}" style="--heat-color: var(${colorVar}); --heat: ${heat}%">${pct(x)}</td>`;
}

function renderSets() {
  const b = batch;
  const groups = [...new Set(b.rows.map((r) => r.group))].filter(Boolean);
  const box = $('#set-analysis');
  box.hidden = groups.length < 2;
  if (box.hidden) return;
  if (box.dataset.init !== b.path) { box.open = store.get('setsOpen', true); box.dataset.init = b.path; }
  const sets = groups.map((g) => ({ g, ...aggregate(b.rows.filter((r) => r.group === g)) }));
  for (const btn of document.querySelectorAll('[data-setview]')) btn.classList.toggle('on', btn.dataset.setview === setView);
  const done = sets.reduce((a, s) => a + s.done, 0);
  $('#set-coverage').textContent = `${done} / ${b.rows.length} files run${done < b.rows.length ? ' · numbers cover these only' : ''}`;
  $('#set-run-all').hidden = b.running || done >= b.rows.length;

  const max = Math.max(1e-9, ...sets.map((s) => s.der ?? 0));
  const p1 = (x) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`);
  // one bar per set, then one for the whole folder (data-set "": click shows all sets)
  const bar = (s, key, name, cls = '') => {
    const label = `<span class="set-label${key === (b.set || '') ? ' on' : ''}${cls}" data-set="${esc(key)}"
      title="${key ? `Show only ${esc(key)}` : 'Show all sets'}">${esc(name)} <span class="muted">${s.scored}/${s.n}</span></span>`;
    if (s.der == null) return `${label}<div class="track empty" data-set="${esc(key)}"></div><span class="total muted">not run</span>`;
    const err = s.miss + s.fa + s.conf;
    const parts = ERROR_PARTS.map(([k, name, color]) => {
      const x = s[k] / s.speech, share = err ? s[k] / err : 0;
      const w = setView === 'share' ? share * 100 : (x / max) * 100;
      if (w <= 0) return '';
      const text = setView === 'share' ? `${Math.round(share * 100)}%` : p1(x);
      return `<span style="width:${w}%;background:var(${color})" data-set="${esc(key)}" data-part="${k}">${w > 9 ? text : ''}</span>`;
    }).join('');
    return `${label}<div class="track" data-set="${esc(key)}">${parts}</div><span class="total${cls}">${p1(s.der)}</span>`;
  };
  $('#set-chart').innerHTML = sets.map((s) => bar(s, s.g, s.g)).join('')
    + `<div class="set-sep"></div>${bar(aggregate(b.rows), '', 'all sets', ' all')}`;

  const main = (s) => {
    const top = ERROR_PARTS.reduce((a, p) => (s[p[0]] > s[a[0]] ? p : a));
    const err = s.miss + s.fa + s.conf;
    return err ? `<i class="dot" style="background:var(${top[2]})"></i>${top[1]} ${Math.round((s[top[0]] / err) * 100)}%` : '–';
  };
  const share = (x, n) => (n ? `${Math.round((x / n) * 100)}%` : '–');
  const total = { g: 'all sets', ...aggregate(b.rows) };
  const row = (s, cls = '') => `<tr class="${cls}${s.g === b.set ? ' active' : ''}" ${s.g !== 'all sets' ? `data-set="${esc(s.g)}"` : ''}>
    <td>${esc(s.g)}</td><td class="num">${s.done} / ${s.n}</td>
    <td class="num">${p1(s.der)}</td><td class="num">${p1(s.speech ? s.miss / s.speech : null)}</td>
    <td class="num">${p1(s.speech ? s.fa / s.speech : null)}</td><td class="num">${p1(s.speech ? s.conf / s.speech : null)}</td>
    <td class="num">${p1(s.jer)}</td><td>${s.der == null ? '–' : main(s)}</td>
    <td class="num">${s.refSpk == null ? '–' : speakerCount(s.refSpk, s.hypSpk, 1)}</td>
    <td class="num">${share(s.exact, s.counted)}</td><td class="num">${share(s.under, s.counted)}</td><td class="num">${share(s.over, s.counted)}</td>
    <td class="num">${p1(s.overlap)}</td></tr>`;
  $('#set-table').innerHTML = `<table class="table">
    <thead><tr><th>set</th><th class="num">files run</th><th class="num">DER</th><th class="num">missed</th>
      <th class="num">false alarm</th><th class="num">confusion</th><th class="num">JER</th><th>main error</th>
      <th class="num" title="Mean number of speakers per file: reference → predicted">speakers</th>
      <th class="num" title="Files where the predicted number of speakers is right">count right</th>
      <th class="num" title="Files with fewer predicted speakers than in the reference">too few</th>
      <th class="num" title="Files with more predicted speakers than in the reference">too many</th>
      <th class="num" title="Share of reference speech where 2+ speakers talk at once">overlap</th></tr></thead>
    <tbody>${sets.map((s) => row(s)).join('')}${row(total, 'total')}</tbody></table>`;
}

// Hover text for one part of a per-set bar.
function setTip(e) {
  const tip = $('#set-tip');
  const el = e.target.closest('[data-part]');
  if (!el) { tip.hidden = true; return; }
  const set = el.dataset.set;  // '' = the whole folder
  const s = aggregate(set ? batch.rows.filter((r) => r.group === set) : batch.rows);
  const err = s.miss + s.fa + s.conf;
  const lines = ERROR_PARTS.map(([k, name, color]) => {
    const line = `<i class="dot" style="background:var(${color})"></i>${name}: ${(s[k] / s.speech * 100).toFixed(2)}% of speech · ${
      err ? Math.round((s[k] / err) * 100) : 0}% of the error · ${s[k].toFixed(1)} s`;
    return k === el.dataset.part ? `<b>${line}</b>` : line;
  });
  tip.innerHTML = `<b>${esc(set || 'all sets')}</b> · DER ${(s.der * 100).toFixed(2)}% · ${s.scored} files<br>${lines.join('<br>')}`;
  tip.hidden = false;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  Object.assign(tip.style, { left: `${x}px`, top: `${e.clientY + 14}px` });
}

// Indices of the rows in the chosen set (all rows when no set is chosen).
function setRows(b) {
  return b.rows.flatMap((r, i) => (!b.set || r.group === b.set ? [i] : []));
}

function stepFolder(step) {
  const inSet = setRows(batch), pos = inSet.indexOf(batch.active);
  const i = pos < 0 ? inSet[0] : inSet[pos + step];  // the open file is from another set: start this one
  if (i == null) return setStatus(step > 0 ? 'Last file in this set' : 'First file in this set');
  openFolderFile(i);
}

function chooseSet(set) {
  const b = batch;
  b.set = set;
  const sets = Object.entries(store.get('folderSet', {})).filter(([path]) => path !== b.path);
  store.set('folderSet', Object.fromEntries([[b.path, set], ...sets].slice(0, FOLDER_MEMORY)));
  const inSet = setRows(b);
  if (!inSet.includes(b.active)) openFolderFile(inSet[0]);
  else renderFolder();
}

function rerunFolder() {
  for (const k of ['audio', 'ref', 'hyp']) setVal(k, batch.fields[k]);
  lastAudio = batch.fields.audio;
  batch.checkpoint = val('checkpoint');
  save();
  scoreFolder(++batchSeq, 'all');
}

function downloadFolder(kind) {
  const name = [basename(batch.path) || 'folder', batch.set].filter(Boolean).join('_');
  const rows = setRows(batch).map((i) => batch.rows[i]);  // the chosen set
  if (kind === 'rttm') {
    download(`${name}.rttm`, rows.map((r) => r.res?.rttm || '').join(''), 'text/plain');
    return;
  }
  const cols = ['file', 'group', 'duration_s', 'ref_speakers', 'pred_speakers', 'der', 'miss', 'fa', 'conf', 'jer', 'ref_speech_s', 'infer_s', 'error'];
  const cell = (x) => (x == null ? '' : /[",\n]/.test(String(x)) ? `"${String(x).replace(/"/g, '""')}"` : String(x));
  const lines = rows.map(({ audio, group, res, error }) => [
    basename(audio), group, res?.duration, res?.ref?.speakers, res?.hyp.speakers, res?.score?.der, res?.score?.miss,
    res?.score?.fa, res?.score?.conf, res?.score?.jer, res?.score?.ref_speech, res?.source === 'model' ? res.infer_time : null, error,
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
  if (res.audio !== loadedAudio) { timeline.setLoop(null); loadAudio(res.audio); }
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
    cards.push(card('JER', pct(s.jer), `mean over ${s.jer_speakers} reference speakers`));
    const nr = res.ref.speakers, nh = res.hyp.speakers;
    cards.push(card('Speakers', speakerCount(nr, nh), 'reference → predicted'));
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
  const showDiff = $('#show-diff').checked && res.score;
  const { onset, offset } = getSettings();
  const rows = [];
  let hidden = 0;
  const map = res.score?.hyp_to_ref || [];
  const shown = (key) => !hiddenSpeakers.has(key);
  const chips = [];
  const hypRow = (h, label, band) => ({
    kind: 'hyp', label, band, spk: hypKey(h), color: v.hypColor[h], segs: v.hypSegs[h],
    probs: v.probs?.[h], step: res.frame_step, onset, offset, muted: !v.hypSegs[h].length,
  });

  // who speaks when: one lane per predicted speaker, overlapped speech highlighted across all lanes
  // (can be hidden, unless nothing else would be drawn: a prediction RTTM without reference)
  const showDiar = $('#show-diar').checked || (!res.score && !res.probs);
  if (showDiar) rows.push({ kind: 'header', label: 'Diarization · who speaks when · highlighted: 2+ speakers at once' });
  res.hyp.names.forEach((name, h) => {
    if (!(showAll || v.hypVisible[h] || res.score?.hyp_to_ref[h] != null)) return;
    if (!res.score) chips.push({ key: hypKey(h), label: name, color: v.hypColor[h] });
    if (!showDiar || !shown(hypKey(h))) return;
    rows.push({ kind: 'seg', label: name, spk: hypKey(h), color: v.hypColor[h], segs: v.hypSegs[h], highlight: v.hypOverlap,
                muted: !v.hypSegs[h].length });
  });

  if (res.score) {
    const lanes = res.score.lanes;
    const unscored = mergeSegs(lanes.unscored);
    // per speaker: reference time the prediction misses, and predicted time the reference does not have
    const diff = (refSegs, hypSegs) => (showDiff ? {
      miss: subtractSegs(subtractSegs(mergeSegs(refSegs), mergeSegs(hypSegs)), unscored),
      extra: subtractSegs(subtractSegs(mergeSegs(hypSegs), mergeSegs(refSegs)), unscored),
    } : null);
    rows.push({ kind: 'header', label: `Speakers · reference ▸ prediction${showDiff ? ' · red: missed · amber: extra' : ''}` });
    res.ref.names.forEach((name, r) => {
      chips.push({ key: `ref:${name}`, label: name, color: v.refColor[r] });
      if (!shown(`ref:${name}`)) return;
      const band = r % 2 === 1;
      rows.push({ kind: 'ref', label: name, tag: 'ref', spk: `ref:${name}`, color: v.refColor[r], segs: v.refSegs[r], band });
      const h = map.indexOf(r);
      rows.push(h >= 0
        ? { ...hypRow(h, res.hyp.names[h], band), tag: 'pred', diff: diff(v.refSegs[r], v.hypSegs[h]) }
        : { kind: 'hyp', label: 'not detected', tag: 'pred', spk: `ref:${name}`, color: v.refColor[r], segs: [], band, muted: true,
            diff: diff(v.refSegs[r], []) });
    });
    res.hyp.names.forEach((name, h) => {
      if (map[h] != null) return;
      if (!(showAll || v.hypVisible[h])) { hidden++; return; }
      chips.push({ key: hypKey(h), label: `${name} (extra)`, color: v.hypColor[h] });
      if (shown(hypKey(h))) rows.push({ ...hypRow(h, `${name} (extra)`, false), tag: 'pred', diff: diff([], v.hypSegs[h]) });
    });
    rows.push(...overlapRows(diff));
    const scoring = `collar ±${res.score.collar} s${res.score.ignore_overlap ? ' · overlap not scored' : ''}`;
    const sc = res.score, sum = (segs) => segs.reduce((acc, [s, e]) => acc + e - s, 0);
    rows.push({ kind: 'header', label: `Errors · ${derParts(sc)} · ${scoring}` });
    rows.push({ kind: 'err', label: `missed · ${pct(sc.miss)}`, colorVar: '--miss', segs: lanes.miss, unscored: lanes.unscored, numbers: true });
    rows.push({ kind: 'err', label: `false alarm · ${pct(sc.fa)}`, colorVar: '--fa', segs: lanes.fa, unscored: lanes.unscored, numbers: true });
    rows.push({ kind: 'err', label: `confusion · ${pct(sc.conf)}`, colorVar: '--conf', segs: lanes.conf, unscored: lanes.unscored, numbers: true });
    rows.push({ kind: 'err', label: `ref overlap · ${sum(lanes.overlap).toFixed(1)} s`, colorVar: '--overlap', segs: lanes.overlap });
  } else if (res.probs) {  // without probabilities this would repeat the diarization lanes
    rows.push({ kind: 'header', label: 'Speaker probability · dashed: onset / offset' });
    res.hyp.names.forEach((name, h) => {
      if (!(showAll || v.hypVisible[h])) hidden++;
      else if (shown(hypKey(h))) rows.push(hypRow(h, name, h % 2 === 1));
    });
  }
  // with fewer than two speakers there is no filter to undo a hidden one with, so show it
  if (chips.length < 2 && chips.some((c) => hiddenSpeakers.has(c.key))) {
    return setHiddenSpeakers([...hiddenSpeakers].filter((k) => !chips.some((c) => c.key === k)));
  }
  if (viewMode === 'simple') rows.splice(0, rows.length, ...simpleRows(shown));
  renderSpeakerFilter(chips);
  $('#show-all-wrap').hidden = !res.probs;
  $('#show-diff-wrap').hidden = !res.score;
  $('#show-diar-wrap').hidden = viewMode !== 'detailed' || (!res.score && !res.probs);
  for (const b of document.querySelectorAll('[data-view]')) b.classList.toggle('on', b.dataset.view === viewMode);
  $('#show-all-label').textContent = `silent outputs${hidden ? ` (${hidden} hidden)` : ''}`;
  $('#prev-err').disabled = $('#next-err').disabled = !res.score?.regions.length;
  timeline.setData({
    duration: res.duration, rows, keepView, laneScale: parseFloat(laneSize),
    marks: res.score ? res.score.regions.map((g) => [g.start, g.end]) : [],
  });
}

// The simple layout: one compact lane per speaker with the reference on top and its matched prediction right
// below (same color; red / amber where they differ), then one error lane.
function simpleRows(shown) {
  const res = result, v = view;
  const showAll = $('#show-all').checked;
  const visible = (h) => (showAll || v.hypVisible[h]) && shown(hypKey(h));
  if (!res.score) {
    const tracks = [];
    res.hyp.names.forEach((_, h) => { if (visible(h)) tracks.push({ color: v.hypColor[h], segs: v.hypSegs[h] }); });
    return [{ kind: 'header', label: 'Prediction · one track per speaker' },
            { kind: 'stack', label: 'prediction', tracks, h: Math.max(26, tracks.length * 8 + 6) }, ...overlapRows(null)];
  }
  const map = res.score.hyp_to_ref, lanes = res.score.lanes;
  const unscored = mergeSegs(lanes.unscored);
  const diff = (refSegs, hypSegs) => ($('#show-diff').checked ? {
    miss: subtractSegs(subtractSegs(mergeSegs(refSegs), mergeSegs(hypSegs)), unscored),
    extra: subtractSegs(subtractSegs(mergeSegs(hypSegs), mergeSegs(refSegs)), unscored),
  } : null);
  const pair = (spk, refName, hypName, color, refSegs, hypSegs, band) => ({
    kind: 'stack', label: refName, spk, color, band, h: 36, diff: diff(refSegs, hypSegs),
    tracks: [{ color, segs: refSegs, label: refName, tag: 'ref', bgVar: '--track-ref' },
             { color, segs: hypSegs, label: hypName, tag: 'pred', bgVar: '--track-pred' }],
  });
  const rows = [{ kind: 'header', label: 'Per speaker · REF: reference (top) · PRED: prediction (bottom, darker) · red: missed · amber: extra' }];
  res.ref.names.forEach((name, r) => {
    if (!shown(`ref:${name}`)) return;
    const h = map.indexOf(r);
    rows.push(pair(`ref:${name}`, name, h >= 0 ? res.hyp.names[h] : 'not detected', v.refColor[r], v.refSegs[r],
                   h >= 0 ? v.hypSegs[h] : [], r % 2 === 1));
  });
  res.hyp.names.forEach((name, h) => {  // predicted speakers with no reference speaker
    if (map[h] == null && visible(h)) rows.push(pair(hypKey(h), '— no reference', `${name} (extra)`, v.hypColor[h], [], v.hypSegs[h], false));
  });
  rows.push(...overlapRows(diff));
  const scoring = `collar ±${res.score.collar} s${res.score.ignore_overlap ? ' · overlap not scored' : ''}`;
  rows.push(
    { kind: 'header', label: `Errors · ${derParts(res.score)} · ${scoring}` },
    { kind: 'stack', label: 'errors', h: 48, unscored: lanes.unscored,
      tracks: [{ colorVar: '--miss', segs: lanes.miss, label: `missed · ${pct(res.score.miss)}`, numbers: true },
               { colorVar: '--fa', segs: lanes.fa, label: `false alarm · ${pct(res.score.fa)}`, numbers: true },
               { colorVar: '--conf', segs: lanes.conf, label: `confusion · ${pct(res.score.conf)}`, numbers: true }] },
  );
  return rows;
}

// "DER 16.63% = (missed 0.00 s + false alarm 1.05 s + confusion 1.44 s) ÷ 15.0 s scored speech": the marks show seconds,
// the DER divides their sum by the scored reference speech (silence and the collar around every boundary do not count).
function derParts(s) {
  return `DER ${pct(s.der)} = (missed ${s.miss_s.toFixed(2)} s + false alarm ${s.fa_s.toFixed(2)} s + confusion ${
    s.conf_s.toFixed(2)} s) ÷ ${s.ref_speech.toFixed(1)} s scored speech`;
}

// Overlap: where 2+ speakers talk at once in the reference (top) and in the prediction (bottom), with the overlap the
// prediction misses (red) or adds (amber) when `diff` (refSegs, hypSegs) => { miss, extra } is given. All speakers
// count, hidden ones too. Clicking a part plays it (see Timeline._errorSegAt).
function overlapRows(diff) {
  const res = result, v = view;
  const ref = res.ref ? overlapRegions(v.refSegs) : null;
  const secs = (segs) => `${segs.reduce((acc, [s, e]) => acc + e - s, 0).toFixed(1)} s`;
  const tracks = [{ colorVar: '--highlight', segs: v.hypOverlap, label: secs(v.hypOverlap), tag: 'pred', bgVar: '--track-pred' }];
  if (ref) tracks.unshift({ colorVar: '--highlight', segs: ref, label: secs(ref), tag: 'ref', bgVar: '--track-ref' });
  return [
    { kind: 'header', label: `Overlap · 2+ speakers at once${ref ? ` · REF: real (top) · PRED: model (bottom)${diff && $('#show-diff').checked ? ' · red: missed overlap · amber: extra overlap' : ''}` : ' · prediction'}` },
    { kind: 'stack', label: 'overlap', h: ref ? 36 : 20, tracks, diff: ref && diff ? diff(ref, v.hypOverlap) : null },
  ];
}

// One toggle per speaker above the timeline; a hidden speaker's lanes are left out of every section.
function renderSpeakerFilter(chips) {
  const box = $('#speaker-filter');
  box.hidden = chips.length < 2;
  const keys = JSON.stringify(chips.map((c) => c.key));
  if (box.dataset.keys === keys) {  // same speakers: update in place, so a double-click lands on the same button
    for (const b of box.querySelectorAll('[data-speaker]')) b.classList.toggle('on', !hiddenSpeakers.has(b.dataset.speaker));
    return;
  }
  box.innerHTML = `<span>Speakers</span>${chips.map((c) => `
    <button class="btn small chip${hiddenSpeakers.has(c.key) ? '' : ' on'}" data-speaker="${esc(c.key)}"
      title="Show / hide this speaker (click) · show only this speaker (double-click)"><i style="background:${c.color}"></i>${esc(c.label)}</button>`).join('')}
    <button class="btn small" data-speaker-all title="Show every speaker">all</button>`;
  box.dataset.keys = keys;
}

function setHiddenSpeakers(keys) {
  hiddenSpeakers.clear();
  for (const k of keys) hiddenSpeakers.add(k);
  store.set('hiddenSpeakers', [...hiddenSpeakers]);
  renderTimeline(true);
}

// Merge [[start, end], ...] into sorted, non-overlapping segments.
function mergeSegs(segs) {
  const out = [];
  for (const [s, e] of [...segs].sort((x, y) => x[0] - y[0])) {
    if (out.length && s <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
    else out.push([s, e]);
  }
  return out;
}

// Parts of `a` not covered by `b`; both sorted and non-overlapping.
function subtractSegs(a, b) {
  const out = [];
  let j = 0;
  for (let [s, e] of a) {
    while (j < b.length && b[j][1] <= s) j++;
    for (let k = j; k < b.length && b[k][0] < e; k++) {
      if (b[k][0] > s) out.push([s, b[k][0]]);
      s = Math.max(s, b[k][1]);
      if (s >= e) break;
    }
    if (s < e) out.push([s, e]);
  }
  return out;
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
  const regions = s.regions.map((g, i) => ({ ...g, i, total: g.miss + g.fa + g.conf })).filter(hasErrorType);
  $('#error-count').textContent = regions.length < s.regions.length ? `${regions.length} / ${s.regions.length}` : `${regions.length}`;
  if (!regions.length) {
    box.innerHTML = '<p class="empty">No errors of the selected types.</p>';
    return;
  }
  if (errorSort === 'size') regions.sort((a, b) => b.total - a.total);
  const shown = regions.slice(0, 500);
  const names = (list, ids, colors) => ids.map((k) => `${dot(colors[k])}${esc(list[k])}`).join(' ');
  const tag = (cls, label, x) => (x > 0 ? `<span class="tag ${cls}">${label} ${x.toFixed(2)}</span>` : '');
  box.innerHTML = `
    <table class="table">
      <thead><tr><th></th><th></th><th>start</th><th>dur (s)</th><th title="speaker-seconds">error</th><th>reference</th><th>prediction</th></tr></thead>
      <tbody>${shown.map((g) => `
        <tr data-i="${g.i}">
          <td class="nowrap"><button class="btn icon play" title="Play this region">▶</button><button class="btn icon loop" title="Play this region over and over (A–B loop)">↻</button></td>
          <td>${saveButton(g)}</td>
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
        <td class="num">${pct(sp.jer, 1)}</td>
      </tr>`).join('');
    box.innerHTML = `
      <table class="table">
        <thead><tr><th>reference</th><th>prediction</th><th>ref (s)</th><th>pred (s)</th>
          <th title="Scored time of the reference speaker that its mapped prediction also covers">recall</th>
          <th title="Scored time of the prediction during which its mapped reference speaker talks">precision</th>
          <th title="1 − intersection over union with its best-matching prediction (JER mapping)">JER</th></tr></thead>
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

// A predicted speaker mapped to a reference speaker is shown / hidden (and focused) together with it.
function hypKey(h) {
  const map = result.score?.hyp_to_ref || [];
  return map[h] != null ? `ref:${result.ref.names[map[h]]}` : `hyp:${result.hyp.names[h]}`;
}

// Which speakers cause the error at `t`: reference speakers whose matched prediction is silent and predicted speakers
// whose matched reference is silent; paired up they are confusion, the rest missed / false alarm (as DER counts them).
function errorSpeakers(t) {
  const res = result, v = view, map = res.score.hyp_to_ref;
  const on = (segs) => segs.some(([s, e]) => t >= s && t < e);
  const hypOn = res.hyp.names.map((_, h) => on(v.hypSegs[h]));
  const refOn = res.ref.names.map((_, r) => on(v.refSegs[r]));
  const lost = refOn.flatMap((a, r) => (a && !hypOn[map.indexOf(r)] ? [r] : []));
  const stray = hypOn.flatMap((a, h) => (a && !(map[h] != null && refOn[map[h]]) ? [h] : []));
  const n = Math.min(lost.length, stray.length);
  return { conf: lost.slice(0, n).map((r, i) => [r, stray[i]]), miss: lost.slice(n), fa: stray.slice(n) };
}

// While the pointer is on an error mark: the speaker keys (see `spk` on the rows) behind the error at `t`, or null.
function errorFocus(t) {
  if (!result?.score) return null;
  const { conf, miss, fa } = errorSpeakers(t);
  const keys = new Set([...miss.map((r) => `ref:${result.ref.names[r]}`), ...fa.map(hypKey),
                        ...conf.flatMap(([r, h]) => [`ref:${result.ref.names[r]}`, hypKey(h)])]);
  return keys.size ? keys : null;
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
    // the model's probability that this speaker talks here, in percent
    return `${dot(v.hypColor[h])}${esc(name)}${p != null ? ` <span class="muted" title="model probability">${Math.round(p * 100)}%</span>` : ''}`;
  }).filter(Boolean);
  lines.push(`<div><span class="k">pred</span>${pred.join(' ') || '<i>silence</i>'}</div>`);
  if (res.score) {
    const L = res.score.lanes;
    const who = errorSpeakers(t), map = res.score.hyp_to_ref;
    const ref = (r) => `${dot(v.refColor[r])}${esc(res.ref.names[r])}`;
    const hyp = (h) => `${dot(v.hypColor[h])}${esc(res.hyp.names[h])}`;
    const silent = (name) => ` <span class="muted">(${esc(name)} silent)</span>`;
    // e.g. "missed spk1 (speaker_0 silent)", "confusion spk0 → speaker_1", "false alarm speaker_2"
    const errors = [
      ['missed', L.miss, who.miss.map((r) => ref(r) + (map.includes(r) ? silent(res.hyp.names[map.indexOf(r)]) : ''))],
      ['false alarm', L.fa, who.fa.map((h) => hyp(h) + (map[h] != null ? silent(res.ref.names[map[h]]) : ''))],
      ['confusion', L.conf, who.conf.map(([r, h]) => `${ref(r)} → ${hyp(h)}`)],
    ].filter(([, segs]) => on(segs));
    errors.forEach(([name, , spk], i) => lines.push(
      `<div><span class="k">${i ? '' : 'error'}</span><span class="err-type">${name}</span> ${spk.join(' ')}</div>`));
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

// Does a region contain an error of one of the selected types (miss / fa / conf)?
function hasErrorType(g) {
  return [...errorTypes].some((type) => g[type] > 0);
}

function jumpError(dir) {
  const regions = result?.score?.regions.filter(hasErrorType);
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

// Play the whole file from 0:00.
function playFromStart() {
  if (!player.src) return;
  timeline.seekBy(-player.currentTime);  // to 0:00, scrolling a zoomed-in view back to the start
  player.play();
}

// A–B loop: the first press sets A at the playback position, the second sets B and starts looping, the third clears it.
function toggleLoop() {
  const loop = timeline.loop, t = player.currentTime;
  if (!player.src) return;
  if (!loop) return timeline.setLoop([t]);
  if (loop.length === 1) {
    if (Math.abs(t - loop[0]) < 0.05) return setStatus('Move to where B should be (play or click the timeline), then press A–B again');
    timeline.setLoop([Math.min(loop[0], t), Math.max(loop[0], t)]);
    timeline.seek(timeline.loop[0]);
    player.play();
    return;
  }
  timeline.setLoop(null);
}

// Loop one error region, with a little audio around it.
function loopRegion(g) {
  focusRegion(g, false);
  timeline.setLoop([Math.max(0, g.start - 0.3), Math.min(result.duration, g.end + 0.3)]);
  timeline.seek(timeline.loop[0]);
  player.play();
}

function showLoop(loop) {
  const btn = $('#loop-btn');
  btn.classList.toggle('on', Boolean(loop));
  btn.textContent = !loop ? 'A–B' : loop.length === 1 ? `A ${fmtTime(loop[0])} · set B` : `↻ ${fmtTime(loop[0])} – ${fmtTime(loop[1])} ✕`;
  btn.title = !loop ? 'A–B loop (l): set A here, then B; alt+drag on the timeline also selects a loop'
    : loop.length === 1 ? 'Set B at the playback position and start looping (l)' : 'Stop looping (l)';
  $('#loop-mp3').hidden = loop?.length !== 2;
}

// Save the A–B section of the open file as MP3 (cut and encoded by the server).
function saveLoopMp3() {
  const loop = timeline.loop;
  if (loop?.length !== 2 || !result) return;
  const a = document.createElement('a');
  a.href = `/api/clip?path=${encodeURIComponent(result.audio)}&start=${loop[0].toFixed(3)}&end=${loop[1].toFixed(3)}`;
  a.click();
  setStatus(`Saving ${fmtTime(loop[0])} – ${fmtTime(loop[1])} as MP3`);
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
  if (mode === 'folder') {
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
  if (batch && batch.fields.audio === current) return stepFolder(step);
  if (mode === 'folder') return run();
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

// Reject when `promise` takes longer than `ms` (a file the browser cannot read otherwise spins forever).
function withTimeout(promise, ms, message, onTimeout) {
  let timer;
  const expired = new Promise((_, reject) => {
    // reject before onTimeout, so the race settles with `message` and not with the abort error onTimeout causes
    timer = setTimeout(() => { reject(new Error(message)); onTimeout?.(); }, ms);
  });
  return Promise.race([promise, expired]).finally(() => clearTimeout(timer));
}

async function uploadFile(field, file) {
  if (field !== 'checkpoint') setMode('file');  // uploads are single files
  setStatus(`Uploading ${file.name}…`, true);
  try {
    // read the file first, so a file the browser cannot read gives its own error instead of a stalled request
    const unreadable = `The browser could not read ${file.name}. If it is on the server, use Browse or type its path instead.`;
    const bytes = await withTimeout(file.arrayBuffer().catch((e) => { throw new Error(`${unreadable} (${e.message})`); }),
      30000, unreadable);
    const form = new FormData();
    form.append('file', new Blob([bytes]), file.name);
    form.append('field', field);
    if (field === 'audio') form.append('sample_rate', $('#upload-rate').value);
    const data = await postWithProgress('/api/upload', form, field, file.name,
      120000 + bytes.byteLength / 500);  // 2 min plus 2 s per MB: the server also converts non-WAV audio
    setVal(field, data.path);
    if (field === 'audio') await audioChanged();
    save();
    setStatus(`Uploaded ${file.name}`);
  } catch (e) {
    setStatus('');
    showError(e);
  }
}

// POST `form` with a thin progress line under the field's input row (fetch cannot report upload progress).
// Once every byte is sent the line pulses until the server answers (saving, converting to WAV).
function postWithProgress(url, form, field, name, timeoutMs) {
  const row = $(`#${field}`).closest('.row');
  const bar = document.createElement('div');
  bar.className = 'upload-bar';
  bar.innerHTML = '<div></div>';
  row.after(bar);
  const fill = bar.firstChild;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.timeout = timeoutMs;
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      fill.style.width = `${pct}%`;
      setStatus(`Uploading ${name}… ${pct}%`, true);
    };
    xhr.upload.onload = () => {
      bar.classList.add('busy');
      setStatus(`Processing ${name} on the server…`, true);
    };
    xhr.onload = () => {
      let data;
      try { data = JSON.parse(xhr.responseText); } catch { data = { detail: `${xhr.status} ${xhr.statusText}` }; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.detail || xhr.statusText));
    };
    xhr.onerror = () => reject(new Error(`Uploading ${name} failed: the connection to the server was lost.`));
    xhr.ontimeout = () => reject(new Error(`Uploading ${name} timed out: the server did not answer. Is it still running?`));
    xhr.send(form);
  }).finally(() => bar.remove());
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
  for (const btn of document.querySelectorAll('[data-clear]')) {
    btn.addEventListener('click', () => {
      const field = btn.dataset.clear;
      setVal(field, '');  // lastAudio stays, so the next audio still replaces RTTMs paired with this one
      save();
      $(`#${field}`).focus();
    });
  }
  const fileInput = $('#file-input');
  for (const btn of document.querySelectorAll('[data-upload]')) {
    btn.addEventListener('click', () => {
      fileInput.dataset.field = btn.dataset.upload;
      fileInput.accept = { audio: 'audio/*,.wav,.flac,.mp3,.ogg,.opus,.m4a', checkpoint: '.nemo,.ckpt' }[btn.dataset.upload] || '.rttm';
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
    for (const file of files) {
      if (/\.ya?ml$/i.test(file.name)) await loadYaml(file);
      else await uploadFile(fieldForFile(file.name, droppedOn), file);
    }
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
  for (const b of document.querySelectorAll('[data-mode]')) b.addEventListener('click', () => setMode(b.dataset.mode));
  $('#load-btn').addEventListener('click', loadModel);
  $('#run-btn').addEventListener('click', async () => { await audioChanged(); run(); });
  $('#prev-file').addEventListener('click', () => stepAudio(-1));
  $('#next-file').addEventListener('click', () => stepAudio(1));
  $('#reset-settings').addEventListener('click', () => { setSettings({}); save(); scheduleRescore(); });
  $('#copy-yaml').addEventListener('click', copyYaml);
  $('#load-yaml').addEventListener('click', () => $('#yaml-input').click());
  $('#yaml-input').addEventListener('change', (e) => {
    if (e.target.files[0]) loadYaml(e.target.files[0]);
    e.target.value = '';
  });
  $('#yaml-history').addEventListener('change', (e) => {
    const i = e.target.value;
    e.target.value = '';
    if (i === 'clear') { store.set('yamlHistory', []); renderYamlHistory(); return; }
    const entry = store.get('yamlHistory', [])[Number(i)];
    if (!entry) return;
    setSettings({ ...getSettings(), ...entry.values });
    save();
    scheduleRescore();
    setStatus(`Applied ${entry.name}`);
  });

  $('#folder-stop').addEventListener('click', stopFolder);
  $('#folder-all').addEventListener('click', () => { batch.checkpoint = val('checkpoint'); scoreFolder(++batchSeq, 'missing'); });
  $('#folder-first').addEventListener('click', () => openFolderFile(setRows(batch)[0]));
  $('#folder-set').addEventListener('change', (e) => { chooseSet(e.target.value); e.target.blur(); });
  for (const btn of document.querySelectorAll('[data-setview]')) {
    btn.addEventListener('click', () => { setView = btn.dataset.setview; store.set('setView', setView); renderSets(); });
  }
  $('#set-analysis').addEventListener('toggle', (e) => store.set('setsOpen', e.target.open));
  $('#set-run-all').addEventListener('click', () => { batch.checkpoint = val('checkpoint'); scoreFolder(++batchSeq, 'missing', true); });
  for (const id of ['#set-chart', '#set-table']) {
    $(id).addEventListener('click', (e) => {
      const el = e.target.closest('[data-set]');
      if (!el) return;
      const set = el.dataset.set === batch.set ? '' : el.dataset.set;  // the chosen set again: back to all sets
      $('#folder-set').value = set;
      chooseSet(set);
    });
  }
  $('#set-chart').addEventListener('mousemove', setTip);
  $('#set-chart').addEventListener('mouseleave', () => { $('#set-tip').hidden = true; });
  $('#folder-prev').addEventListener('click', () => stepFolder(-1));
  $('#folder-next').addEventListener('click', () => stepFolder(1));
  $('#folder-rerun').addEventListener('click', rerunFolder);
  $('#folder-csv').addEventListener('click', () => downloadFolder('csv'));
  $('#folder-rttm').addEventListener('click', () => downloadFolder('rttm'));
  $('#folder-table').addEventListener('click', (e) => {
    const th = e.target.closest('th[data-sort]');
    if (th) {
      const col = th.dataset.sort;  // first click: names A→Z, numbers largest (worst) first; again: flip
      return setFolderSort(col, col === folderSort.col ? !folderSort.desc : col !== 'file' && col !== 'set');
    }
    const tr = e.target.closest('tr[data-f]');
    if (tr) openFolderFile(Number(tr.dataset.f), true);
  });
  for (const btn of document.querySelectorAll('[data-fsort]')) {
    btn.addEventListener('click', () => (btn.dataset.fsort === 'der' ? setFolderSort('DER', true) : setFolderSort('file', false)));
  }

  // file browser dialog
  $('#br-list').addEventListener('click', (e) => {
    const li = e.target.closest('li');
    if (li?.dataset.dir) listDir(li.dataset.dir);
    else if (li?.dataset.file) closePicker(picker.folders ? $('#br-path').value : li.dataset.file, picker.folders);
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
    const folders = mode === 'folder' && field !== 'checkpoint';  // a file then means its folder
    picker = { field, kind: FIELDS[field].kind, resolve, folders };
    $('#br-title').textContent = folders ? `Select ${{ audio: 'the audio folder (or a dataset folder holding wav/ + rttm/)', ref: 'the reference RTTM folder', hyp: 'the prediction RTTM folder' }[field]}`
      : `Select ${FIELDS[field].title}`;
    $('#br-use-dir').hidden = !folders;
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

// Settings from a YAML file: every `key: value` line naming a setting counts, at any nesting level, so both a
// NeMo postprocessing_yaml (keys under `parameters:`) and a flat file work. Settings it does not name keep their value.
async function loadYaml(file) {
  try {
    const text = await file.text();
    const found = {};
    for (const line of text.split(/\r?\n/)) {
      const m = line.replace(/\s+#.*$/, '').match(/^\s*([A-Za-z_]+)\s*:\s*(\S.*?)\s*$/);
      const setting = m && SETTINGS.find((s) => s.key === m[1]);
      if (!setting) continue;
      const raw = m[2].replace(/^['"]|['"]$/g, '');
      if (typeof setting.def === 'boolean') found[setting.key] = /^(true|yes|on|1)$/i.test(raw);
      else if (Number.isFinite(parseFloat(raw))) found[setting.key] = parseFloat(raw);
    }
    const keys = Object.keys(found);
    if (!keys.length) throw new Error(`${file.name} has none of the settings (${SETTINGS.map((s) => s.key).join(', ')}).`);
    setSettings({ ...getSettings(), ...found });
    save();
    scheduleRescore();
    rememberYaml(file.name, found);
    setStatus(`Loaded ${keys.join(', ')} from ${file.name}`);
  } catch (e) {
    showError(e);
  }
}

// The last few loaded YAMLs, newest first, so one can be re-applied without uploading it again.
const YAML_HISTORY_SIZE = 15;

function rememberYaml(name, values) {
  const same = (e) => e.name === name && JSON.stringify(e.values) === JSON.stringify(values);
  const history = store.get('yamlHistory', []).filter((e) => !same(e));
  history.unshift({ name, values, at: Date.now() });
  store.set('yamlHistory', history.slice(0, YAML_HISTORY_SIZE));
  renderYamlHistory();
}

function renderYamlHistory() {
  const select = $('#yaml-history');
  const history = store.get('yamlHistory', []);
  select.hidden = !history.length;
  select.innerHTML = `<option value="" selected disabled>Previous YAMLs (${history.length})</option>`
    + history.map((e, i) => {
      const detail = Object.entries(e.values).map(([k, v]) => `${k}: ${v}`).join('\n');
      const when = new Date(e.at).toLocaleString();
      return `<option value="${i}" title="${esc(`${when}\n${detail}`)}">${esc(e.name)} · ${esc(when)}</option>`;
    }).join('')
    + '<option value="clear">Clear history</option>';
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
  store.set(`paths.${mode}`, Object.fromEntries(['audio', 'ref', 'hyp'].map((k) => [k, val(k)])));
  store.set('settings', getSettings());
}

function restore() {
  const fields = store.get('fields', {});
  for (const k of Object.keys(FIELDS)) setVal(k, fields[k] || '');
  setSettings(store.get('settings', {}));
  mode = store.get('mode', isFolder(val('audio')) ? 'folder' : 'file');  // before the switch: what the field holds
  setMode(mode);
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

// ---- saved errors -------------------------------------------------------------------------

const CLIP_PAD = 1;  // seconds of audio saved before and after an error

function savedFor(g) {
  return savedItems.find((x) => x.audio === result.audio && Math.abs(x.start - g.start) < 0.005 && Math.abs(x.end - g.end) < 0.005);
}

function saveButton(g) {
  const saved = savedFor(g);
  return `<button class="btn icon save${saved ? ' on' : ''}" title="${saved ? 'Saved · click to remove' : 'Save this error with its audio'}">${saved ? '★' : '☆'}</button>`;
}

async function loadSaved() {
  try {
    const d = await api('/api/saved');
    savedItems = d.items;
    $('#saved-dir').textContent = d.dir;
    $('#saved-dir').title = 'Clips, JSON and saved.csv are kept here on the server';
  } catch { /* server not reachable yet */ }
  renderSaved();
}

// Save an error of the open file (or remove it when it is saved already).
async function toggleSave(g) {
  const saved = savedFor(g);
  try {
    if (saved) {
      await api(`/api/saved/${encodeURIComponent(saved.id)}`, undefined, 'DELETE');
      savedItems = savedItems.filter((x) => x !== saved);
      setStatus('Removed from saved errors');
    } else {
      const res = result, from = g.start - CLIP_PAD, to = g.end + CLIP_PAD;
      const near = (side) => (res[side]?.segments || []).filter(([s, e]) => e > from && s < to)
        .map(([s, e, k]) => [s, e, res[side].names[k]]);
      const item = await api('/api/saved', {
        audio: res.audio, start: g.start, end: g.end, pad: CLIP_PAD, miss: g.miss, fa: g.fa, conf: g.conf,
        group: batch?.rows.find((r) => r.audio === res.audio)?.group || '',
        ref_speakers: g.ref.map((k) => res.ref.names[k]), hyp_speakers: g.hyp.map((k) => res.hyp.names[k]),
        ref_rttm: shown?.ref || '', hyp_rttm: shown?.hyp || '', checkpoint: res.model?.path || '',
        settings: getSettings(), file_der: res.score?.der ?? null, segments: { ref: near('ref'), hyp: near('hyp') },
      });
      savedItems.unshift(item);
      setStatus(`Saved ${item.clip}`);
    }
  } catch (e) { showError(e); }
  renderErrors();
  renderSaved();
}

function renderSaved() {
  const n = savedItems.length;
  $('#saved').hidden = !n;
  $('#saved-link').hidden = !n;
  $('#saved-link').textContent = `★ ${n} saved`;
  $('#saved-count').textContent = n ? `${n}` : '';
  const terms = $('#saved-search').value.toLowerCase().split(/\s+/).filter(Boolean);
  const items = savedItems.filter((x) => {
    const text = [x.group, x.file_id, ...x.types, ...x.ref_speakers, ...x.hyp_speakers, x.note].join(' ').toLowerCase();
    return terms.every((t) => text.includes(t));
  });
  if (n && items.length < n) $('#saved-count').textContent = `${items.length} / ${n}`;
  const tag = (cls, label, x) => (x > 0 ? `<span class="tag ${cls}">${label} ${x.toFixed(2)}</span>` : '');
  $('#saved-table').innerHTML = !items.length ? '<p class="empty">No saved error matches.</p>' : `
    <table class="table">
      <thead><tr><th></th><th>file</th><th>start</th><th>dur (s)</th><th>error</th><th>reference</th><th>prediction</th>
        <th>note</th><th>saved</th><th></th></tr></thead>
      <tbody>${items.map((x) => `
        <tr data-id="${esc(x.id)}">
          <td><button class="btn icon clip" title="Play the saved clip">▶</button></td>
          <td title="${esc(x.audio)}">${x.group ? `<span class="muted">${esc(x.group)} · </span>` : ''}${esc(x.file_id)}</td>
          <td class="num">${fmtTime(x.start)}</td>
          <td class="num">${x.duration.toFixed(2)}</td>
          <td>${tag('miss', 'miss', x.miss)}${tag('fa', 'FA', x.fa)}${tag('conf', 'conf', x.conf)}</td>
          <td>${esc(x.ref_speakers.join(' '))}</td>
          <td>${esc(x.hyp_speakers.join(' '))}</td>
          <td><input class="note" value="${esc(x.note)}" placeholder="add a note" spellcheck="false"></td>
          <td class="muted" title="${esc(x.checkpoint)}">${esc(x.saved_at.replace('T', ' ').slice(0, 16))}</td>
          <td><button class="btn small open" title="Open this file at the error">Open</button>
            <a class="btn small" href="/api/saved/${encodeURIComponent(x.id)}/clip" download="${esc(x.clip)}" title="Download the clip">WAV</a>
            <button class="btn small remove" title="Delete the clip and its data">✕</button></td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

// Go back to a saved error: open its file (in the folder when it is part of it) and zoom to the error.
async function openSaved(x) {
  const i = mode === 'folder' && batch ? batch.rows.findIndex((r) => r.audio === x.audio) : -1;
  if (i >= 0) await openFolderFile(i);
  else await runFile({ audio: x.audio, ref: x.ref_rttm, hyp: x.hyp_rttm });
  if (result?.audio !== x.audio) return;
  const g = result.score?.regions.find((r) => Math.abs(r.start - x.start) < 0.005 && Math.abs(r.end - x.end) < 0.005);
  if (g) focusRegion(g, false);
  else { timeline.zoomTo(x.start - CLIP_PAD, x.end + CLIP_PAD); timeline.seek(x.start); }  // other settings: not an error now
  $('#results').scrollIntoView({ behavior: 'smooth' });
}

function bindSaved() {
  $('#saved-link').addEventListener('click', () => $('#saved').scrollIntoView({ behavior: 'smooth' }));
  $('#saved-search').addEventListener('input', renderSaved);
  $('#saved-zip').addEventListener('click', () => { window.location.href = '/api/saved-export'; });
  const clipStopped = () => { for (const b of document.querySelectorAll('#saved-table .clip')) b.textContent = '▶'; };
  clipPlayer.addEventListener('ended', clipStopped);
  player.addEventListener('play', () => { clipPlayer.pause(); clipStopped(); });
  $('#saved-table').addEventListener('click', async (e) => {
    const tr = e.target.closest('tr[data-id]');
    const x = tr && savedItems.find((it) => it.id === tr.dataset.id);
    if (!x) return;
    if (e.target.closest('.clip')) {
      const url = `/api/saved/${encodeURIComponent(x.id)}/clip`;
      const playing = !clipPlayer.paused && clipPlayer.dataset.id === x.id;
      clipPlayer.pause();
      for (const b of document.querySelectorAll('#saved-table .clip')) b.textContent = '▶';
      if (playing) return;
      player.pause();
      Object.assign(clipPlayer, { src: url }).dataset.id = x.id;
      clipPlayer.play();
      e.target.closest('.clip').textContent = '❚❚';
    } else if (e.target.closest('.open')) {
      openSaved(x);
    } else if (e.target.closest('.remove')) {
      if (!confirm(`Delete the saved error ${x.clip}?`)) return;
      try {
        await api(`/api/saved/${encodeURIComponent(x.id)}`, undefined, 'DELETE');
        savedItems = savedItems.filter((it) => it !== x);
        renderSaved();
        if (result) renderErrors();
      } catch (err) { showError(err); }
    }
  });
  $('#saved-table').addEventListener('change', async (e) => {
    const tr = e.target.closest('tr[data-id]');
    const x = tr && e.target.classList.contains('note') && savedItems.find((it) => it.id === tr.dataset.id);
    if (!x) return;
    try {
      Object.assign(x, await api(`/api/saved/${encodeURIComponent(x.id)}`, { note: e.target.value }, 'PATCH'));
      setStatus('Note saved');
    } catch (err) { showError(err); }
  });
}

// ---- results toolbar & keys ---------------------------------------------------------------

function bindResults() {
  $('#play-btn').addEventListener('click', togglePlay);
  $('#restart-btn').addEventListener('click', playFromStart);
  $('#loop-btn').addEventListener('click', toggleLoop);
  $('#loop-mp3').addEventListener('click', saveLoopMp3);
  showLoop(null);
  player.addEventListener('play', () => { $('#play-btn').textContent = '❚❚'; });
  player.addEventListener('pause', () => { $('#play-btn').textContent = '▶'; });
  player.addEventListener('timeupdate', updateTime);
  player.addEventListener('seeked', updateTime);
  $('#speed').addEventListener('change', (e) => { player.playbackRate = parseFloat(e.target.value); });
  $('#volume').addEventListener('input', (e) => { player.volume = parseFloat(e.target.value); player.muted = false; });
  $('#volume').addEventListener('change', (e) => e.target.blur());  // give the arrow keys back to seeking
  $('#mute-btn').addEventListener('click', toggleMute);
  player.addEventListener('volumechange', showVolume);
  const volume = store.get('volume', { level: 1, muted: false });
  player.volume = volume.level;
  player.muted = volume.muted;
  showVolume();
  $('#zoom-in').addEventListener('click', () => timeline.zoom(0.5));
  $('#zoom-out').addEventListener('click', () => timeline.zoom(2));
  $('#zoom-fit').addEventListener('click', () => timeline.fit());
  $('#fullscreen-btn').addEventListener('click', toggleFullscreen);
  document.addEventListener('fullscreenchange', () => {
    $('#fullscreen-btn').classList.toggle('on', Boolean(document.fullscreenElement));
  });
  $('#prev-err').addEventListener('click', () => jumpError(-1));
  $('#next-err').addEventListener('click', () => jumpError(1));
  $('#show-all').addEventListener('change', () => renderTimeline(true));
  $('#show-diff').addEventListener('change', (e) => { store.set('showDiff', e.target.checked); renderTimeline(true); });
  $('#show-diff').checked = store.get('showDiff', true);
  $('#show-diar').addEventListener('change', (e) => { store.set('showDiar', e.target.checked); renderTimeline(true); });
  $('#show-diar').checked = store.get('showDiar', true);
  $('#lane-size').value = laneSize;
  $('#lane-size').addEventListener('change', (e) => { laneSize = e.target.value; store.set('laneSize', laneSize); renderTimeline(true); });
  for (const b of document.querySelectorAll('[data-view]')) {
    b.addEventListener('click', () => { viewMode = b.dataset.view; store.set('viewMode', viewMode); renderTimeline(true); });
  }
  $('#upload-rate').value = store.get('uploadRate', '0');
  $('#upload-rate').addEventListener('change', (e) => store.set('uploadRate', e.target.value));
  $('#speaker-filter').addEventListener('click', (e) => {
    const keys = JSON.parse($('#speaker-filter').dataset.keys || '[]');
    if (e.target.closest('[data-speaker-all]')) return setHiddenSpeakers([...hiddenSpeakers].filter((k) => !keys.includes(k)));
    const chip = e.target.closest('[data-speaker]');
    if (!chip) return;
    const key = chip.dataset.speaker;
    if (e.detail === 2) return setHiddenSpeakers([...hiddenSpeakers, ...keys.filter((k) => k !== key)].filter((k) => k !== key));  // solo
    setHiddenSpeakers(hiddenSpeakers.has(key) ? [...hiddenSpeakers].filter((k) => k !== key) : [...hiddenSpeakers, key]);
  });
  $('#dl-rttm').addEventListener('click', () => download(`${result.file_id}.rttm`, result.rttm, 'text/plain'));
  $('#dl-json').addEventListener('click', () => download(`${result.file_id}.json`, JSON.stringify(result), 'application/json'));
  for (const btn of document.querySelectorAll('[data-sort]')) {
    btn.addEventListener('click', () => {
      errorSort = btn.dataset.sort;
      for (const b of document.querySelectorAll('[data-sort]')) b.classList.toggle('on', b === btn);
      renderErrors();
    });
  }
  for (const btn of document.querySelectorAll('[data-etype]')) {
    btn.classList.toggle('on', errorTypes.has(btn.dataset.etype));
    btn.addEventListener('click', () => {
      const type = btn.dataset.etype;
      if (errorTypes.has(type)) errorTypes.delete(type); else errorTypes.add(type);
      btn.classList.toggle('on', errorTypes.has(type));
      store.set('errorTypes', [...errorTypes]);
      renderErrors();
    });
  }
  $('#errors').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-i]');
    if (tr && e.target.closest('.save')) return toggleSave(result.score.regions[Number(tr.dataset.i)]);
    if (tr && e.target.closest('.loop')) return loopRegion(result.score.regions[Number(tr.dataset.i)]);
    if (tr) focusRegion(result.score.regions[Number(tr.dataset.i)], Boolean(e.target.closest('.play')));
  });
}

// The results (metrics, timeline, error list, speakers table) on the whole screen; Esc or f leaves.
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else $('#results').requestFullscreen().catch((e) => showError(e));
}

function toggleMute() {
  if (!player.muted && player.volume === 0) player.volume = 1;  // unmuting at volume 0 would stay silent
  else player.muted = !player.muted;
}

// Reflect the player's volume in the slider and button, and remember it across reloads.
function showVolume() {
  const silent = player.muted || player.volume === 0;
  $('#volume').value = player.muted ? 0 : player.volume;
  $('#mute-btn').textContent = silent ? '🔇' : player.volume < 0.5 ? '🔉' : '🔊';
  store.set('volume', { level: player.volume, muted: player.muted });
}

const SEEK_STEP = 5;  // seconds the arrow keys jump

function bindKeys() {
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest('input, select, textarea, dialog')) return;
    if (e.key === '[' || e.key === ']') { e.preventDefault(); stepAudio(e.key === ']' ? 1 : -1); return; }
    if (!result) return;
    const actions = {
      ' ': togglePlay,
      l: toggleLoop,
      Escape: () => timeline.unpin(),
      Home: playFromStart,
      ArrowLeft: () => (e.shiftKey ? timeline.pan(-0.25) : timeline.seekBy(-SEEK_STEP)),
      ArrowRight: () => (e.shiftKey ? timeline.pan(0.25) : timeline.seekBy(SEEK_STEP)),
      '+': () => timeline.zoom(0.5),
      '=': () => timeline.zoom(0.5),
      '-': () => timeline.zoom(2),
      0: () => timeline.fit(),
      n: () => jumpError(1),
      p: () => jumpError(-1),
      m: toggleMute,
      f: toggleFullscreen,
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
