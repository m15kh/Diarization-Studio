# Diarization Inspector

A small web UI to run a NeMo Sortformer diarization checkpoint on an audio file and see exactly where it is wrong.
FastAPI backend (Python) + plain HTML/JS frontend. No database, no build step.

## Run

Use the Python environment where NeMo is installed (the one you train with):

```bash
pip install -r requirements.txt
python3 backend/server.py                http://127.0.0.1:7860
```

| option | |
|---|---|
| `--checkpoint PATH` | load a `.nemo` / `.ckpt` at startup |
| `--audio PATH` | audio file to open (`.wav`, `.flac`, ...), or a folder to run every audio file in it |
| `--rttm PATH` | reference RTTM file or folder (also `--ref-rttm`). Default: found next to the audio or in `../rttm/` |
| `--hyp-rttm PATH` | prediction RTTM file or folder to inspect instead of running the checkpoint |
| `--device auto` | `auto`, `cpu`, `cuda`, `cuda:1`, ... |
| `--port 7860` | |
| `--host 127.0.0.1` | `0.0.0.0` makes it reachable from other machines; the UI can browse the server's files, so only do that on a trusted network |

Server on a remote GPU machine: run it there and tunnel the port, then open http://localhost:7860 locally:

```bash
ssh -L 7860:localhost:7860 user@gpu-server
```

### Set the inputs from the terminal

Everything can also be chosen in the UI. The paths given on the command line are filled into the UI fields when the page opens:

```bash
# checkpoint + audio: the model loads at startup and the file runs as soon as the page opens
python3 backend/server.py --checkpoint /models/sortformer_8spk.nemo --audio /data/test/wav/meeting_01.wav

# with an explicit reference RTTM
python3 backend/server.py --checkpoint /models/sortformer_8spk.nemo \
    --audio /data/test/wav/meeting_01.wav --rttm /data/test/rttm/meeting_01.rttm

# no model: inspect an existing prediction (pyannote, DiariZen, an older run, ...)
python3 backend/server.py --audio /data/test/wav/meeting_01.wav --hyp-rttm /runs/pyannote/meeting_01.rttm

# a whole folder: every wav runs, scored against the RTTM with the same name in the RTTM folder
python3 backend/server.py --checkpoint /models/sortformer_8spk.nemo --audio /data/test/wav --rttm /data/test/rttm

# a folder of existing predictions against the references
python3 backend/server.py --audio /data/test/wav --rttm /data/test/rttm --hyp-rttm /runs/pyannote

# only the checkpoint: the audio fields keep what you used last time
python3 backend/server.py --checkpoint /models/sortformer_8spk.nemo
```

- Paths can be relative to the folder you start the server in, or start with `~`. A path that does not exist stops the server with an error.
- Without `--rttm`, the reference is looked up the same way as in the UI: `meeting_01.rttm` next to the audio, in `../rttm/` or in `./rttm/`. The terminal shows which one was found.
- With `--audio`, the RTTM fields only hold what was given or found for that file. RTTMs left over from another file are cleared.
- The page opens with these paths once per server start. If you change a field in the UI and reload the page, your change stays. Restart the server to apply the terminal paths again.

## Use

1. **Checkpoint**: a path on the server (Browse lists `.nemo` / `.ckpt`). The model stays loaded. The number of speaker outputs (2, 4, 8, ...) is read from the checkpoint.
2. **Audio**: a server path, Browse, Upload, or drag and drop. ◀ ▶ step through the files of the folder and run them.
   A **folder** runs every audio file in it (Browse → **Use this folder**, see [Folders](#folders)).
3. **Reference RTTM** (optional): ground truth. It is found automatically next to the audio or in `../rttm/` (the `wav/` + `rttm/` layout of the datasets).
4. **Prediction RTTM** (optional): inspect any existing output (pyannote, DiariZen, older runs, ...) instead of running the model.

### Folders

Put a folder in **Audio** (and optionally in **Reference RTTM** / **Prediction RTTM**) and press Run, or give the folders in the terminal:

- Each audio file is paired with the RTTM of the same name: `wav/meeting_01.wav` ↔ `rttm/meeting_01.rttm`.
  An RTTM *file* in a folder field means its folder. With the reference field empty, it is looked up next to each audio file, in `../rttm/` or in `./rttm/`.
- The files run one after the other. The **Folder** table fills in as they finish: length, speakers (reference → predicted), DER, missed, false alarm, confusion and JER per file. Files without a reference show `–`.
- **Totals** at the top: DER over all files, computed from the summed error and speech time of every file (the way NeMo and pyannote report a total). It is not the average of the per-file DERs. Total JER is the mean over the reference speakers of all files, like dscore.
- **Sort by DER** puts the worst files first. **Click a file** to open it in the full inspector below (timeline, errors, speakers). ◀ ▶ then step through the folder.
- Changing post-processing, the collar or "ignore overlap" re-scores the whole folder. Model outputs are cached, so the totals update in seconds.
- **Run folder again** reruns every file, for example after loading another checkpoint. **Stop** stops after the current file.
- **CSV**: one row per file. **Download RTTM**: all predictions in one RTTM file.

The post-processing fields use NeMo's `PostProcessingParams` names. Changing them, the collar or "ignore overlap" re-scores instantly: the model output is cached, so only post-processing and scoring re-run. **Copy YAML** gives a `postprocessing_yaml` for `e2e_diarize_speech.py`. **Load YAML** (or dropping a `.yaml` file on the page) does the reverse: it reads the values from such a file (keys under `parameters:` or at the top level; `collar` and `ignore_overlap` too), and settings it does not name keep their value.

What you get:

- **Metrics**: DER = missed + false alarm + confusion, JER, speaker count, overlap, inference time / RTF.
- **Timeline**, in two views (all rows share one zoomable time axis):
  - **Simple** (default): the waveform, then one compact lane per speaker with the reference on top (yellow **REF** badge) and its matched prediction right below (cyan **PRED** badge, darker background) in the same color (with **differences** on: red where the prediction misses the reference, amber where it adds speech the reference does not have), then one error lane (missed / false alarm / confusion);
  - **Detailed**, with every speaker on its own lanes:
    - the waveform;
    - **diarization**: who speaks when, one lane per predicted speaker, with highlighted columns where 2+ speakers talk at once;
    - each reference speaker, with the predicted speaker it was mapped to drawn below in the same color; with **differences** on, that lane is marked red where the reference speaker talks but the prediction misses it and amber where the prediction adds speech the reference does not have (collar excluded);
    - the model's speaker probability, with the onset/offset thresholds as dashed lines;
    - error lanes: missed / false alarm / confusion / reference overlap (gray = inside the collar, not scored);
  - both: **lanes S / M / L / XL** sets the height of the speaker and error lanes, an overview strip with every error position, and **Speakers** chips above the timeline: click to hide / show a speaker, double-click to show only that one, **all** to show everyone.
- **Error list**: click a row to zoom to it, ▶ to listen to it. Show only some error types (**miss** / **FA** / **conf**; ◀ error ▶ follows the same choice). Sort by time or by size.
- **Speakers table**: mapping, talk time, recall and precision per speaker.
- **Download**: RTTM, or JSON (includes the raw speaker probabilities).

Mouse: wheel = zoom, drag or shift+wheel = pan, click = seek, shift+drag = zoom to a region, double-click = fit.
Keys: `space` play, `←` `→` seek 5 s (`shift` + `←` `→` pan), `m` mute, `f` full screen (metrics, timeline, error list and speakers table; `Esc` leaves), `+` `−` zoom, `0` fit, `n` / `p` next / previous error.

## Scoring

- 10 ms frames, optimal one-to-one speaker mapping (Hungarian), DER = (missed + false alarm + confusion) / reference speech.
- JER (DIHARD / dscore): per reference speaker 1 − IoU with its optimally mapped prediction (separate IoU-maximizing mapping, 100% when unmatched), averaged over reference speakers. Same collar / overlap settings as DER. pyannote.metrics maps speakers by overlap time instead, so its JER can be a little higher.
- **Collar** is ± seconds around every reference boundary (NIST / NeMo convention). pyannote's `collar` is the *total* width,
  so pyannote `collar=0.25` equals collar ±0.125 here.
- If the reference has a speaker overlapping with themselves, that time is counted once. pyannote.metrics counts it twice.
- Checked against pyannote.metrics on 720 RTTM pairs: within 0.2 DER points.
- Predicted segments come from NeMo's own post-processing, identical to `SortformerEncLabelModel.diarize()`.

## Files

```
backend/server.py    FastAPI app: the API, and serves the frontend
backend/model.py     NeMo wrapper: load .nemo / .ckpt, inference, post-processing
backend/scoring.py   RTTM read/write, DER, speaker mapping, error regions
frontend/            index.html, app.js (UI), timeline.js (canvas timeline), style.css
```

## Notes

- Inference is offline (whole file in one pass). That is fine for minutes of audio; hours may run out of GPU memory.
- The last 256 model outputs are cached in memory (only the speaker probabilities: about 70 KB per 3 minutes of audio with 8 speakers). Nothing is written to disk except uploads, which go to `uploads/` in the project: `audio/`, `rttm/` (reference), `pred_rttm/` (prediction) and `checkpoints/`. Re-uploading a file with the same name replaces it.
