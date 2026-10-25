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
| `--device auto` | `auto`, `cpu`, `cuda`, `cuda:1`, ... |
| `--port 7860` | |
| `--host 127.0.0.1` | `0.0.0.0` makes it reachable from other machines; the UI can browse the server's files, so only do that on a trusted network |

Server on a remote GPU machine: run it there and tunnel the port, then open http://localhost:7860 locally:

```bash
ssh -L 7860:localhost:7860 user@gpu-server
```

## Use

1. **Checkpoint**: a path on the server (Browse lists `.nemo` / `.ckpt`). The model stays loaded. The number of speaker outputs (2, 4, 8, ...) is read from the checkpoint.
2. **Audio**: a server path, Browse, Upload, or drag and drop. ◀ ▶ step through the files of the folder and run them.
3. **Reference RTTM** (optional): ground truth. It is found automatically next to the audio or in `../rttm/` (the `wav/` + `rttm/` layout of the datasets).
4. **Prediction RTTM** (optional): inspect any existing output (pyannote, DiariZen, older runs, ...) instead of running the model.

The post-processing fields use NeMo's `PostProcessingParams` names. Changing them, the collar or "ignore overlap" re-scores instantly: the model output is cached, so only post-processing and scoring re-run. **Copy YAML** gives a `postprocessing_yaml` for `e2e_diarize_speech.py`.

What you get:

- **Metrics**: DER = missed + false alarm + confusion, speaker count, overlap, inference time / RTF.
- **Timeline** (all rows share one zoomable time axis):
  - the waveform;
  - each reference speaker, with the predicted speaker it was mapped to drawn below in the same color;
  - the model's speaker probability, with the onset/offset thresholds as dashed lines;
  - error lanes: missed / false alarm / confusion / reference overlap (gray = inside the collar, not scored);
  - an overview strip with every error position.
- **Error list**: click a row to zoom to it, ▶ to listen to it. Sort by time or by size.
- **Speakers table**: mapping, talk time, recall and precision per speaker.
- **Download**: RTTM, or JSON (includes the raw speaker probabilities).

Mouse: wheel = zoom, drag or shift+wheel = pan, click = seek, shift+drag = zoom to a region, double-click = fit.
Keys: `space` play, `←` `→` pan, `+` `−` zoom, `0` fit, `n` / `p` next / previous error.

## Scoring

- 10 ms frames, optimal one-to-one speaker mapping (Hungarian), DER = (missed + false alarm + confusion) / reference speech.
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
- The last 32 model outputs are cached in memory. Nothing is written to disk except uploads (`/tmp/diar_inspector_uploads`).
