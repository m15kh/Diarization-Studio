"""Diarization Inspector: FastAPI backend that also serves the web UI in ../frontend.

    python backend/server.py [--host 127.0.0.1] [--port 7860] [--device auto]
                             [--checkpoint model.nemo] [--audio a.wav] [--rttm a.rttm] [--hyp-rttm pred.rttm]

No database: the loaded model and recent model outputs live in memory only.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import scoring

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
# uploads/audio and uploads/rttm sit side by side, so an uploaded reference is found next to its audio (../rttm)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_FOLDERS = {"audio": "audio", "ref": "rttm", "hyp": "pred_rttm", "checkpoint": "checkpoints"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm"}
AUDIO_SUBDIRS = ("wav", "wavs", "audio")  # audio folders of a dataset (test/spk2/wav/): not part of a file's group
EXTENSIONS = {"checkpoint": {".nemo", ".ckpt"}, "audio": AUDIO_EXTS, "rttm": {".rttm"}}
PP_KEYS = {"onset", "offset", "pad_onset", "pad_offset", "min_duration_on", "min_duration_off"}
CACHE_SIZE = 256  # model outputs kept in memory, so changing thresholds never re-runs the model

app = FastAPI(title="Diarization Inspector")
state = {"device": "auto", "model": None}
startup = {"launch": uuid.uuid4().hex[:8], "paths": {}}  # paths given on the command line, filled into the UI
lock = threading.Lock()  # one model, one inference at a time
cache: OrderedDict = OrderedDict()


class LoadRequest(BaseModel):
    path: str


class RunRequest(BaseModel):
    audio: str
    checkpoint: str = ""
    ref_rttm: str = ""
    hyp_rttm: str = ""  # an existing prediction to inspect instead of running the model
    onset: float = 0.5
    offset: float = 0.5
    pad_onset: float = 0.0
    pad_offset: float = 0.0
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    collar: float = 0.25
    ignore_overlap: bool = False
    brief: bool = False  # only the numbers (folder runs): no probabilities, segments or error lanes


@app.get("/api/model")
def model_info():
    return state["model"].info() if state["model"] else None


@app.post("/api/model")
def load_model(req: LoadRequest):
    path = existing(req.path, "Checkpoint")
    with lock:
        return get_model(path).info()


@app.get("/api/startup")
def startup_paths():
    """Paths from the command line. `launch` changes on every server start, so the UI applies them once."""
    return startup


@app.post("/api/run")
def run(req: RunRequest):
    audio = existing(req.audio, "Audio")
    file_id = Path(audio).stem
    info = audio_info(audio)
    out = {"file_id": file_id, "audio": audio, "audio_info": info, "warnings": []}

    if req.hyp_rttm:
        hyp_path = existing(req.hyp_rttm, "Prediction RTTM")
        hyp_names, hyp_segs, hyp_id = scoring.read_rttm(hyp_path, file_id)
        check_recording_id(out, "Prediction RTTM", hyp_id, file_id)
        duration = info.get("duration") or 0.0
        out.update(source="rttm", hyp_path=hyp_path)
    else:
        if not req.checkpoint:
            raise HTTPException(400, "Choose a checkpoint, or a prediction RTTM to inspect instead.")
        checkpoint = existing(req.checkpoint, "Checkpoint")
        with lock:
            model = get_model(checkpoint)
            result = infer(model, audio)
        from model import postprocess

        probs = result["probs"]
        hyp_segs = postprocess(probs, model.frame_step, {key: getattr(req, key) for key in PP_KEYS})
        hyp_names = [f"speaker_{k}" for k in range(probs.shape[1])]
        duration = result["duration"]
        out.update(
            source="model",
            model=model.info(),
            frame_step=model.frame_step,
            probs=np.round(probs.T, 3).tolist(),
            infer_time=round(result["time"], 3),
            cached=result["cached"],
        )

    out["duration"] = round(duration, 3)
    out["hyp"] = {"names": hyp_names, "segments": hyp_segs,
                  **scoring.activity_stats(hyp_segs, len(hyp_names), duration)}

    if req.ref_rttm:
        ref_names, ref_segs, ref_id = scoring.read_rttm(existing(req.ref_rttm, "Reference RTTM"), file_id)
        check_recording_id(out, "Reference RTTM", ref_id, file_id)
        self_overlaps = scoring.count_self_overlaps(ref_segs)
        if self_overlaps:
            out["warnings"].append(
                f"Reference has {self_overlaps} place(s) where a speaker overlaps with themselves. They are merged "
                "for scoring (pyannote.metrics counts that time twice, which inflates its DER)."
            )
        out["ref"] = {"names": ref_names, "segments": ref_segs,
                      **scoring.activity_stats(ref_segs, len(ref_names), duration)}
        out["score"] = scoring.score((ref_names, ref_segs), (hyp_names, hyp_segs), duration,
                                     collar=req.collar, ignore_overlap=req.ignore_overlap)
        late = max((seg[1] for seg in ref_segs), default=0)
        if duration and late > duration + 0.5:
            out["warnings"].append(f"Reference goes to {late:.2f} s but the audio is only {duration:.2f} s long.")

    out["rttm"] = scoring.write_rttm(file_id, hyp_names, hyp_segs)
    if req.brief:
        out.pop("probs", None)
        for key in ("hyp", "ref"):
            out.get(key, {}).pop("segments", None)
        if "score" in out:
            out["score"] = {k: v for k, v in out["score"].items() if k not in ("lanes", "regions", "speakers")}
    return out


@app.get("/api/folder")
def folder(audio: str, ref: str = "", hyp: str = ""):
    """The audio files under a folder, each paired with <name>.rttm from the `ref` / `hyp` folders.

    Sub-folders are searched too, so a whole test set (test/spk2/wav/, test/spk3/wav/, ...) works from its top folder;
    each file's `group` is the sub-folder it came from (spk2, without a trailing wav/). RTTM folders are searched
    the same way. A file path for `ref` / `hyp` means its folder. An empty reference is looked up next to each audio
    file (same folder, ../rttm, ./rttm), like /api/pair does.
    """
    audio_dir = Path(audio).expanduser().absolute()
    if not audio_dir.is_dir():
        raise HTTPException(400, f"Audio folder not found: {audio}")
    files = find_files(audio_dir, AUDIO_EXTS)
    if not files:
        raise HTTPException(400, f"No audio files in {audio_dir} or its sub-folders")

    def rttm_index(path: str, what: str) -> tuple[Path | None, dict[str, str]]:
        if not path:
            return None, {}
        d = Path(path).expanduser().absolute()
        d = d.parent if d.is_file() else d
        if not d.is_dir():
            raise HTTPException(400, f"{what} folder not found: {path}")
        index: dict[str, str] = {}
        for f in find_files(d, {".rttm"}):
            index.setdefault(f.stem, str(f))  # the same name twice: the first in sort order wins
        return d, index

    ref_dir, refs = rttm_index(ref, "Reference RTTM")
    hyp_dir, hyps = rttm_index(hyp, "Prediction RTTM")

    def group(f: Path) -> str:
        parts = f.parent.relative_to(audio_dir).parts
        if parts and parts[-1].lower() in AUDIO_SUBDIRS:
            parts = parts[:-1]
        return "/".join(parts)

    return {
        "path": str(audio_dir),
        "ref_dir": str(ref_dir or ""),
        "hyp_dir": str(hyp_dir or ""),
        "files": [{"audio": str(f), "name": str(f.relative_to(audio_dir)), "group": group(f),
                   "ref": refs.get(f.stem, "") if ref_dir else pair(str(f))["ref"],
                   "hyp": hyps.get(f.stem, "")} for f in files],
    }


@app.get("/api/browse")
def browse(path: str = "", kind: str = "audio"):
    """List sub-folders and files of one kind (checkpoint, audio, rttm) in a server directory."""
    folder = Path(path).expanduser().absolute() if path else Path.cwd()
    while not folder.is_dir() and folder != folder.parent:  # typo or file path: use the closest folder
        folder = folder.parent
    exts = EXTENSIONS.get(kind, set())
    dirs, files = [], []
    for entry in sorted(os.scandir(folder), key=lambda e: scoring.natural_key(e.name)):
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": entry.path})
            elif Path(entry.name).suffix.lower() in exts:
                st = entry.stat()
                files.append({"name": entry.name, "path": entry.path, "size": st.st_size, "mtime": st.st_mtime})
        except OSError:
            continue
    parent = str(folder.parent) if folder.parent != folder else None
    return {"path": str(folder), "parent": parent, "dirs": dirs, "files": files}


@app.get("/api/neighbor")
def neighbor(path: str, step: int = 1):
    """Previous / next audio file in the same folder (natural sort order)."""
    current = Path(path).expanduser().absolute()
    if not current.parent.is_dir():
        return {"path": ""}
    files = audio_files(current.parent)
    names = [f.name for f in files]
    index = names.index(current.name) + step if current.name in names else 0
    return {"path": str(files[index]) if 0 <= index < len(files) else ""}


@app.get("/api/pair")
def pair(audio: str, prev_audio: str = "", ref: str = "", hyp: str = ""):
    """Suggest the reference / prediction RTTMs that belong to a newly chosen audio file.

    An RTTM that was paired with the previous audio (same file name) follows the new audio;
    an empty reference is looked up next to the audio (same folder, ../rttm, ./rttm);
    an RTTM folder gives the file in it named like the audio;
    anything else the user picked by hand is kept.
    """
    stem = Path(audio).stem
    prev_stem = Path(prev_audio).stem if prev_audio else None
    audio_dir = Path(audio).expanduser().parent

    def follow(current: str, fallback_dirs: list[Path]) -> str:
        if current and Path(current).expanduser().is_dir():  # an RTTM folder: the file named like the audio
            dirs = [Path(current).expanduser()] + fallback_dirs
        elif current and Path(current).stem != prev_stem:
            return current
        else:
            dirs = ([Path(current).expanduser().parent] if current else []) + fallback_dirs
        return next((str(d / f"{stem}.rttm") for d in dirs if (d / f"{stem}.rttm").is_file()), "")

    return {
        "ref": follow(ref, [audio_dir, audio_dir.parent / "rttm", audio_dir / "rttm"]),
        "hyp": follow(hyp, []),
    }


@app.post("/api/upload")
def upload(file: UploadFile = File(...), field: str = Form(""), sample_rate: int = Form(0)):
    """Store an uploaded file in uploads/<kind>/ on the server and return its path.

    A file with the same name is replaced (results are cached by modification time, so this is safe).
    Uploaded audio that is not WAV (mp3, m4a, ...) is converted to WAV and only the WAV is kept;
    with `sample_rate` > 0 it is also resampled to that rate (a WAV at another rate included).
    """
    dest = UPLOAD_DIR / UPLOAD_FOLDERS.get(field, "other") / Path(file.filename or "upload").name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    if field == "audio":
        is_wav = dest.suffix.lower() == ".wav"
        if not is_wav or (sample_rate > 0 and audio_info(str(dest)).get("sample_rate") != sample_rate):
            dest = to_wav(dest, sample_rate)
    return {"path": str(dest)}


def to_wav(src: Path, sample_rate: int = 0) -> Path:
    """Convert an audio file to 16-bit PCM WAV next to it with ffmpeg (resampled when `sample_rate` > 0).

    The original is replaced by the WAV.
    """
    if shutil.which("ffmpeg") is None:
        raise ValueError(f"Cannot convert {src.name} to WAV: ffmpeg is not installed on the server.")
    wav = src.with_suffix(".wav")
    tmp = src.with_name(f".{uuid.uuid4().hex}.wav")  # ffmpeg cannot write over its own input (a WAV being resampled)
    rate = ["-ar", str(sample_rate)] if sample_rate > 0 else []
    proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *rate, "-c:a", "pcm_s16le", str(tmp)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Could not convert {src.name} to WAV: {proc.stderr.strip() or 'ffmpeg failed'}")
    src.unlink()
    tmp.replace(wav)
    return wav


@app.get("/api/audio")
def audio_file(path: str):
    p = Path(path).expanduser()
    if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
        raise HTTPException(404, f"Not an audio file: {path}")
    return FileResponse(p)


@app.exception_handler(ValueError)
async def bad_input(_request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def server_error(_request, exc: Exception):
    trace = "".join(traceback.format_exception(exc))
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}", "trace": trace})


def get_model(path: str):
    """Return the loaded model, (re)loading it if the path or the file changed. Call with `lock` held."""
    from model import Diarizer

    current = state["model"]
    if current is None or current.path != path or current.mtime != os.path.getmtime(path):
        if not path.endswith((".nemo", ".ckpt")):
            raise ValueError(f"Checkpoint must be a .nemo or .ckpt file: {path}")
        state["model"] = current = None  # free the old model before loading the new one
        gc.collect()
        state["model"] = Diarizer(path, state["device"])
    return state["model"]


def infer(model, audio: str) -> dict:
    """Run the model on one file, reusing the cached output when model and audio are unchanged."""
    key = (model.path, model.mtime, audio, os.path.getmtime(audio))
    if key in cache:
        cache.move_to_end(key)
        return {**cache[key], "cached": True}
    start = time.time()
    probs, duration = model.predict(audio)
    cache[key] = {"probs": probs, "duration": duration, "time": time.time() - start}
    while len(cache) > CACHE_SIZE:
        cache.popitem(last=False)
    return {**cache[key], "cached": False}


def audio_files(folder: Path) -> list[Path]:
    """The audio files in a folder, in natural sort order ([] when it is not a folder)."""
    if not folder.is_dir():
        return []
    return sorted((f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS and f.is_file()),
                  key=lambda f: scoring.natural_key(f.name))


def find_files(folder: Path, exts: set[str], limit: int = 50000) -> list[Path]:
    """Files with one of `exts` in a folder and its sub-folders (hidden ones skipped), in natural order of their paths."""
    found, seen = [], set()
    for root, dirs, names in os.walk(folder, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:  # a symlink loop, or a folder linked twice
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        found += [Path(root, n) for n in names if Path(n).suffix.lower() in exts and not n.startswith(".")]
        if len(found) > limit:
            raise HTTPException(400, f"More than {limit} files under {folder}: choose a smaller folder")
    return sorted(found, key=lambda f: scoring.natural_key(str(f.relative_to(folder))))


def existing(path: str, what: str) -> str:
    full = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(full):
        raise HTTPException(400, f"{what} not found: {path}")
    return full


def audio_info(path: str) -> dict:
    try:
        import soundfile

        info = soundfile.info(path)
        return {"duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels,
                "format": f"{info.format} {info.subtype}"}
    except Exception:
        return {}


def check_recording_id(out: dict, what: str, rec_id: str, file_id: str) -> None:
    if rec_id != file_id:
        out["warnings"].append(f"{what} is for recording '{rec_id}' but the audio file is '{file_id}'.")


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Make browsers revalidate the page on every load, so an edited frontend is never served from cache."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main():
    parser = argparse.ArgumentParser(description="Diarization inspector: FastAPI backend + web UI.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 makes the UI reachable from other machines (it can browse server files)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--checkpoint", help="checkpoint (.nemo / .ckpt) to load at startup")
    parser.add_argument("--audio", help="audio file or folder to show in the UI (runs when the page opens)")
    parser.add_argument("--rttm", "--ref-rttm", dest="ref_rttm",
                        help="reference RTTM file or folder (default: found next to the audio or in ../rttm/)")
    parser.add_argument("--hyp-rttm", help="prediction RTTM file or folder to inspect instead of running the checkpoint")
    args = parser.parse_args()

    def cli_path(path: str, what: str) -> str:
        full = os.path.abspath(os.path.expanduser(path))
        if not (os.path.isfile(full) or what != "Checkpoint" and os.path.isdir(full)):
            parser.error(f"{what} not found: {path}")
        return full

    # keys are the UI field names
    paths = startup["paths"]
    for key, path, what in [("checkpoint", args.checkpoint, "Checkpoint"), ("audio", args.audio, "Audio"),
                            ("ref", args.ref_rttm, "Reference RTTM"), ("hyp", args.hyp_rttm, "Prediction RTTM")]:
        if path:
            paths[key] = cli_path(path, what)
            print(f"{what + ':':17}{paths[key]}")
    if "audio" in paths and "ref" not in paths and os.path.isfile(paths["audio"]):  # a folder pairs per file
        paths["ref"] = pair(paths["audio"])["ref"]
        print(f"{'Reference RTTM:':17}{paths['ref'] or '(none found next to the audio)'}")

    state["device"] = args.device
    if args.checkpoint:
        print(f"Loading {args.checkpoint} ...")
        state["model"] = get_model(paths["checkpoint"])

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
