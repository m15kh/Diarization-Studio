"""Diarization Inspector: FastAPI backend that also serves the web UI in ../frontend.

    python backend/server.py [--host 127.0.0.1] [--port 7860] [--device auto]
                             [--checkpoint model.nemo] [--audio a.wav] [--rttm a.rttm] [--hyp-rttm pred.rttm]

No database: the loaded model and recent model outputs live in memory only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import io
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import unicodedata
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import scoring

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
# files the app writes for itself: data/uploads/ and data/decoded/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# uploads/audio and uploads/rttm sit side by side, so an uploaded reference is found next to its audio (../rttm)
UPLOAD_DIR = DATA_DIR / "uploads"
# saved errors: <id>.json (what, where, which model and settings) + <id>.wav (the audio around it) + saved.csv (all of them)
SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"
# audio libsndfile cannot read (m4a, webm, ...) is decoded once with ffmpeg into a WAV here, for the model and the clips
DECODED_DIR = DATA_DIR / "decoded"
# the Label tab keeps the original of every RTTM it overwrites here (the first time only)
LABEL_BACKUP_DIR = DATA_DIR / "label_backups"
UPLOAD_FOLDERS = {"audio": "audio", "ref": "rttm", "hyp": "pred_rttm", "checkpoint": "checkpoints", "checkpoint2": "checkpoints"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm"}
AUDIO_SUBDIRS = ("wav", "wavs", "audio")  # audio folders of a dataset (test/spk2/wav/): not part of a file's group
EXTENSIONS = {"checkpoint": {".nemo", ".ckpt"}, "audio": AUDIO_EXTS, "rttm": {".rttm"}}
PP_KEYS = {"onset", "offset", "pad_onset", "pad_offset", "min_duration_on", "min_duration_off"}
CACHE_SIZE = 256  # model outputs kept in memory, so changing thresholds never re-runs the model
MODEL_SLOTS = 2   # checkpoints kept loaded at once, so two models can be compared without reloading

app = FastAPI(title="Diarization Inspector")
state = {"device": "auto", "model": None, "models": OrderedDict()}  # model: the one loaded last; models: by path
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
    info = audio_info(decoded(audio))
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
            result = infer(model, decoded(audio))
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
            index.setdefault(name_key(f.stem), str(f))  # the same name twice: the first in sort order wins
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
                   "ref": refs.get(name_key(f.stem), "") if ref_dir else pair(str(f))["ref"],
                   "hyp": hyps.get(name_key(f.stem), "")} for f in files],
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
    prev_key = name_key(Path(prev_audio).stem) if prev_audio else None  # "a b.wav" goes with a_b.rttm too
    audio_dir = Path(audio).expanduser().parent

    def follow(current: str, fallback_dirs: list[Path]) -> str:
        if current and Path(current).expanduser().is_dir():  # an RTTM folder: the file named like the audio
            dirs = [Path(current).expanduser()] + fallback_dirs
        elif current and name_key(Path(current).stem) != prev_key:
            return current
        else:
            dirs = ([Path(current).expanduser().parent] if current else []) + fallback_dirs
        return next((found for d in dirs if (found := rttm_named(d, stem))), "")

    return {
        "ref": follow(ref, [audio_dir, audio_dir.parent / "rttm", audio_dir / "rttm"]),
        "hyp": follow(hyp, []),
    }


class SaveRequest(BaseModel):
    audio: str
    start: float
    end: float
    pad: float = 1.0  # seconds of audio kept before and after the error in the clip
    group: str = ""
    miss: float = 0.0
    fa: float = 0.0
    conf: float = 0.0
    ref_speakers: list[str] = []
    hyp_speakers: list[str] = []
    note: str = ""
    ref_rttm: str = ""
    hyp_rttm: str = ""
    checkpoint: str = ""
    settings: dict = {}
    file_der: float | None = None
    segments: dict = {}  # {"ref": [[start, end, speaker], ...], "hyp": [...]} around the error


class NoteRequest(BaseModel):
    note: str


SAVED_CSV_COLUMNS = ["id", "saved_at", "group", "file_id", "start", "end", "duration", "types", "miss", "fa", "conf",
                     "ref_speakers", "hyp_speakers", "note", "file_der", "clip", "audio", "ref_rttm", "hyp_rttm", "checkpoint"]


@app.get("/api/saved")
def saved_list():
    """Every saved error, newest first."""
    return {"dir": str(SAVED_DIR), "items": saved_items()}


@app.post("/api/saved")
def save_error(req: SaveRequest):
    """Save an error region: its audio (with `pad` seconds around it) as a WAV, and everything about it as JSON."""
    import soundfile

    audio = existing(req.audio, "Audio")
    file_id = Path(audio).stem
    info = soundfile.info(decoded(audio))
    clip_start, clip_end = max(0.0, req.start - req.pad), min(info.duration, req.end + req.pad)
    data, sr = soundfile.read(decoded(audio), start=int(clip_start * info.samplerate), stop=int(clip_end * info.samplerate))
    types = [t for t in ("miss", "fa", "conf") if getattr(req, t) > 0]
    item_id = f"{file_id}_{req.start:.2f}s_{'-'.join(types) or 'error'}_{uuid.uuid4().hex[:4]}"
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    soundfile.write(SAVED_DIR / f"{item_id}.wav", data, sr)
    item = {
        "id": item_id, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **req.model_dump(exclude={"pad", "audio"}), "audio": audio, "file_id": file_id, "types": types,
        "duration": round(req.end - req.start, 3), "clip": f"{item_id}.wav",
        "clip_start": round(clip_start, 3), "clip_end": round(clip_end, 3),
    }
    (SAVED_DIR / f"{item_id}.json").write_text(json.dumps(item, ensure_ascii=False, indent=1))
    write_saved_csv()
    return item


@app.patch("/api/saved/{item_id}")
def saved_note(item_id: str, req: NoteRequest):
    path = saved_path(item_id, ".json")
    item = json.loads(path.read_text())
    item["note"] = req.note
    path.write_text(json.dumps(item, ensure_ascii=False, indent=1))
    write_saved_csv()
    return item


@app.delete("/api/saved/{item_id}")
def saved_delete(item_id: str):
    saved_path(item_id, ".json").unlink()
    (SAVED_DIR / f"{item_id}.wav").unlink(missing_ok=True)
    write_saved_csv()
    return {"ok": True}


@app.get("/api/saved/{item_id}/clip")
def saved_clip(item_id: str):
    return FileResponse(saved_path(item_id, ".wav"))


@app.get("/api/saved-export")
def saved_export():
    """All saved errors in one zip: the clips, every JSON, and saved.csv."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SAVED_DIR.glob("*")) if SAVED_DIR.is_dir() else []:
            if f.suffix in (".wav", ".json", ".csv"):
                z.write(f, f.name)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="saved_errors_{stamp}.zip"'})


def saved_items() -> list[dict]:
    items = []
    for f in SAVED_DIR.glob("*.json") if SAVED_DIR.is_dir() else []:
        try:
            items.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(items, key=lambda x: x.get("saved_at", ""), reverse=True)


def saved_path(item_id: str, suffix: str) -> Path:
    path = SAVED_DIR / f"{Path(item_id).name}{suffix}"  # .name: no way out of the folder
    if not path.is_file():
        raise HTTPException(404, f"No saved error {item_id}")
    return path


def write_saved_csv() -> None:
    """saved.csv: one row per saved error, so they can be searched outside the UI too."""
    rows = saved_items()
    with (SAVED_DIR / "saved.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, SAVED_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for item in rows:
            w.writerow({**item, **{k: " ".join(item.get(k, [])) for k in ("types", "ref_speakers", "hyp_speakers")}})


class LabelSaveRequest(BaseModel):
    path: str
    rec_id: str  # recording id written in the RTTM lines
    names: list[str]
    segments: list[list[float]]  # [[start, end, speaker index], ...]


@app.get("/api/label")
def label_open(audio: str, rttm: str = "", folder: str = ""):
    """An audio file and the labels to correct in the Label tab: the speakers and segments of `rttm` (none: no labels
    yet), and where Save writes them (see label_path). When the save `folder` holds labels saved for this file before,
    those open instead, to go on from them (`original` is then the `rttm` asked for)."""
    audio = existing(audio, "Audio")
    file_id = Path(audio).stem
    info = audio_info(decoded(audio))
    out = {"audio": audio, "file_id": file_id, "rec_id": "_".join(file_id.split()), "audio_info": info,
           "duration": round(info.get("duration") or 0.0, 3), "rttm": "", "original": "", "names": [], "segments": [],
           "warnings": []}
    saved = label_path(audio, rttm, folder) if folder.strip() else ""
    if saved and os.path.isfile(saved) and os.path.abspath(os.path.expanduser(rttm.strip())) != saved:
        out["original"], rttm = rttm, saved
    if rttm:
        path = existing(rttm, "RTTM")
        names, segments, rec_id = scoring.read_rttm(path, file_id, allow_empty=True)
        check_recording_id(out, "The RTTM", rec_id, file_id)
        out.update(rttm=path, names=names, segments=segments, rec_id="_".join(rec_id.split()) or out["rec_id"])
        self_overlaps = scoring.count_self_overlaps(segments)
        if self_overlaps:
            out["warnings"].append(
                f"{self_overlaps} segment(s) overlap another segment of the same speaker (darker where they overlap). "
                "Moving or resizing one of them merges the two.")
        late = max((seg[1] for seg in segments), default=0)
        if out["duration"] and late > out["duration"] + 0.5:
            out["warnings"].append(f"The labels go to {late:.2f} s but the audio is only {out['duration']:.2f} s long.")
    out["save_path"] = label_path(audio, out["rttm"], folder)
    if not out["rttm"] and os.path.isfile(out["save_path"]):
        out["warnings"].append(f"{out['save_path']} exists: Save replaces its labels for this recording "
                               f"(a copy of the original is kept in {LABEL_BACKUP_DIR}).")
    return out


@app.post("/api/label")
def label_save(req: LabelSaveRequest):
    """Write labels from the Label tab as an RTTM. Lines of other recordings in that file are kept; the first time an
    existing file is replaced, a copy of it goes to data/label_backups/."""
    path = Path(req.path.strip()).expanduser().absolute()
    if path.suffix.lower() != ".rttm":
        raise HTTPException(400, f"Labels are saved as an .rttm file: {req.path}")
    # RTTM fields are separated by whitespace
    rec_id, names = "_".join(req.rec_id.split()), ["_".join(name.split()) for name in req.names]
    if not rec_id or not all(names) or len(set(names)) < len(names):
        raise HTTPException(400, "Every speaker needs a name of its own")
    segments = []
    for seg in req.segments:
        if len(seg) != 3 or not 0 <= seg[0] < seg[1] or not 0 <= int(seg[2]) < len(names):
            raise HTTPException(400, f"Not a segment: {seg}")
        segments.append([round(seg[0], 3), round(seg[1], 3), int(seg[2])])
    keep, backup = [], ""
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and not (parts[0] == "SPEAKER" and len(parts) > 1 and parts[1] == rec_id):
                keep.append(f"{line}\n")
        original = LABEL_BACKUP_DIR / f"{path.stem}_{uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex[:8]}.rttm"
        if not original.is_file():
            LABEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, original)
            backup = str(original)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{uuid.uuid4().hex}.rttm")  # write next to it, then rename: never a half-written RTTM
    tmp.write_text("".join(keep) + scoring.write_rttm(rec_id, names, segments), encoding="utf-8")
    tmp.replace(path)
    return {"path": str(path), "backup": backup, "segments": len(segments), "kept": len(keep)}


@app.get("/api/label-target")
def label_target(audio: str, rttm: str = "", folder: str = ""):
    """Where Save writes the labels of `audio` (see label_path), and whether that file exists already."""
    path = label_path(os.path.abspath(os.path.expanduser(audio.strip())), rttm, folder)
    return {"path": path, "exists": os.path.isfile(path)}


def label_path(audio: str, rttm: str = "", folder: str = "") -> str:
    """Where Save writes the labels of `audio` (an absolute path).

    With a `folder`: <folder>/<name>.rttm, named like the RTTM opened (else like the audio). A folder given by name
    goes beside wav/ and rttm/ (when the audio is in a wav/, wavs/ or audio/ folder; else into the audio's folder).
    Without: the RTTM opened, else <name>.rttm where /api/pair looks for it (rttm/ beside wav/, else next to the audio).
    """
    a = Path(audio).expanduser()
    in_wav = a.parent.name.lower() in AUDIO_SUBDIRS
    if folder.strip():
        out = Path(folder.strip()).expanduser()
        if not out.is_absolute():
            out = (a.parent.parent if in_wav else a.parent) / out
        return str(out / (Path(rttm).name if rttm.strip() else f"{a.stem}.rttm"))
    return rttm or str((a.parent.parent / "rttm" if in_wav else a.parent) / f"{a.stem}.rttm")


@app.post("/api/upload")
def upload(file: UploadFile = File(...), field: str = Form(""), sample_rate: int = Form(0)):
    """Store an uploaded file in data/uploads/<kind>/ on the server and return its path.

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


@app.post("/api/upload-folder")
def upload_folder(file: UploadFile = File(...), relpath: str = Form(...), batch: str = Form(...)):
    """Store one file of an uploaded folder at data/uploads/folders/<batch>/<relpath> and return the folder's path.

    `relpath` is the file's path inside the chosen folder, starting with that folder's name
    (data/rttm/a.rttm); `batch` keeps uploads of folders with the same name apart. Audio is kept as it is.
    """
    parts = [p for p in Path(relpath.replace("\\", "/")).parts if p not in ("", ".", "..", "/")]
    if len(parts) < 2:
        raise HTTPException(400, f"Bad file path in the upload: {relpath}")
    root = UPLOAD_DIR / "folders" / Path(batch).name / parts[0]
    dest = root.joinpath(*parts[1:])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"path": str(root)}


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


@app.get("/api/clip")
def audio_clip(path: str, start: float, end: float):
    """A section of an audio file as MP3 (for saving the A–B loop)."""
    audio = existing(path, "Audio")
    if end <= start:
        raise HTTPException(400, "The section is empty: B must come after A")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(500, "Cannot make an MP3: ffmpeg is not installed on the server.")
    proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", audio,
                           "-vn", "-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3", "pipe:1"], capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"Could not make the MP3: {proc.stderr.decode(errors='replace').strip() or 'ffmpeg failed'}")
    name = f"{Path(audio).stem}_{start:.2f}-{end:.2f}s.mp3"
    return Response(proc.stdout, media_type="audio/mpeg", headers={"Content-Disposition": f'attachment; filename="{name}"'})


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

    models = state["models"]
    current = models.get(path)
    if current is None or current.mtime != os.path.getmtime(path):
        if not path.endswith((".nemo", ".ckpt")):
            raise ValueError(f"Checkpoint must be a .nemo or .ckpt file: {path}")
        models.pop(path, None)
        while len(models) >= MODEL_SLOTS:  # free the least recently used model before loading another
            models.popitem(last=False)
        state["model"] = current = None
        gc.collect()
        models[path] = current = Diarizer(path, state["device"])
    models.move_to_end(path)
    state["model"] = current
    return current


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


def name_key(name: str) -> str:
    """A file name as matched against RTTM names: spaces and underscores alike, case and Unicode form ignored,
    Arabic and Persian yeh / kaf alike ("۲۰ تیر دیلی" finds ۲۰_تیر_دیلی.rttm)."""
    name = unicodedata.normalize("NFKC", name).casefold().replace("ي", "ی").replace("ك", "ک")
    return "_".join(name.replace("_", " ").split())


def rttm_named(folder: Path, stem: str) -> str:
    """The RTTM in `folder` named like an audio file (exactly, else by `name_key`), or ""."""
    exact = folder / f"{stem}.rttm"
    if exact.is_file():
        return str(exact)
    if not folder.is_dir():
        return ""
    key = name_key(stem)
    return next((str(f) for f in sorted(folder.glob("*.rttm")) if name_key(f.stem) == key), "")


def decoded(audio: str) -> str:
    """`audio` itself when libsndfile reads it, else a WAV decoded from it once with ffmpeg (kept in data/decoded/)."""
    import soundfile

    try:
        soundfile.info(audio)
        return audio
    except Exception:
        pass
    st = os.stat(audio)
    tag = uuid.uuid5(uuid.NAMESPACE_URL, f"{audio}|{st.st_mtime_ns}|{st.st_size}").hex[:12]
    wav = DECODED_DIR / f"{Path(audio).stem}_{tag}.wav"
    if not wav.is_file():
        if shutil.which("ffmpeg") is None:
            raise ValueError(f"Cannot read {Path(audio).name}: it needs ffmpeg, which is not installed on the server.")
        DECODED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = wav.with_name(f".{uuid.uuid4().hex}.wav")
        proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio, "-vn", "-c:a", "pcm_s16le", str(tmp)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Could not decode {Path(audio).name}: {proc.stderr.strip() or 'ffmpeg failed'}")
        tmp.replace(wav)
    return str(wav)


def check_recording_id(out: dict, what: str, rec_id: str, file_id: str) -> None:
    if name_key(rec_id) != name_key(file_id):
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
