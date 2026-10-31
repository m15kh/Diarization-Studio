"""RTTM reading/writing and frame-based diarization scoring.

DER follows the NIST md-eval definition:

    DER = (missed speech + false alarm + speaker confusion) / total reference speech

It is computed on a 10 ms grid with an optimal one-to-one speaker mapping (Hungarian algorithm),
an unscored collar of +/- `collar` seconds around every reference segment boundary, and, optionally,
without the regions where two or more reference speakers overlap.

JER follows the DIHARD (dscore) definition: every reference speaker is paired with at most one
predicted speaker by a separate optimal mapping that maximizes intersection over union, its error is
1 - IoU (1 when unpaired), and JER is the mean over reference speakers. It uses the same scored region.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

STEP = 0.01  # scoring grid in seconds


def natural_key(text: str) -> list:
    """Sort key that orders spk2 before spk10."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def read_rttm(path: str, file_id: str | None = None, allow_empty: bool = False) -> tuple[list[str], list[list], str]:
    """Parse an RTTM file.

    Returns (speaker names, segments as [start, end, speaker index], recording id).
    If the file holds several recordings, only the one named `file_id` is kept.
    A file without SPEAKER lines is an error, unless `allow_empty` (then: no speakers, recording id `file_id`).
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            rows.append((parts[1], float(parts[3]), float(parts[4]), parts[7]))
    if not rows:
        if allow_empty:
            return [], [], file_id or ""
        raise ValueError(f"No SPEAKER lines found in {path}")

    rec_ids = sorted({row[0] for row in rows})
    if len(rec_ids) > 1:
        if file_id not in rec_ids:
            shown = ", ".join(rec_ids[:5]) + (", ..." if len(rec_ids) > 5 else "")
            raise ValueError(f"{Path(path).name} holds {len(rec_ids)} recordings ({shown}) and none is '{file_id}'")
        rows = [row for row in rows if row[0] == file_id]
        rec_ids = [file_id]

    names = sorted({row[3] for row in rows}, key=natural_key)
    index = {name: i for i, name in enumerate(names)}
    segments = sorted([round(start, 3), round(start + dur, 3), index[name]] for _, start, dur, name in rows if dur > 0)
    return names, segments, rec_ids[0]


def write_rttm(file_id: str, names: list[str], segments: list[list]) -> str:
    """Render segments as RTTM text."""
    return "".join(
        f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} <NA> <NA> {names[spk]} <NA> <NA>\n"
        for start, end, spk in sorted(segments)
    )


def count_self_overlaps(segments: list[list]) -> int:
    """Number of segments that start before the previous segment of the same speaker has ended."""
    count, last_end = 0, {}
    for start, end, spk in sorted(segments):
        if start < last_end.get(spk, float("-inf")):
            count += 1
        last_end[spk] = max(end, last_end.get(spk, float("-inf")))
    return count


def activity_stats(segments: list[list], num_speakers: int, duration: float) -> dict:
    """Speech, overlap and per-speaker talk time (seconds) of one annotation."""
    act = _activity(segments, num_speakers, _num_frames(duration, segments))
    count = act.sum(axis=1)
    per_speaker = act.sum(axis=0) * STEP
    return {
        "speakers": int((per_speaker > 0).sum()),
        "speech": _sec(np.count_nonzero(count)),
        "overlap": _sec(np.count_nonzero(count > 1)),
        "per_speaker": [round(float(t), 2) for t in per_speaker],
    }


def score(ref: tuple[list[str], list[list]], hyp: tuple[list[str], list[list]], duration: float,
          collar: float = 0.25, ignore_overlap: bool = False) -> dict:
    """Score a hypothesis against a reference.

    `ref` and `hyp` are (speaker names, [[start, end, speaker index], ...]).
    Returns DER and its components, the speaker mapping, error lanes, error regions and per-speaker stats.
    """
    ref_names, ref_segs = ref
    hyp_names, hyp_segs = hyp
    n = _num_frames(duration, ref_segs + hyp_segs)
    R = _activity(ref_segs, len(ref_names), n)
    H = _activity(hyp_segs, len(hyp_names), n)
    n_ref = R.sum(axis=1)
    n_hyp = H.sum(axis=1)

    # Collars go around the boundaries of each speaker's merged activity, so self-overlapping or
    # touching segments of one speaker do not create fake boundaries.
    scored = np.ones(n, dtype=bool)
    c = int(round(collar / STEP))
    if c > 0:
        for r in range(len(ref_names)):
            for a, b in _runs(R[:, r]):
                scored[max(a - c, 0):a + c] = False
                scored[max(b - c, 0):b + c] = False
    if ignore_overlap:
        scored &= n_ref <= 1

    # Optimal one-to-one mapping: maximize the scored time where both speakers are active.
    cooc = R[scored].T.astype(np.float32) @ H[scored].astype(np.float32)
    hyp_to_ref: list[int | None] = [None] * len(hyp_names)
    if cooc.size:
        for r, h in zip(*linear_sum_assignment(cooc, maximize=True)):
            if cooc[r, h] > 0:
                hyp_to_ref[h] = int(r)
    mapped = np.zeros_like(R)
    for h, r in enumerate(hyp_to_ref):
        if r is not None:
            mapped[:, r] = H[:, h]
    correct = (R & mapped).sum(axis=1)

    miss = np.where(scored, np.maximum(n_ref - n_hyp, 0), 0)
    fa = np.where(scored, np.maximum(n_hyp - n_ref, 0), 0)
    conf = np.where(scored, np.minimum(n_ref, n_hyp) - correct, 0)
    total = int(n_ref[scored].sum())

    def rate(frames) -> float | None:
        return round(float(frames) / total, 5) if total else None

    # JER: its own mapping, maximizing the per-pair intersection over union.
    ref_time = R[scored].sum(axis=0).astype(np.float32)
    hyp_time = H[scored].sum(axis=0).astype(np.float32)
    union = ref_time[:, None] + hyp_time[None, :] - cooc
    iou = np.divide(cooc, union, out=np.zeros_like(cooc), where=union > 0)
    ref_jer = np.ones(len(ref_names))
    if iou.size:
        for r, h in zip(*linear_sum_assignment(iou, maximize=True)):
            ref_jer[r] = 1.0 - iou[r, h]
    jer_speakers = ref_time > 0  # speakers with no scored time are left out, like dscore

    regions = []
    for a, b in _runs(miss + fa + conf > 0):
        regions.append({
            "start": _sec(a),
            "end": _sec(b),
            "miss": _sec(miss[a:b].sum()),
            "fa": _sec(fa[a:b].sum()),
            "conf": _sec(conf[a:b].sum()),
            "ref": [int(k) for k in np.flatnonzero(R[a:b].any(axis=0))],
            "hyp": [int(k) for k in np.flatnonzero(H[a:b].any(axis=0))],
        })

    speakers = []
    ref_to_hyp = {r: h for h, r in enumerate(hyp_to_ref) if r is not None}
    for r in range(len(ref_names)):
        h = ref_to_hyp.get(r)
        ref_scored = np.count_nonzero(R[:, r] & scored)
        hit = np.count_nonzero(R[:, r] & mapped[:, r] & scored)
        hyp_scored = np.count_nonzero(H[:, h] & scored) if h is not None else 0
        speakers.append({
            "ref": r,
            "hyp": h,
            "ref_time": _sec(np.count_nonzero(R[:, r])),
            "hyp_time": _sec(np.count_nonzero(H[:, h])) if h is not None else 0.0,
            "recall": round(hit / ref_scored, 4) if ref_scored else None,
            "precision": round(hit / hyp_scored, 4) if hyp_scored else None,
            "jer": round(float(ref_jer[r]), 4) if jer_speakers[r] else None,
        })
    for h in range(len(hyp_names)):
        if hyp_to_ref[h] is None and H[:, h].any():
            speakers.append({"ref": None, "hyp": h, "ref_time": 0.0, "hyp_time": _sec(np.count_nonzero(H[:, h])),
                             "recall": None, "precision": 0.0, "jer": None})

    return {
        "der": rate(miss.sum() + fa.sum() + conf.sum()),
        "miss": rate(miss.sum()),
        "fa": rate(fa.sum()),
        "conf": rate(conf.sum()),
        "jer": round(float(ref_jer[jer_speakers].mean()), 5) if jer_speakers.any() else None,
        "jer_sum": round(float(ref_jer[jer_speakers].sum()), 5),
        "jer_speakers": int(jer_speakers.sum()),
        "miss_s": _sec(miss.sum()),
        "fa_s": _sec(fa.sum()),
        "conf_s": _sec(conf.sum()),
        "ref_speech": _sec(total),
        "collar": collar,
        "ignore_overlap": ignore_overlap,
        "hyp_to_ref": hyp_to_ref,
        "lanes": {
            "miss": _runs_sec(miss > 0),
            "fa": _runs_sec(fa > 0),
            "conf": _runs_sec(conf > 0),
            "overlap": _runs_sec(n_ref > 1),
            "unscored": _runs_sec(~scored),
        },
        "regions": regions,
        "speakers": speakers,
    }


def _num_frames(duration: float, segments: list[list]) -> int:
    end = max([duration or 0.0] + [seg[1] for seg in segments])
    return int(np.ceil(end / STEP)) + 1


def _activity(segments: list[list], num_speakers: int, num_frames: int) -> np.ndarray:
    act = np.zeros((num_frames, num_speakers), dtype=bool)
    for start, end, spk in segments:
        act[max(int(round(start / STEP)), 0):int(round(end / STEP)), spk] = True
    return act


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last + 1) frame indices of every run of True values."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _runs_sec(mask: np.ndarray) -> list[list[float]]:
    return [[_sec(a), _sec(b)] for a, b in _runs(mask)]


def _sec(frames) -> float:
    return round(float(frames) * STEP, 3)
