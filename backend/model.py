"""NeMo Sortformer wrapper: load a .nemo / .ckpt checkpoint, run offline inference, post-process.

Post-processing calls NeMo's own `binarization_vectorized` + `filtering`, so the segments are
identical to what `SortformerEncLabelModel.diarize()` produces with the same parameters.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch


class Diarizer:
    def __init__(self, path: str, device: str = "auto"):
        from nemo.collections.asr.models import SortformerEncLabelModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.path = path
        self.mtime = os.path.getmtime(path)

        start = time.time()
        if path.endswith(".nemo"):
            model = SortformerEncLabelModel.restore_from(restore_path=path, map_location=self.device, strict=False)
        elif path.endswith(".ckpt"):
            model = SortformerEncLabelModel.load_from_checkpoint(
                checkpoint_path=path, map_location=self.device, strict=False
            )
        else:
            raise ValueError(f"Checkpoint must be a .nemo or .ckpt file: {path}")
        self.model = model.eval()
        self.load_time = time.time() - start

        self.sample_rate = int(model.preprocessor._sample_rate)
        self.num_speakers = int(model._cfg.max_num_of_spks)
        window_stride = float(model._cfg.preprocessor.get("window_stride", 0.01))
        self.frame_step = round(model.output_subsampling_factor * window_stride, 6)
        self.num_params = sum(p.numel() for p in model.parameters())

    def info(self) -> dict:
        return {
            "path": self.path,
            "name": os.path.basename(self.path),
            "arch": type(self.model).__name__,
            "num_speakers": self.num_speakers,
            "sample_rate": self.sample_rate,
            "frame_step": self.frame_step,
            "params_m": round(self.num_params / 1e6, 1),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "load_time": round(self.load_time, 1),
            "streaming": bool(self.model.streaming_mode),
        }

    @torch.inference_mode()
    def predict(self, audio_path: str) -> tuple[np.ndarray, float]:
        """Speaker probabilities of shape [frames, speakers] and the audio duration in seconds."""
        from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

        audio = AudioSegment.from_file(audio_path, target_sr=self.sample_rate, channel_selector="average")
        signal = torch.as_tensor(audio.samples, dtype=torch.float32, device=self.device).unsqueeze(0)
        length = torch.tensor([signal.shape[1]], device=self.device)
        preds = self.model.forward(audio_signal=signal, audio_signal_length=length)
        return preds[0].float().cpu().numpy(), signal.shape[1] / self.sample_rate


def postprocess(probs: np.ndarray, frame_step: float, params: dict) -> list[list]:
    """Binarize speaker probabilities into [[start, end, speaker index], ...] with NeMo's post-processing.

    `params` holds NeMo's PostProcessingParams: onset, offset, pad_onset, pad_offset,
    min_duration_on, min_duration_off.
    """
    from nemo.collections.asr.parts.utils.vad_utils import binarization_vectorized, filtering

    pp = {key: float(value) for key, value in params.items()}
    pp.update(frame_length_in_sec=float(frame_step), filter_speech_first=1.0)
    probs = torch.from_numpy(probs)
    segments = []
    for spk in range(probs.shape[1]):
        spans = binarization_vectorized(probs[:, spk], pp)
        if spans.numel() > 0:
            spans = filtering(spans, pp)
        segments += [[round(start, 2), round(end, 2), spk] for start, end in spans.tolist()]
    return sorted(segments)
