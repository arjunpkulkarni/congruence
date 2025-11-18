from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import librosa

try:
    import parselmouth  # Praat bindings for precise jitter/shimmer
    _HAS_PARSELMOUTH = True
except Exception:
    _HAS_PARSELMOUTH = False


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    i0 = max(int(start_s * sr), 0)
    i1 = min(int(end_s * sr), len(y))
    return y[i0:i1]


def compute_f0(y: np.ndarray, sr: int) -> Dict[str, Any]:
    if y.size == 0:
        return {"f0_mean": 0.0, "f0_median": 0.0, "f0_std": 0.0, "f0_series": []}
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        # Fallback: yin
        f0 = librosa.yin(y.astype(np.float32), fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_mean = float(np.mean(f0)) if f0.size else 0.0
    f0_median = float(np.median(f0)) if f0.size else 0.0
    f0_std = float(np.std(f0)) if f0.size else 0.0
    return {"f0_mean": f0_mean, "f0_median": f0_median, "f0_std": f0_std, "f0_series": f0.tolist()}


def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, Any]:
    if y.size == 0:
        return {"mfcc_mean": [0.0] * n_mfcc, "mfcc_std": [0.0] * n_mfcc}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return {"mfcc_mean": np.mean(mfcc, axis=1).tolist(), "mfcc_std": np.std(mfcc, axis=1).tolist()}


def compute_energy_stats(y: np.ndarray) -> Dict[str, Any]:
    if y.size == 0:
        return {"rms_mean": 0.0, "rms_std": 0.0}
    rms = librosa.feature.rms(y=y).squeeze(0)
    return {"rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms))}


def compute_jitter_shimmer(y: np.ndarray, sr: int) -> Dict[str, Any]:
    if y.size == 0:
        return {"jitter_local": 0.0, "shimmer_local": 0.0}
    if _HAS_PARSELMOUTH:
        try:
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            pitch = snd.to_pitch()
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = float(parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 75, 500, 1.3, 1.6))
            shimmer_local = float(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6, 0.03, 0.45))
            return {"jitter_local": jitter_local, "shimmer_local": shimmer_local}
        except Exception:
            pass
    # Heuristic fallback: use f0 variability and amplitude variability
    f0 = compute_f0(y, sr)["f0_series"]
    jitter = float(np.std(np.diff(f0)) / (np.mean(f0) + 1e-6)) if len(f0) > 2 else 0.0
    rms = librosa.feature.rms(y=y).squeeze(0)
    shimmer = float(np.std(np.diff(rms)) / (np.mean(rms) + 1e-6)) if len(rms) > 2 else 0.0
    return {"jitter_local": jitter, "shimmer_local": shimmer}


def compute_speech_rate_and_pauses(
    segment_text: str,
    start_s: float,
    end_s: float,
    words: Optional[List[Dict[str, Any]]] = None,
    pause_threshold: float = 0.3,
) -> Dict[str, Any]:
    dur = max(end_s - start_s, 1e-6)
    tokens = [t for t in segment_text.strip().split() if t]
    words_per_sec = len(tokens) / dur
    pauses = []
    if words and len(words) > 1:
        ws = sorted(words, key=lambda w: w["start"])
        for i in range(1, len(ws)):
            gap = float(ws[i]["start"]) - float(ws[i - 1]["end"])
            if gap > pause_threshold:
                pauses.append(gap)
    # Also consider inter-segment pauses in the pipeline by comparing consecutive segments.
    return {
        "speech_rate_wps": float(words_per_sec),
        "pause_count": int(len(pauses)),
        "avg_pause_s": float(np.mean(pauses)) if pauses else 0.0,
        "long_pause_ratio": float(len([p for p in pauses if p >= 0.5])) / max(len(pauses), 1) if pauses else 0.0,
        "filled_pause_count": int(sum(1 for t in tokens if t.lower().strip(",.?!") in {"um", "uh", "erm", "hmm"})),
    }


def extract_prosody_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    feats = {}
    feats.update(compute_f0(y, sr))
    feats.update(compute_mfcc(y, sr))
    feats.update(compute_energy_stats(y))
    feats.update(compute_jitter_shimmer(y, sr))
    return feats


