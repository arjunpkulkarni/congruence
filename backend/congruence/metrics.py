from typing import Dict, Any, Optional, Tuple
import numpy as np


def _cosine_dissimilarity(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    un = np.linalg.norm(u) + 1e-8
    vn = np.linalg.norm(v) + 1e-8
    sim = float(np.dot(u, v) / (un * vn))
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = p.astype(np.float32)
    q = q.astype(np.float32)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * (np.log(p + eps) / np.log(q + eps))))


def valence_gap(v1: float, v2: float) -> float:
    return float(abs(v1 - v2))


def suppression_indicator(micro_label: Optional[str], macro_label: Optional[str]) -> float:
    micro_present = micro_label is not None and micro_label.lower() not in ("", "neutral")
    macro_neutral = macro_label is not None and macro_label.lower() in ("neutral", "calm")
    return 1.0 if (micro_present and macro_neutral) else 0.0


def stress_spike(prev_stress: float, cur_stress: float, window_mean: float, threshold: float = 0.2) -> float:
    """
    Detect sudden rise in stress vs local mean; returns 1.0 if spike, else 0.0
    """
    delta = float(cur_stress - max(window_mean, prev_stress))
    return 1.0 if delta >= threshold else 0.0


def audio_text_kl(audio_dist: Dict[str, float], text_label: str) -> float:
    """
    Build a smoothed one-hot dist for text and compute KL(audio || text).
    """
    labels = sorted(set(list(audio_dist.keys()) + [text_label]))
    if not labels:
        return 0.0
    eps = 1e-6
    p = np.array([max(audio_dist.get(l, 0.0), 0.0) + eps for l in labels], dtype=np.float32)
    p = p / p.sum()
    q = np.array([eps for _ in labels], dtype=np.float32)
    # one-hot with smoothing
    if text_label in labels:
        idx = labels.index(text_label)
        q[idx] = 1.0
    q = q / q.sum()
    return _kl_divergence(p, q)


