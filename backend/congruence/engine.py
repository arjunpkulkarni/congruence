from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .alignment import align_streams_to_frames, FrameClock
from .vad import macro_item_to_vad, micro_item_to_vad, audio_segment_to_vad, text_segment_to_vad, emotion_label_to_vad
from .metrics import _cosine_dissimilarity, valence_gap, audio_text_kl, suppression_indicator, stress_spike


@dataclass
class Weights:
    w_micro_macro: float = 0.30
    w_audio_text: float = 0.25
    w_macro_text: float = 0.20
    w_micro_text: float = 0.15
    w_stress_suppression: float = 0.10


def _to_vec(vad_tuple):
    return np.array([vad_tuple[0], vad_tuple[1], vad_tuple[2]], dtype=np.float32)


def _audio_dist_from_segment(audio_seg: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not audio_seg:
        return {}
    emo = audio_seg.get("emotion") or {}
    dist = {}
    cats = emo.get("categories")
    if isinstance(cats, list):
        for c in cats:
            dist[str(c.get("label", "")).lower()] = float(c.get("score", 0.0))
    return dist


def fuse_per_frame(
    num_frames: int,
    fps: float,
    audio_segments: List[Dict[str, Any]],
    text_segments: List[Dict[str, Any]],
    macro_stream: Optional[List[Dict[str, Any]]] = None,
    micro_stream: Optional[List[Dict[str, Any]]] = None,
    weights: Weights = Weights(),
) -> List[Dict[str, Any]]:
    """
    Returns list of per-frame dicts with E(t) and incongruence.
    Required inputs:
      - audio_segments: from AudioEmotionPipeline
      - text_segments: from TranscriptSemanticPipeline
    Optional:
      - macro_stream: [{"label": str, "probs": {...}}, ...] length >= num_frames
      - micro_stream: same structure as macro_stream, per-frame
    """
    frame_clock = FrameClock(fps=fps, start_time_s=0.0)
    aligned = align_streams_to_frames(
        num_frames=num_frames,
        frame_clock=frame_clock,
        audio_segments=audio_segments,
        text_segments=text_segments,
        macro_stream=macro_stream,
        micro_stream=micro_stream,
    )

    outputs: List[Dict[str, Any]] = []
    prev_stress = 0.0
    mean_window: List[float] = []
    for row in aligned:
        t = row["t"]
        macro_v = _to_vec(macro_item_to_vad(row.get("macro") or {}))
        micro_v = _to_vec(micro_item_to_vad(row.get("micro") or {}))
        audio_v = _to_vec(audio_segment_to_vad(row.get("audio") or {}))
        text_v = _to_vec(text_segment_to_vad(row.get("text") or {}))

        # stress from audio if available
        audio_emo = (row.get("audio") or {}).get("emotion") or {}
        stress = float(audio_emo.get("stress", 0.0))
        mean_window.append(stress)
        if len(mean_window) > 30:
            mean_window.pop(0)
        window_mean = float(np.mean(mean_window)) if mean_window else 0.0
        spike = stress_spike(prev_stress, stress, window_mean)
        prev_stress = stress

        # pairwise congruence
        c_micro_macro = _cosine_dissimilarity(micro_v, macro_v)
        c_macro_text = valence_gap(float(macro_v[0]), float(text_v[0]))
        c_micro_text = valence_gap(float(micro_v[0]), float(text_v[0]))
        audio_dist = _audio_dist_from_segment(row.get("audio"))
        text_label = ((row.get("text") or {}).get("analysis") or {}).get("text_emotion") or "neutral"
        c_audio_text = audio_text_kl(audio_dist, str(text_label).lower())

        # suppression indicator
        micro_label = (row.get("micro") or {}).get("label")
        macro_label = (row.get("macro") or {}).get("label")
        suppress = suppression_indicator(micro_label, macro_label)

        incongruence = (
            weights.w_micro_macro * c_micro_macro +
            weights.w_audio_text * c_audio_text +
            weights.w_macro_text * c_macro_text +
            weights.w_micro_text * c_micro_text +
            weights.w_stress_suppression * (spike * suppress)
        )
        incongruence = float(np.clip(incongruence, 0.0, 1.0))

        outputs.append({
            "t": t,
            "time_s": row["time_s"],
            "E": {
                "macro_vad": macro_v.tolist(),
                "micro_vad": micro_v.tolist(),
                "audio_vad": audio_v.tolist(),
                "text_vad": text_v.tolist(),
                "text_declared_state": ((row.get("text") or {}).get("analysis") or {}).get("text_declared_state"),
                "stress": stress,
            },
            "pairwise": {
                "C_micro_macro": c_micro_macro,
                "C_audio_text": c_audio_text,
                "C_macro_text": c_macro_text,
                "C_micro_text": c_micro_text,
                "stress_spike": float(spike),
                "suppression_indicator": float(suppress),
            },
            "incongruence": incongruence,
        })
    return outputs


