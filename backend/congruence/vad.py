from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Approximate VAD for common emotions (0..1 scale)
_EMOTION_TO_VAD = {
    "happy": (0.90, 0.70, 0.80),
    "joy": (0.92, 0.72, 0.82),
    "sad": (0.20, 0.30, 0.30),
    "anger": (0.10, 0.80, 0.70),
    "angry": (0.10, 0.80, 0.70),
    "fear": (0.10, 0.90, 0.20),
    "disgust": (0.15, 0.60, 0.60),
    "surprise": (0.60, 0.90, 0.50),
    "neutral": (0.50, 0.50, 0.50),
    "calm": (0.70, 0.30, 0.60),
}


def emotion_label_to_vad(label: Optional[str]) -> Tuple[float, float, float]:
    if not label:
        return (0.5, 0.5, 0.5)
    key = label.lower().strip()
    return _EMOTION_TO_VAD.get(key, (0.5, 0.5, 0.5))


def distribution_to_vad(dist: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Convert a categorical distribution over emotions into a VAD vector by expectation.
    dist: {"happy": p, "sad": p, ...} (not necessarily summing to 1; will be normalized)
    """
    if not dist:
        return (0.5, 0.5, 0.5)
    labels = list(dist.keys())
    probs = np.array([max(float(dist[l]), 0.0) for l in labels], dtype=np.float32)
    s = probs.sum()
    if s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    vad = np.zeros(3, dtype=np.float32)
    for l, p in zip(labels, probs):
        v, a, d = emotion_label_to_vad(l)
        vad += p * np.array([v, a, d], dtype=np.float32)
    return (float(vad[0]), float(vad[1]), float(vad[2]))


def macro_item_to_vad(macro_item: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Expected macro_item:
      {"label": "happy", "probs": {"happy": 0.8, "neutral": 0.2}} OR {"label": "happy"}
    """
    if not macro_item:
        return (0.5, 0.5, 0.5)
    probs = macro_item.get("probs")
    if isinstance(probs, dict) and probs:
        return distribution_to_vad(probs)
    return emotion_label_to_vad(macro_item.get("label"))


def micro_item_to_vad(micro_item: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Similar structure as macro_item; micro expressions are short-lived
    """
    return macro_item_to_vad(micro_item)


def audio_segment_to_vad(audio_seg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Uses audio 'emotion' categories distribution if present; otherwise map arousal/valence to VAD space.
    Expected audio_seg["emotion"] has keys: arousal, valence, categories?
    """
    if not audio_seg:
        return (0.5, 0.5, 0.5)
    emo = audio_seg.get("emotion") or {}
    cats = emo.get("categories")
    if isinstance(cats, list) and cats:
        dist = {}
        for c in cats:
            dist[str(c.get("label", "")).lower()] = float(c.get("score", 0.0))
        return distribution_to_vad(dist)
    # Map valence/arousal to VAD directly; assume dominance ~ (1 - stress)
    v = float(emo.get("valence", 0.5))
    a = float(emo.get("arousal", 0.5))
    stress = float(emo.get("stress", 0.0))
    d = float(np.clip(1.0 - stress, 0.0, 1.0))
    return (v, a, d)


def text_segment_to_vad(text_seg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Use LLM text_emotion and sentiment as hints.
    """
    if not text_seg:
        return (0.5, 0.5, 0.5)
    analysis = text_seg.get("analysis") or {}
    label = str(analysis.get("text_emotion", "")).lower()
    v0, a0, d0 = emotion_label_to_vad(label)
    # adjust valence by sentiment
    sentiment = float(analysis.get("sentiment", 0.0))
    v = float(np.clip(0.5 * v0 + 0.5 * (sentiment * 0.5 + 0.5), 0.0, 1.0))
    return (v, a0, d0)


