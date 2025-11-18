from typing import List, Dict, Any, Optional
import math


def _format_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def build_session_highlights(
    fused: List[Dict[str, Any]],
    top_k: int = 6,
    incongruence_threshold: float = 0.6,
) -> List[str]:
    """
    Generate short, non-diagnostic highlights for therapists.
    """
    if not fused:
        return []
    # Rank by incongruence, prefer stress spikes or suppression markers
    scored = []
    for r in fused:
        pair = r.get("pairwise") or {}
        boost = 0.0
        if float(pair.get("stress_spike", 0.0)) >= 1.0:
            boost += 0.1
        if float(pair.get("suppression_indicator", 0.0)) >= 1.0:
            boost += 0.1
        score = float(r.get("incongruence", 0.0)) + boost
        if score >= incongruence_threshold:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    highlights: List[str] = []
    for score, r in scored[:top_k]:
        ts = _format_ts(float(r.get("time_s", 0.0)))
        pair = r.get("pairwise") or {}
        E = r.get("E") or {}
        bits = []
        # Text declared state if present
        declared = E.get("text_declared_state")
        if declared:
            bits.append(f'verbal statement: "{declared}"')
        # Macro vs micro hints
        c_mm = float(pair.get("C_micro_macro", 0.0))
        c_mt = float(pair.get("C_macro_text", 0.0))
        c_at = float(pair.get("C_audio_text", 0.0))
        c_mtxt = float(pair.get("C_micro_text", 0.0))
        if c_mm > 0.5:
            bits.append("micro vs macro mismatch")
        if c_mt > 0.5:
            bits.append("macro vs text mismatch")
        if c_mtxt > 0.5:
            bits.append("micro vs text mismatch")
        if c_at > 0.5:
            bits.append("audio vs text mismatch")
        if float(pair.get("suppression_indicator", 0.0)) >= 1.0:
            bits.append("possible suppression (micro present + macro neutral)")
        if float(pair.get("stress_spike", 0.0)) >= 1.0:
            bits.append("stress spike detected")
        if not bits:
            bits.append("elevated incongruence")
        highlights.append(f"At {ts} – " + "; ".join(bits) + ".")
    return highlights


def summarize_linguistic_intentions(text_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"reassurance-seeking": 0, "avoidance": 0, "problem-solving": 0, "self-disclosure": 0, "neutral": 0}
    examples = {k: [] for k in counts.keys()}
    for seg in text_segments:
        a = (seg.get("analysis") or {})
        intent = str(a.get("verbal_intent", "neutral")).lower()
        if intent not in counts:
            intent = "neutral"
        counts[intent] += 1
        txt = seg.get("text", "")
        if txt and len(examples[intent]) < 3:
            examples[intent].append(txt)
    return {"intent_counts": counts, "examples": examples}


def build_report(
    fused: List[Dict[str, Any]],
    text_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a structured report dictionary. The caller can serialize to JSON.
    """
    highlights = build_session_highlights(fused)
    linguistic = summarize_linguistic_intentions(text_segments or [])
    return {
        "highlights": highlights,
        "linguistic_intentions": linguistic,
        "meta": {
            "frames": len(fused),
            "duration_s": float(fused[-1]["time_s"]) if fused else 0.0,
        },
    }


