from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import bisect


@dataclass
class FrameClock:
    fps: float
    start_time_s: float = 0.0

    def t_of_frame(self, frame_index: int) -> float:
        return self.start_time_s + frame_index / max(self.fps, 1e-6)


def _index_by_time(segments: List[Dict[str, Any]]) -> List[float]:
    # Use segment midpoints for nearest neighbor lookup
    return [0.5 * (float(s.get("start", 0.0)) + float(s.get("end", 0.0))) for s in segments]


def lookup_segment_for_time(segments: List[Dict[str, Any]], time_s: float) -> Optional[Dict[str, Any]]:
    if not segments:
        return None
    mids = _index_by_time(segments)
    i = bisect.bisect_left(mids, time_s)
    if i <= 0:
        return segments[0]
    if i >= len(mids):
        return segments[-1]
    # choose nearest
    if abs(mids[i] - time_s) < abs(mids[i - 1] - time_s):
        return segments[i]
    return segments[i - 1]


def align_streams_to_frames(
    num_frames: int,
    frame_clock: FrameClock,
    audio_segments: List[Dict[str, Any]],
    text_segments: List[Dict[str, Any]],
    macro_stream: Optional[List[Dict[str, Any]]] = None,
    micro_stream: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Produces per-frame aligned records by looking up nearest segment per modality.
    Expected segment dicts:
      - audio segment: {"start","end","emotion":{...}, "prosody":{...}}
      - text segment: {"start","end","analysis":{...}, "text": "..."}
      - macro/micro stream items may already be per-frame; if provided, they are directly indexed
    """
    aligned: List[Dict[str, Any]] = []
    for t in range(num_frames):
        time_s = frame_clock.t_of_frame(t)
        audio_seg = lookup_segment_for_time(audio_segments, time_s)
        text_seg = lookup_segment_for_time(text_segments, time_s)
        macro_item = macro_stream[t] if (macro_stream and t < len(macro_stream)) else None
        micro_item = micro_stream[t] if (micro_stream and t < len(micro_stream)) else None
        aligned.append({
            "t": t,
            "time_s": time_s,
            "audio": audio_seg,
            "text": text_seg,
            "macro": macro_item,
            "micro": micro_item,
        })
    return aligned


