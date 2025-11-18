from typing import List, Dict, Any, Optional

import os


def transcribe_to_segments(
    audio_path: str,
    model_size: str = "small",
    use_faster_whisper: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns list of segments:
        {
          "start": float seconds,
          "end": float seconds,
          "text": str,
          "words": Optional[List[{"start": float, "end": float, "word": str}]]
        }
    Tries faster-whisper for better timestamps; falls back to openai/whisper.
    """
    segments: List[Dict[str, Any]] = []
    if use_faster_whisper:
        try:
            from faster_whisper import WhisperModel  # type: ignore
            model = WhisperModel(model_size, device="cuda" if _has_cuda() else "cpu")
            gen, info = model.transcribe(audio_path, vad_filter=True, word_timestamps=True)
            for s in gen:
                words = []
                if s.words:
                    for w in s.words:
                        words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
                segments.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text or "",
                    "words": words if words else None,
                })
            return segments
        except Exception:
            pass
    # Fallback: openai/whisper (word timestamps not guaranteed)
    try:
        import whisper  # type: ignore
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        for s in result.get("segments", []):
            segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": s.get("text", ""),
                "words": None,
            })
        return segments
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


