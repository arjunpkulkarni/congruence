from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..audio.whisper_chunks import transcribe_to_segments
from .schema import TranscriptSegment, SegmentWithAnalysis, TextSemanticOutput
from .llm_classifier import LLMClassifier


@dataclass
class TranscriptPipelineConfig:
    provider: str = "openai"  # "openai" | "gemini" | "ollama"
    model: Optional[str] = None
    temperature: float = 0.0
    whisper_model_size: str = "small"
    use_faster_whisper: bool = True
    max_chars_per_segment: int = 800  # truncate long segments


class TranscriptSemanticPipeline:
    def __init__(self, cfg: TranscriptPipelineConfig = TranscriptPipelineConfig()):
        self.cfg = cfg
        self._clf = LLMClassifier(provider=cfg.provider, model=cfg.model, temperature=cfg.temperature)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.cfg.max_chars_per_segment:
            return text
        return text[: self.cfg.max_chars_per_segment] + " ..."

    def run_on_segments(self, segments: List[TranscriptSegment], context: Optional[str] = None) -> List[SegmentWithAnalysis]:
        outputs: List[SegmentWithAnalysis] = []
        for i, seg in enumerate(segments):
            text = self._truncate(seg.get("text", "") or "")
            analysis: TextSemanticOutput = self._clf.analyze(text, context=context)
            outputs.append({
                "t_index": i,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
                "analysis": analysis,
            })
        return outputs

    def run_on_audio(self, audio_path: str, context: Optional[str] = None) -> List[SegmentWithAnalysis]:
        segments = transcribe_to_segments(audio_path, model_size=self.cfg.whisper_model_size, use_faster_whisper=self.cfg.use_faster_whisper)
        return self.run_on_segments(segments, context=context)


