from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa

from .whisper_chunks import transcribe_to_segments
from .prosody import slice_audio, extract_prosody_features, compute_speech_rate_and_pauses
from .emotion_model import EmotionModel


@dataclass
class AudioPipelineConfig:
    whisper_model_size: str = "small"
    use_faster_whisper: bool = True
    target_sr: int = 16000


class AudioEmotionPipeline:
    def __init__(self, config: AudioPipelineConfig = AudioPipelineConfig(), emotion_model_name: str = "superb/wav2vec2-base-superb-er"):
        self.config = config
        self._emotion = EmotionModel(model_name=emotion_model_name)

    def run(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        High-level runner: transcribe, extract features per segment, produce paralinguistic outputs.
        Returns list over time segments (audio_emotion[t]).
        """
        segments = transcribe_to_segments(audio_path, model_size=self.config.whisper_model_size, use_faster_whisper=self.config.use_faster_whisper)
        y, sr = librosa.load(audio_path, sr=self.config.target_sr, mono=True)

        outputs: List[Dict[str, Any]] = []
        prev_end = 0.0
        for idx, seg in enumerate(segments):
            start = float(seg["start"])
            end = float(seg["end"])
            text = seg.get("text", "")
            words = seg.get("words")

            y_seg = slice_audio(y, sr, start, end)
            prosody = extract_prosody_features(y_seg, sr)
            prosody.update(compute_speech_rate_and_pauses(text, start, end, words=words))

            # Estimate inter-segment pause as well
            inter_pause = max(0.0, start - prev_end)
            prev_end = end
            prosody["inter_segment_pause_s"] = float(inter_pause)

            emo = self._emotion.predict(y_seg, sr, prosody)

            out: Dict[str, Any] = {
                "t_index": idx,
                "start": start,
                "end": end,
                "text": text,
                "prosody": prosody,
                "emotion": {
                    "stress": emo["stress"],
                    "arousal": emo["arousal"],
                    "valence": emo["valence"],
                    "tension": emo["tension"],
                    "hesitation": emo["hesitation"],
                    "vocal_strain": emo["vocal_strain"],
                    "categories": emo.get("categories"),
                },
            }
            outputs.append(out)
        return outputs


