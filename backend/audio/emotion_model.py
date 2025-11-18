from typing import Dict, Any, Optional, List, Tuple

import numpy as np

_HF_AVAILABLE = False
try:
    from transformers import pipeline  # type: ignore
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

try:
    import torch  # type: ignore
    import torchaudio  # type: ignore
    _TORCH_AUDIO = True
except Exception:
    _TORCH_AUDIO = False


class EmotionModel:
    """
    Uses a HuggingFace wav2vec2 emotion model if available.
    Fallback: derive stress/arousal/valence/tension/hesitation/vocal_strain from prosody heuristics.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or "superb/wav2vec2-base-superb-er"
        self.device = device
        self._clf = None
        if _HF_AVAILABLE and _TORCH_AUDIO:
            try:
                self._clf = pipeline("audio-classification", model=self.model_name, device=0 if self._has_cuda() else -1)
            except Exception:
                self._clf = None

    def _has_cuda(self) -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def predict(
        self,
        y: np.ndarray,
        sr: int,
        prosody: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "categories": Optional[List[{"label": str, "score": float}]],
              "arousal": float [0,1],
              "valence": float [0,1],
              "stress": float [0,1],
              "tension": float [0,1],
              "hesitation": float [0,1],
              "vocal_strain": float [0,1],
            }
        """
        categories = None
        if self._clf is not None and y.size > 0:
            try:
                # torchaudio expects tensor or path; the pipeline can accept (array, sr)
                res = self._clf({"array": y.astype(np.float32), "sampling_rate": sr})
                # HF pipeline returns list of dicts
                if isinstance(res, list):
                    categories = [{"label": r["label"], "score": float(r["score"])} for r in res]
            except Exception:
                categories = None

        # Heuristic attributes using prosody
        f0_mean = float(prosody.get("f0_mean", 0.0))
        f0_std = float(prosody.get("f0_std", 0.0))
        rms_mean = float(prosody.get("rms_mean", 0.0))
        rms_std = float(prosody.get("rms_std", 0.0))
        jitter = float(prosody.get("jitter_local", 0.0))
        shimmer = float(prosody.get("shimmer_local", 0.0))
        speech_rate = float(prosody.get("speech_rate_wps", 0.0))
        long_pause_ratio = float(prosody.get("long_pause_ratio", 0.0))
        filled_pause_count = float(prosody.get("filled_pause_count", 0.0))
        avg_pause_s = float(prosody.get("avg_pause_s", 0.0))

        # Normalize with soft bounds
        def nz(x, a, b):
            x = (x - a) / (b - a + 1e-6)
            return float(np.clip(x, 0.0, 1.0))

        # Heuristic scores
        arousal = np.clip(0.5 * nz(rms_mean, 0.01, 0.2) + 0.5 * nz(f0_mean, 80.0, 280.0), 0.0, 1.0)
        stress = np.clip(0.4 * nz(f0_std, 5.0, 40.0) + 0.3 * nz(jitter, 0.0, 0.06) + 0.3 * nz(shimmer, 0.0, 0.08), 0.0, 1.0)
        tension = np.clip(0.5 * nz(rms_std, 0.0, 0.1) + 0.5 * stress, 0.0, 1.0)
        hesitation = np.clip(0.5 * nz(long_pause_ratio, 0.0, 1.0) + 0.3 * nz(avg_pause_s, 0.0, 1.2) + 0.2 * nz(filled_pause_count, 0.0, 3.0), 0.0, 1.0)
        vocal_strain = np.clip(0.5 * nz(f0_mean, 120.0, 320.0) + 0.5 * nz(shimmer, 0.0, 0.08), 0.0, 1.0)

        # Valence: use categories if we have them, else heuristic inverse of stress+tension
        if categories:
            # naive mapping: positive emotions to higher valence
            pos = sum(c["score"] for c in categories if c["label"].lower() in {"happy", "surprise", "joy", "pleasant-surprise"})
            neg = sum(c["score"] for c in categories if c["label"].lower() in {"anger", "sadness", "fear", "disgust"})
            valence = float(np.clip(0.5 + 0.5 * (pos - neg), 0.0, 1.0))
        else:
            valence = float(np.clip(1.0 - 0.6 * stress - 0.4 * tension, 0.0, 1.0))

        return {
            "categories": categories,
            "arousal": float(arousal),
            "valence": float(valence),
            "stress": float(stress),
            "tension": float(tension),
            "hesitation": float(hesitation),
            "vocal_strain": float(vocal_strain),
        }


