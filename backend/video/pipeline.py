from dataclasses import dataclass
from typing import Iterable, Generator, Optional, Dict, Any, Union

import cv2
import numpy as np

from .detectors import MediaPipeFaceAligner, AlignmentConfig
from .preprocess import denoise_pipeline, TemporalSmoother
from .optical_flow import OpticalFlowComputer


@dataclass
class PipelineConfig:
    output_size: int = 224
    use_clahe: bool = True
    gamma: float = 1.1
    temporal_alpha: float = 0.6


class VideoFacePipeline:
    """
    Processes a video frame iterator and yields normalized, aligned facial frames.
    Steps:
    - Detect face and landmarks (MediaPipe Face Mesh)
    - Align to canonical geometry and crop to output_size
    - Denoise: gamma, CLAHE, temporal smoothing
    - Compute optical flow for motion consistency
    """
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self._aligner = MediaPipeFaceAligner(AlignmentConfig(output_size=config.output_size))
        self._smoother = TemporalSmoother(alpha=config.temporal_alpha)
        self._flow = OpticalFlowComputer()

    def reset(self):
        self._smoother.reset()
        self._flow.reset()

    def process_frames(
        self,
        frames: Iterable[np.ndarray],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        frames: iterable of BGR frames (H, W, 3) np.uint8
        yields:
            {
                "face_frame": aligned and normalized BGR frame,
                "flow_magnitude": optional float32 array [H, W] matching output_size,
                "found": bool,
                "raw_aligned": aligned prior to denoise (for debugging),
                "index": t,
            }
        """
        self.reset()
        for t, frame in enumerate(frames):
            out: Dict[str, Any] = {"index": t, "found": False, "face_frame": None, "flow_magnitude": None}
            if frame is None:
                yield out
                continue

            aligned = self._aligner.detect_and_align(frame)
            if aligned is None:
                yield out
                continue

            out["found"] = True
            out["raw_aligned"] = aligned.copy()

            # Denoise
            denoised = denoise_pipeline(
                aligned,
                do_clahe=self.config.use_clahe,
                gamma=self.config.gamma,
                smoother=self._smoother,
            )

            # Optical flow on the aligned face
            flow_info = self._flow(denoised)
            flow_mag = flow_info["magnitude"]
            out["flow_magnitude"] = flow_mag
            out["face_frame"] = denoised

            yield out


def frames_from_opencv_source(source: Union[int, str]) -> Generator[np.ndarray, None, None]:
    """
    Utility generator to read frames from webcam (int device id) or a video file path.
    Yields BGR frames as np.uint8 arrays.
    """
    cap = cv2.VideoCapture(source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


