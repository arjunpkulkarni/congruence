from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


class OpticalFlowComputer:
    """
    Computes dense optical flow magnitude between consecutive frames for motion consistency.
    """
    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self._prev_gray = None

    def __call__(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flow_mag = None
        flow_vec = None
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow_vec = flow
            fx, fy = flow[..., 0], flow[..., 1]
            flow_mag = np.sqrt(fx * fx + fy * fy)
        self._prev_gray = gray
        return {
            "flow": flow_vec,
            "magnitude": flow_mag,
        }


