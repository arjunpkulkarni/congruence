from typing import Optional, Deque
from collections import deque

import cv2
import numpy as np


def apply_gamma(image_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-3:
        return image_bgr
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    CLAHE on Y channel in YCrCb space to avoid color artifacts.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


class TemporalSmoother:
    """
    Simple exponential moving average (EMA) smoother for frames.
    Keeps a running smoothed frame to reduce temporal noise/flicker.
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._prev: Optional[np.ndarray] = None

    def reset(self):
        self._prev = None

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_f = frame_bgr.astype(np.float32)
        if self._prev is None:
            self._prev = frame_f
        else:
            self._prev = self.alpha * frame_f + (1.0 - self.alpha) * self._prev
        return np.clip(self._prev, 0, 255).astype(np.uint8)


def denoise_pipeline(
    frame_bgr: np.ndarray,
    do_clahe: bool = True,
    gamma: float = 1.0,
    smoother: Optional[TemporalSmoother] = None,
) -> np.ndarray:
    out = frame_bgr
    if gamma is not None:
        out = apply_gamma(out, gamma=gamma)
    if do_clahe:
        out = apply_clahe(out)
    if smoother is not None:
        out = smoother(out)
    return out


