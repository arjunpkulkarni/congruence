from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False


def _center_of_points(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def _compute_affine_from_landmarks(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute 2x3 affine transform mapping src_pts (Nx2) to dst_pts (Nx2) using least-squares.
    """
    assert src_pts.shape == dst_pts.shape and src_pts.shape[0] >= 3
    M, _ = cv2.estimateAffinePartial2D(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    return M


def _warp_and_crop(image: np.ndarray, M: np.ndarray, output_size: int) -> np.ndarray:
    aligned = cv2.warpAffine(image, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned


_LEFT_EYE_IDX = [33, 133, 159, 145]   # subset around left eye (MediaPipe Face Mesh)
_RIGHT_EYE_IDX = [362, 263, 386, 374] # subset around right eye
_MOUTH_IDX = [13, 14, 78, 308]        # upper/lower lip and corners


@dataclass
class AlignmentConfig:
    output_size: int = 224
    # canonical target geometry inside output square
    dst_left_eye: Tuple[float, float] = (0.35, 0.38)
    dst_right_eye: Tuple[float, float] = (0.65, 0.38)
    dst_mouth: Tuple[float, float] = (0.50, 0.72)


class MediaPipeFaceAligner:
    """
    Uses MediaPipe Face Mesh to detect landmarks and produce an aligned face crop.
    """
    def __init__(self, alignment: AlignmentConfig = AlignmentConfig()):
        if not _HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required for MediaPipeFaceAligner.")
        self.alignment = alignment
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_and_align(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns aligned face crop of shape (output_size, output_size, 3) or None if not found.
        """
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

        left_eye = _center_of_points(pts[_LEFT_EYE_IDX])
        right_eye = _center_of_points(pts[_RIGHT_EYE_IDX])
        mouth_center = _center_of_points(pts[_MOUTH_IDX])

        size = self.alignment.output_size
        dst = np.array([
            [self.alignment.dst_left_eye[0] * size, self.alignment.dst_left_eye[1] * size],
            [self.alignment.dst_right_eye[0] * size, self.alignment.dst_right_eye[1] * size],
            [self.alignment.dst_mouth[0] * size, self.alignment.dst_mouth[1] * size],
        ], dtype=np.float32)
        src = np.array([left_eye, right_eye, mouth_center], dtype=np.float32)

        M = _compute_affine_from_landmarks(src, dst)
        if M is None:
            return None

        return _warp_and_crop(bgr_image, M, size)


# Placeholder for RetinaFace integration (optional)
class RetinaFaceAligner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RetinaFaceAligner not implemented. Use MediaPipeFaceAligner for now.")


