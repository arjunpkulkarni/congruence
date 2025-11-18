from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _draw_caption(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    bar_h = max(int(0.12 * h), 24)
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(out, text, (8, h - int(bar_h / 2) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _read_frame_from_video(video_path: str, time_s: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(time_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _read_face_frame_from_dir(dir_path: str, t_index: int) -> Optional[np.ndarray]:
    # Try common filenames
    cands = [
        Path(dir_path) / f"frame_{t_index}.jpg",
        Path(dir_path) / f"{t_index}.jpg",
        Path(dir_path) / f"{t_index:06d}.jpg",
    ]
    for p in cands:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def export_micro_snapshots(
    fused: List[Dict[str, Any]],
    out_dir: str,
    top_k: int = 6,
    video_path: Optional[str] = None,
    face_frames_dir: Optional[str] = None,
):
    """
    Save snapshots for top-k incongruent moments where suppression is likely or micro vs text mismatch is high.
    Attempts to read a frame from video (if provided) or a precomputed face-frames directory.
    """
    _ensure_dir(Path(out_dir))
    candidates = []
    for r in fused:
        pair = r.get("pairwise") or {}
        score = float(r.get("incongruence", 0.0))
        # focus on suppression or micro-text mismatch as clinically salient
        if float(pair.get("suppression_indicator", 0.0)) >= 1.0 or float(pair.get("C_micro_text", 0.0)) > 0.5:
            candidates.append((score, r))
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = candidates[:top_k]

    for rank, (score, r) in enumerate(kept, start=1):
        t = int(r.get("t", 0))
        ts = float(r.get("time_s", 0.0))
        caption_bits = [f"incongruence={score:.2f}"]
        pair = r.get("pairwise") or {}
        if float(pair.get("suppression_indicator", 0.0)) >= 1.0:
            caption_bits.append("suppression")
        if float(pair.get("stress_spike", 0.0)) >= 1.0:
            caption_bits.append("stress spike")
        caption = " | ".join(caption_bits) + f" | t={ts:.2f}s"

        img = None
        if face_frames_dir:
            img = _read_face_frame_from_dir(face_frames_dir, t)
        if img is None and video_path:
            img = _read_frame_from_video(video_path, ts)
        if img is None:
            # fallback: blank placeholder
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        vis = _draw_caption(img, caption)
        out_path = Path(out_dir) / f"snapshot_{rank:02d}_t{t}.jpg"
        cv2.imwrite(str(out_path), vis)


