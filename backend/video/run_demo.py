import argparse
import cv2
from pathlib import Path

from .pipeline import VideoFacePipeline, PipelineConfig, frames_from_opencv_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="Camera index or path to video file")
    ap.add_argument("--size", type=int, default=224, help="Output face crop size")
    ap.add_argument("--gamma", type=float, default=1.1, help="Gamma correction")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = VideoFacePipeline(
        PipelineConfig(
            output_size=args.size,
            use_clahe=not args.no_clahe,
            gamma=args.gamma,
            temporal_alpha=0.6,
        )
    )

    for out in pipeline.process_frames(frames_from_opencv_source(source)):
        if not out["found"]:
            cv2.imshow("face_frame", 255 * (0).to_bytes(1, "big"))  # placeholder
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        face = out["face_frame"]
        vis = face.copy()
        # optionally overlay flow magnitude as heatmap
        if out["flow_magnitude"] is not None:
            flow_mag = out["flow_magnitude"]
            flow_norm = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            flow_color = cv2.applyColorMap(flow_norm, cv2.COLORMAP_TURBO)
            vis = cv2.addWeighted(vis, 0.75, flow_color, 0.25, 0.0)
        cv2.imshow("face_frame", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


