import argparse
import json
from pathlib import Path

from .engine import fuse_per_frame, Weights


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True, help="Number of video frames to align")
    ap.add_argument("--fps", type=float, required=True, help="Video FPS")
    ap.add_argument("--audio", type=str, required=True, help="Audio JSON (from backend/audio/pipeline.py)")
    ap.add_argument("--text", type=str, required=True, help="Text JSON (from backend/transcript/pipeline.py)")
    ap.add_argument("--macro", type=str, default="", help="Optional macro per-frame JSON list [{'label','probs'}...]")
    ap.add_argument("--micro", type=str, default="", help="Optional micro per-frame JSON list")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    # weights
    ap.add_argument("--w1", type=float, default=0.30)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--w3", type=float, default=0.20)
    ap.add_argument("--w4", type=float, default=0.15)
    ap.add_argument("--w5", type=float, default=0.10)
    args = ap.parse_args()

    audio_segments = _load_json(args.audio)
    text_segments = _load_json(args.text)
    macro_stream = _load_json(args.macro) if args.macro else None
    micro_stream = _load_json(args.micro) if args.micro else None

    weights = Weights(
        w_micro_macro=args.w1,
        w_audio_text=args.w2,
        w_macro_text=args.w3,
        w_micro_text=args.w4,
        w_stress_suppression=args.w5,
    )

    outputs = fuse_per_frame(
        num_frames=args.frames,
        fps=args.fps,
        audio_segments=audio_segments,
        text_segments=text_segments,
        macro_stream=macro_stream,
        micro_stream=micro_stream,
        weights=weights,
    )

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            for o in outputs:
                f.write(json.dumps(o) + "\n")
        print(f"Wrote {len(outputs)} fused frames to {out_path}")
    else:
        for o in outputs:
            print(json.dumps(o))


if __name__ == "__main__":
    main()


