import argparse
import json
from pathlib import Path

from .plots import plot_timeline, plot_congruence_heatmap
from .report import build_report
from .snapshots import export_micro_snapshots


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", type=str, required=True, help="Path to fused per-frame JSON list or JSONL")
    ap.add_argument("--text", type=str, default="", help="Optional text segments JSON")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--video", type=str, default="", help="Optional raw video path for snapshots")
    ap.add_argument("--face-frames-dir", type=str, default="", help="Optional face frames directory for snapshots")
    ap.add_argument("--topk", type=int, default=6, help="Max snapshots/highlights to export")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read fused data (accept JSON or JSONL)
    fused_path = Path(args.fused)
    fused = []
    if fused_path.suffix.lower() == ".jsonl":
        with fused_path.open("r") as f:
            for line in f:
                fused.append(json.loads(line))
    else:
        fused = _load_json(args.fused)

    text_segments = _load_json(args.text) if args.text else []

    # Plots
    plot_timeline(fused, str(out_dir / "timeline.png"))
    plot_congruence_heatmap(fused, str(out_dir / "congruence_heatmap.png"))

    # Report
    report = build_report(fused, text_segments=text_segments)
    with (out_dir / "session_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    # Snapshots
    export_micro_snapshots(
        fused,
        out_dir=str(out_dir / "snapshots"),
        top_k=args.topk,
        video_path=args.video or None,
        face_frames_dir=args.face_frames_dir or None,
    )

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()


