import argparse
import json
from pathlib import Path

from .pipeline import TranscriptSemanticPipeline, TranscriptPipelineConfig
from ..audio.whisper_chunks import transcribe_to_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="", help="Optional path to audio file for Whisper transcription")
    ap.add_argument("--segments", type=str, default="", help="Optional JSON file with precomputed segments")
    ap.add_argument("--provider", type=str, default="openai", help="openai|gemini|ollama")
    ap.add_argument("--model", type=str, default="", help="Model name (e.g., gpt-4o-mini, gemini-1.5-flash, llama3.1)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--whisper", type=str, default="small")
    ap.add_argument("--no-faster", action="store_true")
    ap.add_argument("--context", type=str, default="", help="Optional session context")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    args = ap.parse_args()

    cfg = TranscriptPipelineConfig(
        provider=args.provider,
        model=args.model or None,
        temperature=args.temperature,
        whisper_model_size=args.whisper,
        use_faster_whisper=not args.no_faster,
    )
    pipe = TranscriptSemanticPipeline(cfg)

    if args.segments:
        with open(args.segments, "r") as f:
            segments = json.load(f)
        outputs = pipe.run_on_segments(segments, context=args.context or None)
    else:
        if not args.audio:
            raise SystemExit("Provide --audio or --segments")
        outputs = pipe.run_on_audio(args.audio, context=args.context or None)

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w") as f:
            for o in outputs:
                f.write(json.dumps(o) + "\n")
        print(f"Wrote {len(outputs)} segments to {out_path}")
    else:
        for o in outputs:
            print(json.dumps(o))


if __name__ == "__main__":
    main()


