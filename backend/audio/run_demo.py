import argparse
import json
from pathlib import Path

from .pipeline import AudioEmotionPipeline, AudioPipelineConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str, help="Path to audio file (wav/mp3/m4a)")
    ap.add_argument("--model", type=str, default="superb/wav2vec2-base-superb-er", help="HF emotion model name")
    ap.add_argument("--whisper", type=str, default="small", help="Whisper model size (tiny/base/small/medium/large)")
    ap.add_argument("--no-faster", action="store_true", help="Disable faster-whisper and use openai/whisper")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    args = ap.parse_args()

    cfg = AudioPipelineConfig(whisper_model_size=args.whisper, use_faster_whisper=not args.no_faster, target_sr=16000)
    pipe = AudioEmotionPipeline(cfg, emotion_model_name=args.model)
    outputs = pipe.run(args.audio)

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


