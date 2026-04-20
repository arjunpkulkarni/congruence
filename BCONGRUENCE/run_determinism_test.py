"""
Determinism test for the clinical notes pipeline (fast mode: transcription + LLM notes,
no facial analysis).

Usage: python run_determinism_test.py /path/to/video.mov

Produces:
  - ./test_results/run1_notes.json
  - ./test_results/run2_notes.json
  - ./test_results/run1_transcript.txt
  - ./test_results/run2_transcript.txt
  - ./test_results/comparison.json
  - console summary (timings + diff)
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env before importing services (they read OPENAI_API_KEY at module import time in some paths).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from app.services.transcription import transcribe_long_audio_chunked  # noqa: E402
from app.services.notes import generate_therapist_notes  # noqa: E402
from app.services.video_processing import extract_audio_with_ffmpeg, get_video_duration  # noqa: E402
from app.services.congruence_engine import build_session_summary, build_congruence_timeline  # noqa: E402
from app.services.analysis import merge_timelines  # noqa: E402


RESULTS_DIR = ROOT / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_once(label: str, audio_path: str, video_duration: float, patient_id: str) -> Dict[str, Any]:
    """Run one full fast-mode pass (transcription + LLM notes) and return outputs + timings."""
    section(f"{label}: transcribing + generating notes")

    timings: Dict[str, float] = {}

    t0 = time.time()
    transcript_text, transcript_segments = transcribe_long_audio_chunked(
        audio_path, "tiny", "en", True, 10
    )
    timings["transcription_s"] = time.time() - t0
    print(f"  transcription: {timings['transcription_s']:.2f}s "
          f"(chars={len(transcript_text or '')}, segments={len(transcript_segments or [])})")

    t1 = time.time()
    merged_timeline = merge_timelines(facial_timeline=None, audio_timeline=None)
    congruence_timeline = build_congruence_timeline(
        merged_timeline=merged_timeline,
        transcript_segments=transcript_segments,
        spikes=[],
        target_hz=10.0,
    )
    session_summary = build_session_summary(
        congruence_timeline=congruence_timeline,
        patient_id=patient_id,
        session_id=int(time.time()),
        transcript_segments=transcript_segments,
        actual_duration_seconds=video_duration,
    )
    timings["summary_build_s"] = time.time() - t1
    print(f"  summary build: {timings['summary_build_s']:.2f}s")

    t2 = time.time()
    notes: Dict[str, Any] | None = None
    if transcript_text and session_summary:
        notes = generate_therapist_notes(
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            session_summary=session_summary,
            patient_id=patient_id,
        )
    timings["notes_generation_s"] = time.time() - t2
    print(f"  notes generation: {timings['notes_generation_s']:.2f}s "
          f"(got_notes={notes is not None})")

    timings["total_s"] = sum(timings.values())
    print(f"  TOTAL: {timings['total_s']:.2f}s")

    # Strip the private attachment fields that generate_therapist_notes adds (makes diffing cleaner).
    clean_notes: Dict[str, Any] | None = None
    if notes is not None:
        clean_notes = {k: v for k, v in notes.items() if not k.startswith("_")}

    return {
        "label": label,
        "timings_s": timings,
        "transcript_text": transcript_text or "",
        "transcript_segments": transcript_segments or [],
        "notes": clean_notes,
    }


def hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def compare_runs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    section("COMPARISON: run 1 vs run 2")

    # Transcript diff
    transcript_equal = a["transcript_text"] == b["transcript_text"]
    t_ratio = difflib.SequenceMatcher(None, a["transcript_text"], b["transcript_text"]).ratio()

    print(f"  transcript identical:   {transcript_equal}")
    print(f"  transcript similarity:  {t_ratio:.4f}")
    print(f"    run1 chars: {len(a['transcript_text'])}")
    print(f"    run2 chars: {len(b['transcript_text'])}")

    # Notes diff
    notes_a = a.get("notes") or {}
    notes_b = b.get("notes") or {}
    notes_equal = notes_a == notes_b
    notes_a_hash = hash_json(notes_a)
    notes_b_hash = hash_json(notes_b)

    print(f"  notes identical:        {notes_equal}")
    print(f"  notes sha256 run1:      {notes_a_hash[:16]}…")
    print(f"  notes sha256 run2:      {notes_b_hash[:16]}…")

    # Per-section equality so we can pinpoint which parts drift.
    per_section_equal: Dict[str, Any] = {}
    all_keys = sorted(set(notes_a.keys()) | set(notes_b.keys()))
    for k in all_keys:
        va = notes_a.get(k)
        vb = notes_b.get(k)
        equal = va == vb
        per_section_equal[k] = equal
        if not equal:
            sa = json.dumps(va, sort_keys=True, ensure_ascii=False)
            sb = json.dumps(vb, sort_keys=True, ensure_ascii=False)
            ratio = difflib.SequenceMatcher(None, sa, sb).ratio()
            print(f"    • '{k}': DIFFERS (similarity {ratio:.3f})")
        else:
            print(f"    • '{k}': identical")

    # Timing diff
    print()
    print("  timings (seconds):")
    for phase in ("transcription_s", "summary_build_s", "notes_generation_s", "total_s"):
        va = a["timings_s"].get(phase, 0.0)
        vb = b["timings_s"].get(phase, 0.0)
        print(f"    {phase:22s}  run1={va:7.2f}  run2={vb:7.2f}  Δ={vb - va:+.2f}")

    return {
        "transcript_identical": transcript_equal,
        "transcript_similarity": t_ratio,
        "notes_identical": notes_equal,
        "notes_sha256_run1": notes_a_hash,
        "notes_sha256_run2": notes_b_hash,
        "per_section_equal": per_section_equal,
        "timings": {"run1": a["timings_s"], "run2": b["timings_s"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--patient-id", default="test-determinism-anger")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"ERROR: video not found: {video_path}")
        return 1

    section("SETUP")
    print(f"  video:        {video_path}")
    print(f"  patient_id:   {args.patient_id}")
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  size:         {size_mb:.1f} MB")
    video_duration = get_video_duration(video_path)
    print(f"  duration:     {video_duration:.2f}s ({video_duration/60:.2f} min)")

    # Extract audio once — audio extraction is a deterministic ffmpeg op; reusing
    # the same audio for both runs isolates the test to the parts that can drift
    # (transcription + LLM), while still exercising both of those per run.
    audio_path = str(RESULTS_DIR / "extracted_audio.wav")
    section("AUDIO EXTRACTION (shared across runs)")
    t_extract = time.time()
    extract_audio_with_ffmpeg(input_video_path=video_path, output_audio_path=audio_path, fast_mode=True)
    print(f"  audio extracted in {time.time() - t_extract:.2f}s → {audio_path}")

    run1 = run_once("RUN 1", audio_path, video_duration, args.patient_id)
    run2 = run_once("RUN 2", audio_path, video_duration, args.patient_id)

    # Persist artifacts
    (RESULTS_DIR / "run1_notes.json").write_text(json.dumps(run1["notes"], indent=2, ensure_ascii=False))
    (RESULTS_DIR / "run2_notes.json").write_text(json.dumps(run2["notes"], indent=2, ensure_ascii=False))
    (RESULTS_DIR / "run1_transcript.txt").write_text(run1["transcript_text"])
    (RESULTS_DIR / "run2_transcript.txt").write_text(run2["transcript_text"])
    (RESULTS_DIR / "run1_timings.json").write_text(json.dumps(run1["timings_s"], indent=2))
    (RESULTS_DIR / "run2_timings.json").write_text(json.dumps(run2["timings_s"], indent=2))

    comparison = compare_runs(run1, run2)
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    section("ARTIFACTS")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
