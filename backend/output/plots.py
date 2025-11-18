from typing import List, Dict, Any, Optional
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_timeline(fused: List[Dict[str, Any]], out_path: str):
    """
    Plot incongruence and stress over time.
    """
    if not fused:
        return
    t = np.array([r["time_s"] for r in fused], dtype=np.float32)
    incong = np.array([r.get("incongruence", 0.0) for r in fused], dtype=np.float32)
    stress = np.array([((r.get("E") or {}).get("stress", 0.0)) for r in fused], dtype=np.float32)

    plt.figure(figsize=(12, 4))
    plt.plot(t, incong, label="Incongruence (0–1)", color="#d62728")
    plt.plot(t, stress, label="Stress (audio)", color="#1f77b4", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.05)
    plt.title("Session Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_congruence_heatmap(fused: List[Dict[str, Any]], out_path: str):
    """
    Heatmap over time for pairwise scores:
      C_micro_macro, C_audio_text, C_macro_text, C_micro_text
    """
    if not fused:
        return
    labels = ["C_micro_macro", "C_audio_text", "C_macro_text", "C_micro_text"]
    data = []
    for name in labels:
        row = [float(((r.get("pairwise") or {}).get(name, 0.0))) for r in fused]
        data.append(row)
    mat = np.array(data, dtype=np.float32)

    plt.figure(figsize=(12, 3))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Score (0–1)")
    plt.xlabel("Frame index")
    plt.title("Congruence Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


