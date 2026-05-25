"""Plot validation and leaderboard NDCG@5 across the model stages.

Run from the repo root with `python scripts/plot_score_progression.py`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["Baseline", "Boost", "Shockwave", "Afterburner", "Overdrive", "Velocity", "Supernova"]
VALIDATION = [0.4003, 0.4026, 0.4063, 0.4101, 0.4149, 0.4165, 0.4197]
PUBLIC = [0.40089, 0.40474, 0.4080, 0.41112, 0.41575, 0.41697, 0.42060]
PRIVATE_FINAL = 0.4221

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "figures" / "score_progression.png"


def main():
    x = range(len(STAGES))
    last = len(STAGES) - 1

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.plot(x, VALIDATION, marker="o", color="#3b6ea5", label="Validation")
    ax.plot(x, PUBLIC, marker="s", color="#d1782b", label="Public leaderboard")
    ax.scatter([last], [PRIVATE_FINAL], marker="*", s=180, color="#2a8c4a",
               zorder=3, label="Private leaderboard (final)")

    labels = [
        (VALIDATION[-1], f"{VALIDATION[-1]:.4f}", "#3b6ea5"),
        (PUBLIC[-1], f"{PUBLIC[-1]:.5f}", "#d1782b"),
        (PRIVATE_FINAL, f"{PRIVATE_FINAL:.4f}", "#2a8c4a"),
    ]
    for y, text, color in labels:
        ax.annotate(text, (last, y), xytext=(10, 0), textcoords="offset points",
                    va="center", color=color)

    ax.set_xticks(list(x), STAGES)
    ax.set_xlim(-0.3, last + 0.95)
    ax.set_ylabel("NDCG@5")
    ax.set_title("NDCG@5 by model stage")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
