"""Export a scientific figure of every recorded seed, including failed races."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/data/manifest.json").read_text())
tracks = list(dict.fromkeys(row["track"] for row in manifest["evaluation"]))
methods = list(dict.fromkeys(row["method"] for row in manifest["evaluation"]))
colors = {"HILEPP": "#176da5", "SAC": "#b67b30", "MPC": "#556b78"}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#bac4cc",
        "text.color": "#25343e",
        "axes.labelcolor": "#53636e",
        "xtick.color": "#53636e",
        "ytick.color": "#53636e",
        "svg.fonttype": "none",
        "svg.hashsalt": "six-car-circuit-v1",
    }
)
fig, axes = plt.subplots(
    len(tracks), 2, figsize=(11, 3.1 * len(tracks)), squeeze=False, layout="constrained"
)
for i, track in enumerate(tracks):
    rows = [row for row in manifest["evaluation"] if row["track"] == track]
    for j, (metric, title, unit) in enumerate(
        [("duration_s", "Recorded duration", "s"), ("progress_m", "Ego progress", "m")]
    ):
        ax = axes[i, j]
        for mi, method in enumerate(methods):
            selected = [r for r in rows if r["method"] == method]
            for ri, row in enumerate(selected):
                offset = (ri - (len(selected) - 1) / 2) * 0.12
                success = row["metrics"]["completed"]
                ax.scatter(
                    row["metrics"][metric],
                    mi + offset,
                    c=colors[method],
                    marker="o" if success else "x",
                    s=38,
                    zorder=3,
                    linewidths=1.4,
                )
        ax.set(
            yticks=range(len(methods)), yticklabels=methods, xlabel=unit, title=f"{track} · {title}"
        )
        ax.set_ylim(-0.5, len(methods) - 0.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e4e9ec", linewidth=0.6)
        ax.set_xlim(left=0)
legend = [
    Line2D([0], [0], marker="o", color="#516472", linestyle="", label="Completed 180 s"),
    Line2D([0], [0], marker="x", color="#516472", linestyle="", label="Early termination"),
]
fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
for extension in ["svg", "png"]:
    fig.savefig(
        root / "docs/assets" / f"evaluation.{extension}",
        dpi=170,
        facecolor="white",
        metadata={"Date": None} if extension == "svg" else None,
    )
print("Exported every evaluation seed to docs/assets/evaluation.svg and .png")

svg = root / "docs/assets/evaluation.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
