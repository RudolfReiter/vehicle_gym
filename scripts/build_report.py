"""Export every evaluation seed, keeping actor execution modes separate."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/data/manifest.json").read_text())
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
modes = list(dict.fromkeys(row.get("execution") for row in manifest["evaluation"]))
for mode in modes:
    records = [row for row in manifest["evaluation"] if row.get("execution") == mode]
    tracks = list(dict.fromkeys(row["track"] for row in records))
    methods = list(dict.fromkeys(row["method"] for row in records))
    fig, axes = plt.subplots(
        len(tracks), 2, figsize=(11, 3.1 * len(tracks)), squeeze=False, layout="constrained"
    )
    for i, track in enumerate(tracks):
        rows = [row for row in records if row["track"] == track]
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
                yticks=range(len(methods)),
                yticklabels=methods,
                xlabel=unit,
                title=f"{track} · {title}",
            )
            ax.set_ylim(-0.5, len(methods) - 0.5)
            ax.invert_yaxis()
            ax.grid(axis="x", color="#e4e9ec", linewidth=0.6)
            maximum = max(row["metrics"][metric] for row in records)
            ax.set_xlim(0, max(1, maximum * 1.07))
    legend = [
        Line2D([0], [0], marker="o", color="#516472", linestyle="", label="Completed 180 s"),
        Line2D([0], [0], marker="x", color="#516472", linestyle="", label="Early termination"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
    stem = "evaluation" + ("-" + mode if mode else "")
    for extension in ["svg", "png"]:
        fig.savefig(
            root / "docs/assets" / f"{stem}.{extension}",
            dpi=170,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
    svg = root / "docs/assets" / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Exported {stem}: all recorded seeds, with early terminations marked.")
