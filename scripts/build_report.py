# Build the research page and scientific figures from recorded evaluation data.
from pathlib import Path
import json
import statistics
import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "docs/data/experiment.json").read_text())
rl = data["kind"] == "hilepp"
name = "HILEPP" if rl else "vehicle_gym"
repo = "hilepp-recreated" if rl else "vehicle_gym"
paper = "https://ieeexplore.ieee.org/document/10178143"
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#526170",
        "text.color": "#202d3b",
        "svg.fonttype": "none",
        "svg.hashsalt": "vehicle-research",
    }
)
methods = [
    ("baseline", "Fixed MPP", "#72877c"),
    ("initial", "Initial policy", "#b44819"),
    ("learned", "Trained SAC", "#2166ac"),
]
if rl:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    validation = data["training"]["validation"]
    axes[0].plot(
        [v["step"] for v in validation],
        [v["return"] for v in validation],
        "o-",
        color="#2166ac",
        linewidth=2,
    )
    axes[0].set(
        title="Checkpoint selection on validation seeds",
        xlabel="Training interactions",
        ylabel="Mean episode return",
    )
    axes[0].grid(axis="y", color="#e3e9ee")
    x = np.arange(3)
    for j, (method, label, color) in enumerate(methods):
        series = [
            [
                r["return"]
                for r in data["evaluation"]
                if r["scenario"] == scenario["id"] and r["method"] == method
            ]
            for scenario in data["scenarios"]
        ]
        means = [np.mean(values) for values in series]
        axes[1].bar(x + (j - 1) * 0.24, means, 0.22, label=label, color=color, alpha=0.85)
        for i, values in enumerate(series):
            axes[1].scatter(
                x[i] + (j - 1) * 0.24 + np.linspace(-0.055, 0.055, len(values)),
                values,
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
    axes[1].set(
        xticks=x,
        xticklabels=[s["id"].capitalize() for s in data["scenarios"]],
        title="Held-out returns · five seeds per task",
        ylabel="Episode return",
    )
    axes[1].legend(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="none",
        fontsize=10,
        loc="lower right",
    )
    rows = []
    for scenario in data["scenarios"]:
        for method, label, _ in methods:
            values = [
                r
                for r in data["evaluation"]
                if r["scenario"] == scenario["id"] and r["method"] == method
            ]
            scores = [r["return"] for r in values]
            rows.append(
                f"<tr><td>{scenario['id'].capitalize()}</td><td>{label}</td>"
                f"<td>{statistics.mean(scores):.1f} ± {statistics.stdev(scores):.1f}</td>"
                f"<td>{statistics.mean(r['final_rank'] for r in values):.1f}</td>"
                f"<td>{sum(r['completed'] for r in values)} / {len(values)}</td>"
                f"<td>{sum(r['solver_failures'] for r in values)}</td>"
                f"<td>{sum(r['collision'] for r in values)}</td></tr>"
            )
    table = (
        "<thead><tr><th>Task</th><th>Method</th><th>Return · mean ± sample SD</th><th>Final rank ↓</th><th>Completed</th><th>Solver failures</th><th>Collisions</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody>"
    )
    base = statistics.mean(r["return"] for r in data["evaluation"] if r["method"] == "baseline")
    learned = statistics.mean(r["return"] for r in data["evaluation"] if r["method"] == "learned")
    gain = (learned / base - 1) * 100
    result_intro = f"The recreated SAC policy achieved a mean held-out return of {learned:.1f}, compared with {base:.1f} for the fixed-reference planner (+{gain:.1f}%). These are small-scale results from this implementation, not the paper’s reported benchmark."
    result_detail = "Each method uses the same five held-out seeds (10000–10004) for each of three tasks: 45 episodes in total. The replay always shows seed 10000. Dots show individual test episodes; bars show means. Validation uses three separate seeds (8000–8002). The best validation checkpoint was selected before test evaluation."
    figure_alt = "Actual validation learning curve and held-out episode returns for fixed MPP, initial policy, and trained SAC across three driving tasks"
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for j, scenario in enumerate(data["scenarios"]):
        values = [r["metrics"]["return"] for r in scenario["runs"]]
        axes[0].bar(
            np.arange(3) + (j - 1) * 0.24,
            values,
            0.22,
            label=scenario["id"].capitalize(),
            color=["#2166ac", "#72877c", "#b44819"][j],
        )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=["Center", "Left", "Right"],
        ylabel="Episode return",
        title="Changing the reference changes the drive",
    )
    axes[0].legend(frameon=False, fontsize=10)
    for run in data["scenarios"][0]["runs"]:
        axes[1].plot(run["time"], [s[1] for s in run["states"]], label=run["name"], linewidth=2)
    axes[1].set(
        xlabel="Time (s)", ylabel="Lateral position (m)", title="Overtaking · executed trajectories"
    )
    axes[1].legend(frameon=False, fontsize=10)
    rows = []
    for scenario in data["scenarios"]:
        for run in scenario["runs"]:
            m = run["metrics"]
            rows.append(
                f"<tr><td>{scenario['id'].capitalize()}</td><td>{run['name']}</td><td>{m['return']:.1f}</td>"
                f"<td>{m['final_rank']}</td><td>{'Yes' if m['completed'] else 'No'}</td>"
                f"<td>{m['solver_failures']}</td><td>{int(m['collision'])}</td></tr>"
            )
    table = (
        "<thead><tr><th>Task</th><th>Reference</th><th>Return</th><th>Final rank ↓</th><th>Completed</th><th>Solver failures</th><th>Collisions</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody>"
    )
    result_intro = "Nine recorded simulations compare center, left, and right references in overtaking, blocking, and mixed traffic. Every trajectory below comes from the nonlinear planner and the original five-state vehicle model."
    result_detail = "All nine runs use seed 101 and a 16-second episode limit. This is a deterministic demonstration of the interface, not a statistical benchmark. Collisions use an axis-aligned Frenet body approximation sampled every 0.05 seconds."
    figure_alt = "Episode returns for nine fixed-reference runs and actual lateral positions in the overtaking example"
for axis in axes:
    axis.grid(axis="y", color="#e3e9ee", zorder=0)
fig.savefig(ROOT / "docs/assets/evaluation.svg", bbox_inches="tight", metadata={"Date": None})
svg_path = ROOT / "docs/assets/evaluation.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
fig.savefig(ROOT / "docs/assets/evaluation.png", dpi=160, bbox_inches="tight")
plt.close(fig)

publication = f'''<aside class="publication"><span class="tag">European Control Conference · ECC 2023</span>
<h3>A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing</h3>
<p class="muted">Rudolf Reiter, Jasper Hoffmann,<br>Joschka Boedecker &amp; Moritz Diehl</p>
<p class="small muted">Bucharest, Romania · 13–16 June 2023<br>DOI: 10.23919/ECC57647.2023.10178143</p>
<div class="links"><a href="{paper}">Read the paper ↗</a><a href="https://publications.syscop.de/Reiter2023.pdf">Author PDF</a></div></aside>'''
hero_title = (
    "Learning race strategy.<br>Planning vehicle motion."
    if rl
    else "Vehicle dynamics.<br>Predictive planning.<br>Research you can run."
)
hero_lead = (
    "A recreated hierarchical racing policy: reinforcement learning selects speed and lateral references; nonlinear model predictive control turns them into a drive."
    if rl
    else "An improved vehicle_gym with a portable nonlinear planner, reproducible multi-car scenarios, and a clear view of what the optimizer predicts."
)
notice = (
    "<strong>Research recreation.</strong> This is a new implementation of the paper’s policy interface, built on vehicle_gym. It is not the original HILEPP training repository. The included policy was trained afresh; it is not an original paper checkpoint."
    if rl
    else "<strong>Improved original repository.</strong> This branch extends Rudolf Reiter’s vehicle_gym and preserves its history. The portable planner and the experiments on this page are new implementations; they do not reproduce the original paper’s full evaluation."
)
flow = (
    [
        ("Observe", "22 values describe curvature, ego motion, and three opponents."),
        (
            "Learn a reference",
            "SAC chooses desired speed and lateral position. HILEPP-II can also choose two cost weights.",
        ),
        (
            "Solve the motion plan",
            "vehicle_gym expands the reference and solves a nonlinear constrained problem.",
        ),
        (
            "Execute & observe",
            "Apply the first control for 0.25 s, measure the new state, and repeat.",
        ),
    ]
    if rl
    else [
        (
            "Model",
            "The original five-state Frenet bicycle model includes speed, steering, rolling resistance, and drag.",
        ),
        (
            "Reference",
            "Set desired speed and lateral position through one typed reference interface.",
        ),
        (
            "Optimize",
            "Solve a 3-second multiple-shooting problem with road, actuator, and obstacle constraints.",
        ),
        (
            "Inspect",
            "Replay actual traffic, compare fixed forecasts, and inspect numerical feasibility.",
        ),
    ]
)
flow_markup = "".join(
    f'<article><p class="number">0{i + 1}</p><h3>{title}</h3><p>{text}</p></article>'
    for i, (title, text) in enumerate(flow)
)
command = (
    "# Put the improved vehicle_gym checkout beside this repository\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -e ../vehicle_gym -e '.[train,dev]'\npython -m hilepp.evaluate --checkpoint-dir checkpoints\npython -m hilepp.train --steps 4000 --seed 7 --output checkpoints/new-run\npython -m http.server 8000 --directory docs"
    if rl
    else "python3 -m venv .venv\nsource .venv/bin/activate\npip install -e '.[visualization,dev]'\npython -m vehiclegym.demo\npython -m pytest\npython -m http.server 8000 --directory docs"
)
research_text = (
    "The policy chooses strategy through reference parameters, while the motion planner retains the vehicle model and constraints. HILEPP-I outputs desired speed and lateral position. HILEPP-II additionally sets the speed and lateral cost weights. Both interfaces are implemented; the included trained checkpoint uses HILEPP-I."
    if rl
    else "The research separates high-level strategic choices from low-level motion planning. vehicle_gym provides the dynamics and planner interface. Its new portable backend makes this separation runnable without compiling acados; the original acados implementation remains in the repository."
)
impl_points = (
    "<li>Actual SAC: a squashed Gaussian actor, twin critics, target critics, and entropy regularization.</li><li>4,000 training interactions, one training seed, 64-step episodes; NumPy-only checkpoint inference.</li><li>Paper-style observations and progress/rank reward, with a −10 failure penalty added for this recreation.</li>"
    if rl
    else "<li>Lazy optional imports and complete package discovery, including track resources.</li><li>Independent planning snapshots, full reference-shape validation, and separating-axis rectangle collision checks.</li><li>Examples separated from regression tests; corrected module names retain compatibility imports.</li>"
)
limits = "<li>CasADi/IPOPT replaces the original acados/HPIPM solver in these runs.</li><li>Opponents use lane tracking and car following; the ego predicts constant opponent speed and lateral position.</li><li>Obstacle ellipses are expressed in Frenet coordinates with bounded slack (0–0.08). Collision checks approximate bodies in Frenet coordinates.</li><li>No recursive-feasibility terminal set or safety guarantee. Synthetic circuit sections and short episodes replace the paper’s larger benchmark.</li>"
checkpoint_link = (
    '<a href="data/evaluation.json" download>All 45 test records</a>'
    if rl
    else '<a href="data/experiment.json" download>All nine simulation records</a>'
)
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name} · Predictive planning for autonomous racing</title>
<meta name="description" content="{hero_lead}"><meta property="og:title" content="{name} · ECC 2023 research">
<meta property="og:description" content="{hero_lead}"><meta property="og:image" content="assets/driving-scene.png">
<link rel="icon" type="image/svg+xml" href="assets/icon.svg"><link rel="stylesheet" href="styles.css"></head>
<body><header><nav class="wrap" aria-label="Main navigation"><a class="brand" href="#">{name}<span class="muted"> / research</span></a><a href="#simulation">Simulation</a><a href="#research">Research</a><a href="#results">Results</a><a href="https://github.com/RudolfReiter/{repo}">GitHub ↗</a></nav></header>
<main><div class="wrap"><section class="hero"><div><p class="eyebrow">Autonomous racing · {"Hierarchical reinforcement learning" if rl else "Vehicle simulation & control"}</p><h1>{hero_title}</h1><p class="lead">{hero_lead}</p><div class="links"><a href="#simulation">Explore the simulation ↓</a><a href="#reproduce">Run it locally</a></div></div>{publication}</section>
<div class="notice">{notice}</div>
<section class="section" id="architecture" style="border-top:0"><div class="section-head"><h2>{"From policy to driving" if rl else "From a reference to a trajectory"}</h2><p>A transparent interface between strategic decisions, physical dynamics, and the trajectory you see on screen.</p></div><div class="flow">{flow_markup}</div></section>
<section class="section" id="simulation"><div class="section-head"><h2>Recorded driving simulation</h2><p>Follow the actual car and compare it with a forecast held fixed for three seconds. Playback starts paused, moves continuously, and stops at the end.</p></div>
<div class="scene-panel"><div class="toolbar"><label>Task <select id="scenario" aria-label="Driving task"></select></label><label>{"Policy" if rl else "Reference"} <select id="method" aria-label="Controller"></select></label><label>Camera <select id="view"><option value="follow">Smooth follow</option><option value="overview">Track overview</option></select></label></div>
<div class="viewport"><canvas id="scene" role="img" aria-label="Recorded multi-car driving scene with an actual ego car and a fixed forecast ghost">Use the downloadable scene image and data if Canvas is unavailable.</canvas><div class="scene-caption"><strong id="scene-title"></strong><p id="scene-subtitle"></p></div></div>
<div class="legend"><span><i class="dot blue"></i>Actual ego</span><span><i class="dot orange"></i>Fixed forecast · same elapsed time</span><span><i class="dot gray"></i>Actual opponents</span><label><input type="checkbox" id="ghost" checked> Show forecast</label></div>
<div class="playback"><button id="play" class="primary">Play</button><button id="reset" aria-label="Restart playback">Reset</button><input id="seek" type="range" min="0" max="16" step=".01" value="0" aria-label="Simulation time"><span class="time" id="time">0.0 / 16.0 s</span><select id="rate" aria-label="Playback speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div>
<div class="metrics"><div class="metric"><p>Actual speed</p><strong id="speed-value">—</strong></div><div class="metric"><p>Requested speed</p><strong id="reference-value">—</strong></div><div class="metric"><p>Position forecast error</p><strong id="error-value">—</strong></div><div class="metric"><p>Forecast age</p><strong id="age-value">—</strong></div></div></div>
<p class="status" id="run-status" aria-live="polite"></p>
<p class="explanation"><strong>How to read the ghost.</strong> At 0, 3, 6… seconds, the orange path freezes the planner’s next three seconds. The ghost then moves along that frozen path at the same elapsed time as the blue car. A gap shows how the executed drive diverged as the controller replanned. At the next boundary, a new forecast begins from the current state. This measures plan consistency under replanning, not an independent learned prediction model.</p>
<div class="plots"><article class="plot"><h3>Speed reference and actual speed</h3><div id="speed-plot"></div><p>Blue: actual speed. Orange: the requested speed given to the optimizer.</p></article><article class="plot"><h3>Lateral reference and actual position</h3><div id="lateral-plot"></div><p>Blue: actual position across the track. Orange: the requested lateral reference.</p></article><article class="plot"><h3>Fixed forecast position error</h3><div id="error-plot"></div><p>Distance in Frenet coordinates from the frozen forecast. It resets when a new forecast starts; the scene readout uses Cartesian distance.</p></article></div>
<noscript>Interactive playback requires JavaScript. <a href="assets/driving-scene.png">View a recorded scene</a> or download the evaluation data below.</noscript></section>
<section class="section" id="research"><p class="eyebrow">Research contribution · ECC 2023</p><h2 class="research-title">A Hierarchical Approach for Strategic Motion Planning in Autonomous Racing</h2><p class="explanation">{research_text}</p>
<div class="research-grid"><article class="card"><h3>{"Implemented in this recreation" if rl else "Repository improvements"}</h3><ul>{impl_points}</ul></article><article class="card"><h3>Differences from the paper</h3><ul>{limits}</ul></article></div><div class="links"><a href="{paper}">IEEE publication ↗</a><a href="https://publications.syscop.de/Reiter2023.pdf">Author manuscript</a><a href="https://rudolfreiter.github.io/hilepp_vis/">Original project page</a></div></section>
<section class="section" id="results"><div class="section-head"><h2>{"Held-out car evaluation" if rl else "Car example evaluation"}</h2><p>{"One training seed · three tasks · five test seeds per task" if rl else "Three references · three tasks · one fixed seed"}</p></div><p>{result_intro}</p>
<img class="figure" src="assets/evaluation.svg" width="1200" height="400" loading="lazy" alt="{figure_alt}"><p class="figure-note">{result_detail}</p><div class="table-scroll"><table><caption class="small muted">{"Held-out evaluation; lower final rank is better." if rl else "Recorded fixed-reference simulations; lower final rank is better."}</caption>{table}</table></div>
<p class="small muted">No collision was detected in the recorded evaluation. This finite sample and approximate body model do not establish safety. A rejected optimizer solution ends the episode before its control is applied. Return combines forward progress and the number of opponents behind the ego.</p><div class="links">{checkpoint_link}<a href="data/experiment.json" download>Replay data · JSON</a><a href="assets/evaluation.svg" download>Download figure · SVG</a><a href="assets/driving-scene.png" download>Download driving scene</a></div></section>
<section class="section" id="reproduce"><div class="section-head"><h2>Run the research</h2><p>{"Use the included new checkpoint, retrain SAC, or connect your own normalized policy to the planner." if rl else "Install the portable backend and regenerate every recorded driving scene with one command."}</p></div><pre><code>{html.escape(command)}</code></pre>
<details><summary>Numerical setup and provenance</summary><p>Python 3.9 · NumPy 1.26.4 · SciPy 1.13.1 · CasADi 3.8.1 / IPOPT{" · PyTorch 2.8.0 on CPU" if rl else ""}. The planner uses 12 intervals of 0.25 s, a five-state Frenet model, RK4 integration, and three opponents. Simulation uses five substeps per interval. Full settings, checkpoint provenance, and limitations are documented in the repository.</p></details>
<details><summary>What is included in the repositories?</summary><p>Python source, regression tests, recorded evaluations, editable web visualization, a scientific figure generator, citation metadata, and GitHub Pages deployment. {"The HILEPP wrapper depends on the improved vehicle_gym package instead of duplicating its dynamics or planner." if rl else "The original acados planner and demonstration scripts remain available as an optional legacy stack."}</p></details>
<div class="links"><a href="https://github.com/RudolfReiter/{repo}">Repository ↗</a><a href="assets/source.zip" download>Download source archive</a><a href="https://github.com/RudolfReiter/{"vehicle_gym" if rl else "hilepp-recreated"}">{"vehicle_gym foundation" if rl else "HILEPP policy wrapper"} ↗</a></div></section>
</div></main><footer><div class="wrap footer-row"><p>Rudolf Reiter · Autonomous racing research</p><p>{"Recreated implementation · Not the original HILEPP repository" if rl else "vehicle_gym · Original models, improved research interface"}</p></div></footer>
<script src="data/experiment.js"></script><script src="scene.js"></script><script src="app.js"></script></body></html>'''
(ROOT / "docs/index.html").write_text(page)
(ROOT / "docs/assets/icon.svg").write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#2166ac"/><path d="M12 46 27 17h10L52 46H39l-7-15-7 15z" fill="white"/></svg>'
)
print(f"Built {name} page and figures from recorded results.")
