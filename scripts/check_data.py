"""Check exported experiment integrity, local page links, and source archive."""

import json
import math
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import zipfile

root = Path(__file__).resolve().parents[1]
docs = root / "docs"
data = json.loads((docs / "data/experiment.json").read_text())
js = (docs / "data/experiment.js").read_text()
assert json.loads(js.removeprefix("window.EXPERIMENT=").strip().removesuffix(";")) == data
count = 0
for scenario in data["scenarios"]:
    for run in scenario["runs"]:
        count += 1
        assert len(run["states"]) == len(run["opponents"]) == len(run["time"])
        assert len(run["plans"]) == len(run["references"])
        assert run["time"][0] == 0
        assert all(b > a for a, b in zip(run["time"], run["time"][1:]))
        assert run["metrics"]["duration_s"] == run["time"][-1]
        for state in run["states"]:
            assert len(state) == 5 and all(math.isfinite(x) for x in state)
        for index, plan in enumerate(run["plans"]):
            assert plan["origin"] == index * data["dt"]
            if plan["success"]:
                assert len(plan["states"]) == 13
                assert plan["max_violation"] < 2e-4
                assert plan["obstacle_slack"] <= 0.0802
                assert (
                    max(abs(a - b) for a, b in zip(plan["states"][0], run["states"][index])) < 2e-4
                )
        assert all(len(row) == 3 for row in run["opponents"])
if data["kind"] == "hilepp":
    training = data["training"]
    assert set(data["test_seeds"]).isdisjoint(training["validation_seeds"])
    assert min(data["test_seeds"]) >= 10000
    assert all(episode["seed"] < 7000 for episode in training["episodes"])
    selected = max(training["validation"], key=lambda v: v["return"])
    assert selected["step"] == training["selected_step"]
    assert len(data["evaluation"]) == len(data["test_seeds"]) * 9
    records = json.loads((docs / "data/evaluation.json").read_text())
    assert records == data["evaluation"]
    for scenario in data["scenarios"]:
        for run in scenario["runs"]:
            row = next(
                r
                for r in records
                if r["scenario"] == scenario["id"]
                and r["method"] == run["id"]
                and r["seed"] == run["seed"]
            )
            assert all(row[k] == v for k, v in run["metrics"].items())


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.refs = [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        for key in ["href", "src"]:
            if key in values:
                self.refs.append(values[key])


page = Page()
page.feed((docs / "index.html").read_text())
assert len(set(page.ids)) == len(page.ids)
for ref in page.refs:
    if ref.startswith("#"):
        assert ref == "#" or ref[1:] in page.ids, ref
    elif not urlparse(ref).scheme:
        assert (docs / ref).is_file(), ref
with zipfile.ZipFile(docs / "assets/source.zip") as archive:
    assert archive.testzip() is None
    assert any(name.endswith("pyproject.toml") for name in archive.namelist())
    assert not any("/.venv/" in name or "/.git/" in name for name in archive.namelist())
print(f"Validated {count} recorded replays, numerical residuals, page links, and source archive.")
