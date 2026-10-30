"""Run a competition between six independent MPC racing policies."""

import argparse
from pathlib import Path
from .simulator.circuit_race import CircuitRace
from .race_replay import record_race, write_replay, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", default=["Oval", "IMS", "Sepang"])
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument(
        "--oval-duration", type=float, default=240, help="Longer repeated-lap oval race"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    args = parser.parse_args()
    entries = []
    records = []
    for track in args.tracks:
        world = CircuitRace(track, args.oval_duration if track == "Oval" else args.duration)
        for seed in args.seeds:
            run = record_race(world, seed, progress=True)
            filename = f"{track.lower()}-mpc-{seed}.json.gz"
            write_replay(run, args.output / filename)
            records.append(dict(method="MPC", seed=seed, track=track, metrics=run["metrics"]))
            entries.append(
                dict(
                    id=f"{track.lower()}-mpc-{seed}",
                    method="MPC",
                    seed=seed,
                    track=track,
                    file=filename,
                    metrics=run["metrics"],
                )
            )
            print(track, seed, run["metrics"], flush=True)
    write_manifest(args.output, "vehicle_gym", entries, records)


if __name__ == "__main__":
    main()
