# Circuit racing interface

The active experiment protocol is `six-car-circuit-v1`. It replaces the previous short, synthetic-road page and its scenario categories. The older portable examples remain source-compatible but are not used for the published racing evaluation.

Use `python -m vehiclegym.race_demo` for the new competition. All six cars now have independent acados MPC controllers. Geometry comes from the packaged IMS and Sepang datasets. A race runs for up to 180 s, with simultaneous updates at 0.2 s.

Replay files are individually compressed JSON, indexed by `docs/data/manifest.json`. They include measured states, every applied command and all six controllers' planning outputs. There is no duplicate full replay embedded in JavaScript. The browser loads only the selected race.

The HILEPP circuit observation has 31 entries: seven curvature samples, four ego states and four states for each of five opponents. Its four actions parameterize HILEPP-II. Direct SAC instead uses two actuator actions. Older 22-input checkpoints are incompatible and are not used by this evaluation.

See [control and prediction semantics](CONTROL_AND_PREDICTION.md) for replanning, obstacle assumptions and numerical limitations.
