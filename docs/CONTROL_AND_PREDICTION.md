# Control, prediction, and replay semantics

## What is replanned, and what is predicted?

Every 0.25 s, `RacingWorld.step` reads the current ego and opponent states, builds new opponent trajectories, solves a fresh 3 s nonlinear MPC problem, applies **only the first force/steering-rate command**, and advances the physical simulation by five 0.05 s substeps. The rest of the control horizon is discarded; the previous solution is only a warm start for the next optimization.

The blue path on the page is that current **ego plan**. Orange paths are the **other vehicles’ obstacle trajectories actually supplied to that solve**. They are exported from `Plan.opponent_predictions`, not reconstructed later by the browser. Translucent cars mark the same selected future timestamp within the current solve. The page replays these recorded solves; it does not run optimization in JavaScript.

Section IV-B.3 of the paper distinguishes roles. A leading or alongside opponent retains measured speed and heading error. An opponent fully behind the ego (more than 4.6 m) is assigned braking at 8 m/s² until standstill. The braking trajectory is a feasibility assumption about a follower yielding, **not a prediction that it intends to stop**. This recreation integrates the position using midpoint speed and current curvature, with lateral clipping at the road boundary; it does not duplicate the paper’s RK4 predictor exactly. No future opponent state or controller action enters the predictor.

An optional display holds one opponent assumption for 3 s and compares its ghost with the actual opponent at the same timestamp. The reported separation is diagnostic, especially for braking assumptions. MPC keeps replanning underneath it. An earlier page instead froze the ego plan; divergence from an old ego plan is expected with replanning and is not an obstacle prediction error. That confusing default comparison has been removed.

## Interactive scenarios

The `interactive-v2` environment uses 24 s episodes, mirrored lateral arrangements, seeded position/speed perturbations, and synthetic track curvature. The ego is capped at 32 m/s.

| Task | Initial traffic | Opponent free-running targets |
|---|---|---|
| Overtaking | Three cars 23, 49, 76 m ahead, staggered across the track | 25, 26, 27 m/s |
| Blocking | Three challengers 20, 43, 66 m behind, approaching on different lines | 36, 35, 34 m/s |
| Mixed | Two slower cars ahead and a faster challenger behind | 25, 36, 27 m/s |

Distances are perturbed by ±2 m; speeds by ±0.4 m/s. Opponents are simulated with the original vehicle dynamics and a fixed gap-seeking controller. They evaluate center/left/right passing lines, hold a changed target for 2.5 s, avoid crossing alongside cars, and slow for a car ahead. They are neither learned policies nor the paper’s MPC opponents. Different speed capabilities and shorter episodes make this a compact interaction task, not the original racing benchmark.

Pass events require a change of relative order followed by 4.6 m of clearance. A defensive move requires more than 0.8 m of ego lateral motion over two seconds, more than 0.8 m reduction in lateral separation to a challenger’s line, and that challenger being held up. A 2.5 s cooldown prevents repeatedly counting the same movement. Held-up vehicle-seconds count opponents 4.6–30 m behind and within 2.7 m laterally whose controller target is reduced by more than 1 m/s and whose actual speed is at least 0.5 m/s below their free-running target. The general SAC policy uses only progress/rank reward. The optional HILEPP defensive specialist has a separately documented lane-coverage and held-up bonus during training; pass and defensive-move event counts are never added to either reward. Multiple opponents can contribute simultaneously.

A defensive event alone does not prove a strong policy: read event counts alongside opponent passes, episode completion, collisions, and solver failures. Oriented Cartesian body rectangles are checked every 0.05 s. `min_body_separation_m` is retained as a Frenet axis-separation proxy; it is not Euclidean clearance.
