# Control and prediction in the six-car race

At control time `t`, the simulator copies the states of all six vehicles. Each MPC receives its own state and the five measured opponent states from that snapshot. Its optimizer returns a five-second trajectory and a sequence of controls. Only the first control is applied for 0.2 seconds; all vehicles then advance simultaneously. The procedure repeats at the next control time.

The browser displays recorded solver outputs. It does not compute or extrapolate the race itself. Interpolation between measured frames smooths playback; the current planning horizon updates at the recorded control frequency.

## Three distinct controllers

| Name | Ego action | Other five vehicles |
|---|---|---|
| MPC | Fixed references and weights → nonlinear MPC → force and steering rate | Independent MPC |
| HILEPP | Four-dimensional SAC action → speed/lateral references and their cost weights → the same MPC → actuators | Independent MPC |
| SAC | Two-dimensional SAC action → force and steering rate directly | Independent MPC |

Only MPC is part of the vehicle_gym competition. The learned comparisons live in the separate recreated HILEPP repository. Direct SAC has no ego MPC plan, safety-filter fallback or obstacle-prediction module. Selecting one of its opponents still shows that opponent's MPC.

## Other-car assumptions

For each observer, a car ahead or alongside keeps its measured speed and heading error in the curvilinear frame. Once it is more than one vehicle length (4.6 m) behind, its assumed acceleration switches to −10 m/s² until standstill. Lateral predictions are clipped to the drivable corridor. RK4 integration splits exactly at standstill and prevents reverse motion.

This full-braking trajectory represents the assumed ability of a follower to yield. It is not a prediction of the other MPC's intended braking command. Different observers can assign different modes to the same physical car. None of them has access to another controller's planned controls or future simulation states.

The cyan curve is the selected controller's planned ego motion. Dashed yellow curves show constant-speed opponent assumptions; dotted rose curves show full-braking assumptions. All solid cars are actual simulated vehicles. There are no ghost cars, held forecasts or prediction-accuracy comparisons.

## Numerical interpretation

SQP iterates are accepted at status 0 or iteration-limit status 2 if state trajectories are finite and equality/inequality residuals are below 0.02. This is numerical acceptance, not an optimality or safety certificate. Complementarity and stationarity are optimization convergence measures and are not described as physical constraint violations.

State and obstacle constraints are soft. Their slack can be positive even when the numerical problem is solved. Elliptical obstacle margins are 9 m longitudinally and 3.8 m laterally. Physical collisions use oriented 4.6 × 1.9 m rectangles and are checked at four integration substeps. Track exits also account for the body footprint. The rendered track edge follows the same conservative usable width as these checks, rather than displaying a wider corridor than the simulation permits.

All race termination reasons and involved vehicle IDs are retained. The horizon shown at a failed terminal solve is not applied. A completed simulation is evidence about that rollout only, not a general safety guarantee.
