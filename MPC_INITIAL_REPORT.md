# Initial learned-MPC experiment

The frozen learned dynamics model successfully selects and executes joint commands
in closed loop. This initial random-shooting MPC does **not** outperform direct
position commands on the five tested scenarios. This is a useful negative result:
accurate learned prediction alone does not guarantee improved control when an
existing position servo already performs the task well.

## Method

256 candidates, 8 predicted steps (0.8 s), 100 ms execution interval, seed 42.
The predictor uses its own previous state within every rollout. At each new plan,
the controller receives actual angles/velocities. Only the first action executes.
Future scheduled goal changes are not supplied to either controller in advance.
Candidates are restricted to actuator limits intersected with joint ranges inset
by 0.08 rad. Predicted nominal joint-limit violations are rejected, never clipped.
The cube is outside reach; no deliberate impacts are commanded.

The architecture, normalization, weights, datasets, and previous evaluation
artifacts remain unchanged. Checkpoint SHA256:
`2e71458992bafe881e95245576f30b6a7bc0733b965a3d33c3b4b05834fd9f4c`.
No configurations were tuned after inspecting benchmark performance.

## Per-scenario results

Tracking RMSE includes the full scenario, with errors measured every 2 ms.
Angle pairs are shoulder / elbow, in degrees. Settling times refer to the last
target segment and require both position and velocity tolerances to remain met.

| Scenario | Direct RMSE | MPC RMSE | Direct settling (s) | MPC settling (s) |
| --- | --- | --- | --- | --- |
| Short | 1.64 / 2.36 | 1.64 / 2.36 | 0.564 | 0.564 |
| Large | 18.47 / 22.08 | 19.30 / 22.30 | 0.924 | 0.924 |
| Moving target | 16.19 / 15.54 | 16.46 / 15.82 | 0.722 | 0.722 |
| Near limits | 25.82 / 27.72 | 26.00 / 28.54 | 0.950 | 0.966 |
| Reversals | 25.99 / 30.11 | 26.65 / 30.58 | 0.860 | 0.858 |

All final errors were below 1e-8 degrees and final speeds below 1e-8 rad/s for
both controllers. Both had zero directional overshoot, limit exceedance samples,
and active limit-constraint physics steps. Detailed per-scenario final errors,
velocities, computation times, and per-target settling times are in the JSON.

## Aggregate and computation

| Metric | Direct | MPC |
| --- | --- | --- |
| Pooled tracking RMSE (degrees) | 20.26 / 22.52 | 20.70 / 22.95 |
| Settled final targets | 5/5 | 5/5 |
| Mean final-target settling time | 0.804 s | 0.807 s |
| Mean controller computation | 0.007 ms | 5.60 ms |
| 95th-percentile computation | 0.018 ms | 8.30 ms |
| Maximum computation | 0.079 ms | 41.56 ms |
| Missed 100 ms computation budget | 0 | 0 |

Aggregate tracking RMSE pools all physics samples, so longer scenarios carry
more weight. Final-error/residual-speed means and maximum overshoot are also
reported in `report.json`. Computation excludes physics and file/plot output;
headless plant simulation does not incorporate controller delays.

## Does the model actually select commands?

Yes. Every plan scores model rollouts, including a constant-goal candidate.
MPC selected a first command different from the goal on 9 of 260 ticks:
0 short, 2 large, 1 moving-target, 2 near-limits, and 4 reversal ticks. Tests
also replace the predictor with a deliberately biased dynamics function and
verify that the selected command changes to compensate rather than copying
the goal. Most real-model selections prefer the existing servo's direct target.

An offline audit compared each initial selected sequence and direct sequence
under the same objective, using both the learned predictor and MuJoCo replays:

| Scenario | Predicted selected / direct cost | MuJoCo selected / direct cost |
| --- | --- | --- |
| Short | 0.130 / 0.130 | 0.126 / 0.126 |
| Large | 15.735 / 16.286 | 15.512 / 16.174 |
| Moving target | 2.621 / 2.621 | 2.659 / 2.659 |
| Near limits | 26.182 / 28.246 | 26.220 / 28.557 |
| Reversals | 2.872 / 2.872 | 2.894 / 2.894 |

No initial-plan ranking reversal was observed: the selected alternatives on large
and near-limit moves also reduced the objective in real physics. The objective
includes velocity and command-change penalties, so lower objective cost need not
mean lower position error. Later replans can differ from the initial sequence.
Executed one-step prediction errors are logged for every scenario. This audit
does not exclude exploitation at later plans or in other states.

## Interpretation and limitations

The learned model is usable for fast candidate prediction, but simple regulation
of an already position-controlled arm leaves little room for improvement. MPC
is slightly more conservative during some transients and costs more computation.
There is no demonstrated tracking advantage here. Random shooting is approximate,
the candidate count is modest, and weights express a particular tradeoff rather
than an independently optimized control specification.

Only five deterministic scenarios and one seed were tested. Different seeds,
moving targets, trajectory constraints, or more demanding goals may change the
comparison; they must be evaluated rather than assumed to favor MPC. The model
has not been established as accurate for hard stops or external contact. No
actual limit contacts occurred here. Legal fallback commands do not guarantee
safe dynamics under all possible momentum states.

## Validation and artifacts

All 17 tests passed. The optional MPC viewer ran the four-second short scenario
and exited successfully. Headless benchmark artifacts are in `runs/mpc_initial/`:
per-scenario plots, per-controller sampled/physics trajectories, planner
diagnostics, and the complete machine-readable report. README documents the
cost equation, weights/units, metrics, configuration, and reproduction commands.
