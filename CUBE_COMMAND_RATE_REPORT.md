# CEM command-rate experiment

**Recommend trying 0.5 rad/s per joint with the frozen stationary-20 model.** In twelve paired five-second MuJoCo trials, success improves from 3/12 to 10/12, workspace escapes fall from 4/12 to zero, and mean final target distance falls from 13.05 cm to 1.16 cm. This is a command constraint applied before model prediction, with no model retraining or changes to the score.

## Use it

From the project folder in PowerShell:

```powershell
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --checkpoint runs/arm_cube_stationary_experiment/stationary_20/model.pt --command-rate-limit 0.5
```

Select `--episode 23`, `99`, `139`, or `159`, and `--seed 42`, `43`, or `44`. Viewer controls remain 6/7 for target X, 8/9 for target Y, Enter for scenario reset. Start with episode 99 for a successful example; episode 23 remains harder.

Set `--command-rate-limit 0` to reproduce the unlimited controller. Values are radians per second per joint, so 0.5 permits at most 0.05 rad (about 2.9 degrees) of command change at each 100 ms update. The existing bare viewer command still uses the old Medium checkpoint and unlimited commands; the command above explicitly selects the tested combination. The feature is configurable and preserves previous defaults.

Repeat the full paired comparison in a fresh directory:

```powershell
.venv/Scripts/python.exe -m simple_arm.cube_mpc compare-rate --command-rate-limit 0.5 --output runs/cube_cem_command_rate_repeat
```

`compare-rate` defaults to the stationary-20 checkpoint and a 0.5 rad/s limited arm, while comparing against rate 0 with the same checkpoint and score. An explicit `--checkpoint` overrides its model. Existing output directories are refused. No parameter sweep was used to select 0.5 after seeing these results.

## Implementation

- `Config.command_rate_limit`: finite nonnegative scalar in rad/s per joint; zero disables the constraint.
- After existing knot interpolation, project every candidate sequentially: `u[t] = u[t-1] + clamp(requested[t] - u[t-1], -rate*0.1, +rate*0.1)`. The first command starts at the previously executed target. Preserve the actuator/joint command bounds.
- Neural rollout, scoring, winner retention and execution all use this projected action sequence. There is no execution-only clipping that would make the predicted and executed first commands disagree.
- Mean/hold candidates and warm starts pass through the same projection. Elite fitting still uses the latent knot proposals; the projection can produce piecewise-limited sequences from those ten variables.
- Fallback holds the previous command and therefore satisfies the rate bound. Target changes reset the warm start while retaining the actual previous command; Enter resets the physical scenario and its initial command.
- Both execution paths reject an unexpected rate violation rather than silently alter a selected command. Each plan logs configured rate and its selected sequence maximum rate, as well as existing cost components.

The 15-step horizon, 256 candidates, five CEM iterations, knot count, sampling spread, elites, original scoring weights and 50 x 2 ms execution interval are unchanged. All planning uses the frozen learned model; MuJoCo is used only for actual closed-loop evaluation. No additional acceleration penalty, model training or dataset collection was introduced.

A command-rate limit bounds the position-target slew, not physical joint speed or cube velocity. Position actuators and dynamics can overshoot. Rapid reversals in command rate are still possible because acceleration is not constrained. The measured physical outcomes below show the effect in these scenarios, not a general bound on cube speed or reachability.

## Paired evaluation

Four saved initial states and demonstrated targets (episodes 23,99,139,159), seeds 42/43/44, five simulation seconds per run. Both arms use `runs/arm_cube_stationary_experiment/stationary_20/model.pt` and `--scoring original`. Success is within 2 cm for every 2 ms sample in the final 0.5 s. Workspace escapes use the existing joint-limited kinematic cube-center proxy. All initial states are identical within each pair.

| Metric | Unlimited | 0.5 rad/s |
|---|---:|---:|
| Successful runs | 3/12 (25.0%) | 10/12 (83.3%) |
| Mean final target distance | 13.05 cm | 1.16 cm |
| Runs escaping workspace | 4/12 | 0/12 |
| Outside-workspace physics samples | 23.72% | 0.00% |
| Peak cube speed | 2.465 m/s | 0.407 m/s |
| Mean per-run peak cube speed | 1.440 m/s | 0.279 m/s |
| Peak command rate | 31.199 rad/s | 0.500 rad/s |
| Peak actual joint speed | 16.892 rad/s | 0.711 rad/s |

| Episode / seed | Final cm, unlimited / limited | Success, unlimited / limited | Escaped, unlimited / limited | Cube peak m/s, unlimited / limited |
|---|---:|---|---|---:|
| 23/42 | 15.68 / 2.97 | False / False | True / False | 1.536 / 0.333 |
| 23/43 | 9.93 / 1.29 | False / True | False / False | 1.480 / 0.354 |
| 23/44 | 19.24 / 5.50 | False / False | True / False | 1.422 / 0.407 |
| 99/42 | 1.75 / 0.33 | True / True | False / False | 1.034 / 0.214 |
| 99/43 | 3.60 / 0.27 | False / True | False / False | 1.715 / 0.224 |
| 99/44 | 2.55 / 0.16 | False / True | False / False | 1.187 / 0.224 |
| 139/42 | 43.23 / 0.98 | False / True | True / False | 2.444 / 0.312 |
| 139/43 | 1.54 / 0.62 | True / True | False / False | 0.921 / 0.305 |
| 139/44 | 49.69 / 0.57 | False / True | True / False | 2.465 / 0.333 |
| 159/42 | 1.85 / 0.66 | True / True | False / False | 0.805 / 0.212 |
| 159/43 | 3.99 / 0.08 | False / True | False / False | 0.543 / 0.219 |
| 159/44 | 3.59 / 0.53 | False / True | False / False | 1.726 / 0.216 |

Every limited run finishes closer to its target than its unlimited counterpart. The two remaining failures are episode 23, seeds 42 and 44 (final distances 2.97 and 5.50 cm). No longer-duration success is inferred. This compares controller settings using the same model, so the difference cannot be attributed to a simultaneous model or workspace-score change.

The numerical maximum measured command rate is 0.50000016 rad/s because commands use float32; enforcement allows a small numerical tolerance. This is not evidence of an actual 0.5 rad/s physical joint-speed bound: maximum actual joint speed is 0.711 rad/s.

## Artifacts, timing and checks

`runs/cube_cem_command_rate/` contains config/provenance, 24 JSON/NPZ traces, summary, `distance_comparison.png`, and `commands_cube_velocity.png`. Per-run logs include executed targets, 2 ms physical states, 100 ms observations, selected model trajectories, costs, rejection/fallback counts, actual workspace escapes, command rates/variation/acceleration, and actual joint/cube speeds.

| Mode | Mean / median / P95 / max planning ms | >100 ms / 600 plans |
|---|---|---:|
| unlimited | 57.9 / 53.4 / 91.1 / 182.9 | 21 |
| limited | 60.6 / 54.2 / 103.0 / 316.7 | 37 |

Timing is machine/load dependent; a focused test run overlapped part of evaluation. The viewer smoke test logged startup overruns of 173 and 131 ms. Simulation-time updates stay 100 ms regardless of planning overrun; this remains a best-effort real-time viewer.

- Thirty-five relevant tests passed: rate projection for every candidate and both joints, first-step bounds, box limits, model/action consistency, score logging, reset/warm starts, fallback, execution guard, short actual execution, disabled mode, overwrite protection, existing CEM/workspace behavior, and stationary training compatibility.
- Viewer smoke test with stationary-20 + 0.5 rad/s launched and closed successfully. Commands and physical states from two representative full episodes were repeated with the final code and matched the saved traces exactly.
- All 35 pre-existing dataset/checkpoint/XML hashes remain unchanged, including every stationary-training checkpoint. No existing experiment directory was overwritten.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_cube_command_rate.py tests/test_cube_mpc.py tests/test_cube_workspace.py tests/test_stationary_training.py -q --basetemp .pytest_cache/cube_rate_verify2
```

This is a small seeded benchmark, not a guarantee on arbitrary interactive targets. The results support using the 0.5 rad/s setting as the next controlled-viewer configuration while retaining unlimited mode for comparison.
