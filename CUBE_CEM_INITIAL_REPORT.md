# Learned CEM cube-pushing experiment

Implemented 2026-11-04. This is an initial frozen-model control experiment, with no retraining or new training-data collection.

## Run it

From the repository root (PowerShell):

```powershell
.venv/Scripts/python.exe -m simple_arm.cube_mpc view
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --episode 99 --target 0.36 0.09
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --mode no-action
.venv/Scripts/python.exe -m simple_arm.cube_mpc evaluate
```

Evaluation refuses an existing output directory. The completed benchmark is in `runs/cube_cem_initial/`; use `--output runs/<new-name>` for another experiment. Dataset/checkpoint paths resolve from the working directory. Viewer `--seconds 3` closes after approximately three wall-clock seconds, including startup. Controls: 6/7 change target X by -/+2 cm, 8/9 change Y by -/+2 cm, Enter resets the selected scenario and its original target. Target changes reset the seeded planner and warm start. Arbitrary viewer targets are exploratory; benchmark targets are saved demonstrated endpoints.

## Implementation and configuration

- `simple_arm/cube_mpc.py`: validated frozen model adapter, tensor recursive rollout, dimensionless cost, CEM, configuration and CLI.
- `simple_arm/cube_mpc_experiment.py`: exact state restoration, first-action execution, evaluation, diagnostics, hashes and plots.
- `simple_arm/cube_mpc_viewer.py`: passive viewer, input queue and decoration-only target/forecast geometry.

The default checkpoint is `runs/arm_cube_capacity_refinement/medium_256/model.pt`. The adapter uses the existing width-aware construction (`hidden_width`, legacy default 128) and `differentiable_next` transition, checks model hash, state/action conventions, link lengths, 20 ms base interval, stride 5, 2 ms physics interval and normalization widths/scales. Parameters are frozen; candidate recursion stays in CPU PyTorch with two threads. NumPy conversion occurs at the plan boundary. The planner has no simulation object or access to recorded actions, contact labels or collision queries.

Default CEM: 15 steps (1.5 s), 256 candidates, 5 iterations, 26 elites (ceil of 10%), 5 linearly interpolated absolute-position knots. Sampling starts with 0.5 rad standard deviation each plan; elite distribution updates retain 20% of the old mean/std, use population standard deviation, and floor std at 0.05 rad. Include the distribution mean and previous-command hold among candidates. Retain the best feasible sampled sequence across all iterations. Warm starts shift the previous winning sequence one step, repeat its endpoint, and linearly resample five knots. Commands use the actuator/joint intersection with a 0.15 rad joint margin. Predicted nonfinite states and joint-limit violations are rejected without clipping. All-infeasible plans hold the supplied previous legal command, clear warm start and flag fallback; their forecast arrays are NaN and their cost is null.

Cost: mean squared XY target distance / 0.1? + 5 times terminal squared XY distance / 0.1? + 0.01 times mean squared command changes (including the previous executed command) / 1 rad? + 0.001 times mean squared joint velocity / (1 rad/s)?. XY distance sums two coordinates; command and velocity MSE average over both joints and all horizon steps.

Every Config field has a CLI flag: `--horizon`, `--candidates`, `--iterations`, `--elite-fraction`, `--knots`, `--initial-std`, `--retention`, `--std-floor`, `--command-margin`, `--distance-weight`, `--terminal-weight`, `--change-weight`, `--velocity-weight`, `--seed`. Also supported: `--checkpoint`, `--dataset`, `--output`, `--episode`, `--target`, `--mode`, `--seconds`. Benchmark seeds are always 42/43/44; `--seed` selects the viewer seed. No effort is reduced to meet deadlines.

Each control update executes only the first winning command for 50 x 2 ms physics steps, extracts fresh truth, then replans. Physics settings, friction, gravity and normal position actuators are unchanged. No-action holds the initial joint commands. Viewer pacing sleeps only when computation and execution take less than 100 ms; overruns are printed and simulation-time intervals remain exactly 100 ms.

## Protocol and artifacts

Four stationary-start clear-push episodes (23, 99, 139, 159) from `data/arm_cube_contact_boundary/independent.npz` (seed 292); their recorded final XY positions are targets. Five simulated seconds per run, three CEM seeds and one deterministic hold baseline per case. Exact saved state restoration is shared by both modes. Success requires all 250 physics samples in the final 0.5 s to be within 2 cm. Minimum distance uses the 2 ms trace as well as the initial observation.

The completed output contains config/provenance and pre-run hashes; per-run NPZ arrays (`states`, `fine_states`, `commands`, `predictions`, `timings`, `joint_limits`, `target`); per-run JSON costs, iteration diagnostics, rejection/fallback counts, prediction/progress comparisons and excursions; and `summary.json` with per-run, per-case, per-seed and pooled summaries. Control-boundary states include the initial state; forecast index 0 is the selected prediction after the next 100 ms. Costs and diagnostics are aligned with executed commands. Baseline forecasts are NaN because it does not plan. Rejection counts are candidate evaluations across iterations; nonfinite and limit reasons may overlap. All failed runs are retained.

Plots: `distance_over_time.png`, `predicted_actual_displacement.png` (first-step vector magnitudes), `planning_time.png`, `endpoint_comparison.png` (actual paths/endpoints, start and demonstrated target). The endpoint plot compares actual controller and hold outcomes. Predicted terminal progress in JSON is explicitly an **unverified planning forecast**; it is not compared against later closed-loop endpoints because replans change the subsequent commands.

## Results

| Case | Seed/mode | Initial cm | Minimum cm | Final cm | Reduction cm | Success |
|---|---|---:|---:|---:|---:|---|
| 23 | hold | 10.21 | 10.21 | 10.21 | 0.00 | no |
| 23 | 42 | 10.21 | 0.23 | 18.78 | -8.57 | no |
| 23 | 43 | 10.21 | 0.19 | 11.88 | -1.67 | no |
| 23 | 44 | 10.21 | 0.21 | 23.15 | -12.93 | no |
| 99 | hold | 8.79 | 8.79 | 8.79 | 0.00 | no |
| 99 | 42 | 8.79 | 0.17 | 5.11 | 3.68 | no |
| 99 | 43 | 8.79 | 3.05 | 5.05 | 3.74 | no |
| 99 | 44 | 8.79 | 3.07 | 4.86 | 3.94 | no |
| 139 | hold | 7.09 | 7.09 | 7.09 | 0.00 | no |
| 139 | 42 | 7.09 | 0.43 | 2.64 | 4.45 | no |
| 139 | 43 | 7.09 | 0.92 | 3.38 | 3.70 | no |
| 139 | 44 | 7.09 | 0.87 | 2.84 | 4.25 | no |
| 159 | hold | 6.90 | 6.90 | 6.90 | 0.00 | no |
| 159 | 42 | 6.90 | 0.94 | 1.63 | 5.27 | yes |
| 159 | 43 | 6.90 | 2.33 | 11.85 | -4.95 | no |
| 159 | 44 | 6.90 | 0.42 | 0.50 | 6.39 | yes |

Pooled CEM success: **2/12 (16.7%)**. Mean initial/minimum/final distance: 8.25/1.07/7.64 cm; mean distance reduction: 0.61 cm. Eight of twelve runs finish closer than hold: case 99 and 139 in all three seeds, case 159 in two seeds, case 23 in none. Hold stays stationary and succeeds in zero cases. Passing near the target is insufficient: case 23 sometimes reaches sub-centimeter distance but moves away again. Case 159 seed 43 also fails, showing seed sensitivity.

| Case | Mean final cm | Success | Runs beating hold | Possible exploitation steps |
|---|---:|---:|---:|---:|
| 23 | 17.93 | 0.0% | 0/3 | 11 |
| 99 | 5.01 | 0.0% | 3/3 | 25 |
| 139 | 2.96 | 0.0% | 3/3 | 10 |
| 159 | 4.66 | 66.7% | 2/3 | 14 |

| Seed | Mean final cm | Mean reduction cm | Success |
|---|---:|---:|---:|
| 42 | 7.04 | 1.21 | 25.0% |
| 43 | 8.04 | 0.21 | 0.0% |
| 44 | 7.84 | 0.41 | 25.0% |

## Timing and prediction failures

| Case/seed | Mean ms | Median ms | P95 ms | Max ms | Overruns | Rejections | Exploitation flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| 23/42 | 53.5 | 50.7 | 77.1 | 91.7 | 0 | 34 | 1 |
| 23/43 | 50.0 | 48.7 | 52.8 | 87.7 | 0 | 0 | 2 |
| 23/44 | 54.5 | 50.7 | 76.8 | 96.3 | 0 | 1 | 8 |
| 99/42 | 55.2 | 52.2 | 70.7 | 92.4 | 0 | 0 | 10 |
| 99/43 | 52.3 | 50.8 | 63.0 | 91.8 | 0 | 0 | 5 |
| 99/44 | 51.8 | 51.4 | 58.6 | 65.8 | 0 | 0 | 10 |
| 139/42 | 53.8 | 50.7 | 80.8 | 93.9 | 0 | 0 | 8 |
| 139/43 | 58.4 | 52.5 | 94.5 | 115.6 | 2 | 0 | 0 |
| 139/44 | 52.4 | 50.8 | 61.3 | 81.2 | 0 | 0 | 2 |
| 159/42 | 53.9 | 51.0 | 79.4 | 95.9 | 0 | 0 | 4 |
| 159/43 | 55.0 | 52.9 | 66.2 | 71.4 | 0 | 0 | 5 |
| 159/44 | 53.6 | 50.3 | 73.9 | 93.6 | 0 | 1 | 5 |

Across 600 plans: mean 53.7 ms, median 50.9 ms, P95 73.8 ms, maximum 115.6 ms; 2 missed 100 ms. There were 36 rejected candidate evaluations, zero fallbacks and zero actual joint-limit excursion samples. Timing is machine/load dependent. The CEM viewer smoke test, run while the broader test suite was active, logged additional overruns up to 155 ms. This is not a hard real-time controller.

Sixty selected first steps satisfy the diagnostic threshold: predicted displacement >=5 mm, actual displacement <=1 mm. These are evidence of possible model exploitation, not proof of its sole cause. The displacement error below compares matching one-step predictions and executed commands, using the actual observed starting state.

| Case/seed, zero-based step | Predicted motion mm | Actual motion mm | Vector error mm | Predicted target progress mm | Actual target progress mm |
|---|---:|---:|---:|---:|---:|
| 23/44, 5 | 43.623 | 0.418 | 43.711 | 27.574 | -0.355 |
| 99/44, 7 | 28.208 | 0.000 | 28.208 | 28.141 | 0.000 |
| 139/42, 38 | 11.948 | 0.000 | 11.948 | 11.935 | 0.000 |
| 159/44, 9 | 13.977 | 0.000 | 13.977 | 12.642 | 0.000 |

Across the 600 selected first steps, mean vector displacement error is 6.01 mm, P95 24.97 mm. All component displacement vectors, target progress values and flagged steps remain in JSON. Long-horizon forecasts are unverified; apparent predicted progress alone does not establish successful physical pushing.

## Verification

- Focused controller suite: 15 tests passed (checkpoint equivalence/guards, recursion, interpolation, cost/previous penalty, elite fitting/selection, best across iterations, legal commands, deterministic reset/warm start, no physics during planning, dynamics-dependent actions, infeasible fallback, first-only execution, identical initialization, hold baseline, tolerance and viewer decorations).
- Both passive viewer modes launched and closed successfully (`--seconds 3` CEM and `--seconds 2` no-action with a custom target); key mapping and decoration-only geometry are also exercised headlessly. No manual assessment of keyboard input or rendering aesthetics was performed.
- Existing model XML, all saved NPZ datasets and all saved PT checkpoints: 29 SHA-256 hashes unchanged before/after the evaluation, including the default checkpoint and independent dataset. Hashes are recorded in config and summary.
- Full suite: 58 tests passed in 71.81 s, covering all existing tests and the first 13 new controller tests. The final focused run passed all 15 controller tests after adding explicit elite/best-across-iterations and overwrite-guard checks. Pytest required an explicit workspace `--basetemp` because its default Windows temporary directory was inaccessible.

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_cache/cube_cem_verify_1
.venv/Scripts/python.exe -m pytest tests/test_cube_mpc.py -q --basetemp .pytest_cache/cube_cem_focused_2
```

Use a fresh temporary directory name when repeating these commands. The benchmark execution itself ran no training or dataset collection; some pre-existing regression tests independently generate temporary fixture datasets and train tiny fixture models.
