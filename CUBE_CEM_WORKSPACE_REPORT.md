# Workspace-aware CEM score experiment

The new score prevents one demonstrated escape, but these initial weights **do not improve overall actual MuJoCo outcomes**. Original success is 2/12; workspace-aware success is 0/12. Mean final target distance grows from 7.64 cm to 15.09 cm; escape frequency stays 1/12 and peak speed increases. The original score remains the default. The added score is available explicitly as `--scoring workspace`.

## Commands

Run from the project root in PowerShell:

```powershell
# New score, known original escape scenario
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --episode 23 --seed 42 --scoring workspace
# Original controller, identical initial scenario and seed
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --episode 23 --seed 42 --scoring original
# Repeat the full paired comparison in a fresh directory
.venv/Scripts/python.exe -m simple_arm.cube_mpc compare --output runs/cube_cem_workspace_repeat
# Example weight overrides
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --scoring workspace --reachability-weight 10 --escape-velocity-weight 1 --terminal-cube-velocity-weight 0.2
```

`compare` always evaluates both scores with identical CEM settings. Its default output is `runs/cube_cem_workspace/`, and it refuses an existing directory. `evaluate --scoring workspace --output runs/<new-name>` also works with the existing hold-baseline evaluation. Viewer controls and display are unchanged.

## Minimal implementation change

`simple_arm/cube_workspace.py` contains the geometric workspace and five extra weighted cost components. `cube_mpc.py` exposes the scoring choice/configuration and logs the selected score breakdown. The existing Gaussian knots, candidates, elites, updates, warm starts, 15-step horizon, 5 iterations, command limits, rejection rules and 100 ms simulation-time replanning interval are unchanged. The only effect on candidate selection is the score. `cube_mpc_comparison.py` adds a separate paired benchmark without changing execution or the old experiment artifacts.

All predicted states still come from the frozen learned world model. No MuJoCo rollout, collision query, contact detector, retraining or RL is used by the score. Geometry is parsed from the existing robot XML and limits; it is not inferred from pushing demonstrations. All twelve rerun original controllers reproduce the historical executed command arrays exactly.

## Workspace derivation

For link lengths L1=0.30 m and L2=0.25 m, the pusher center is

```text
p(q1,q2) = shoulder_origin + [L1*cos(q1)+L2*cos(q1+q2),
                            L1*sin(q1)+L2*sin(q1+q2)]
q1 in [-1.57,1.57], q2 in [-2.1,2.1] radians
```

Inside the tip workspace is tested using both inverse-kinematic branches and wrapped joint intervals. Outside distance is computed analytically against the circular arcs obtained from all four joint-range edges and the singular curves sin(q2)=0 that lie within the limits. This includes the inner hole and angular joint restrictions; it is not a radial cutoff or an arbitrary box. Geometry and limits are read from the current robot, and the planner passes its actual joint limits.

The tip set is dilated by pusher radius (0.022 m) plus cube inscribed radius (0.035 m), giving 0.057 m. This is a conservative, yaw-independent **contact-capable cube-center workspace proxy**. It does not guarantee that the arm can approach from a useful direction, avoid every obstacle or complete a push. A cube just outside this conservative set could still be touched at a corner. Therefore the evaluation also records an unambiguous escape witness: center radius beyond L1+L2+pusher_radius+cube_circumradius = 0.62150 m. Beyond that outer bound, even the cube corner cannot meet the pusher. Both reported escaped runs exceed that stronger bound.

## Added scoring terms and defaults

Let e(x) be meters outside the dilated workspace (zero inside), v the predicted planar cube velocity, u=norm(v), and dt=0.1 s. Let ecoast=e(x+dt*v). This short constant-velocity projection measures boundary-crossing risk using the predicted velocity; it is a scoring calculation, not a second simulator or extra model horizon.

| Component / flag | Weighted contribution | Default |
|---|---|---:|
| workspace / `--reachability-weight` | weight * mean((e/0.1 m)^2) | 10 |
| excess_cube_speed / `--cube-speed-weight` | weight * mean((max(u-vfree,0)/vfree)^2) | 0.1 |
| escape_velocity / `--escape-velocity-weight` | weight * mean((max(ecoast-e,0)/(dt*vfree))^2) | 1 |
| terminal_workspace / `--terminal-reachability-weight` | weight * (ecoast_final/0.1 m)^2 | 20 |
| terminal_cube_velocity / `--terminal-cube-velocity-weight` | weight * (u_final/vfree)^2 | 0.2 |
| free speed / `--free-cube-speed` | vfree in m/s, speed allowance and normalization | 0.25 |

The original target-distance, terminal-target-distance, command-change and joint-velocity terms retain their original weights. Normal pushing below 0.25 m/s in the workspace has no running speed or escape penalty. Equal-magnitude inward and outward velocities differ near the boundary: inward recovery has no escape-velocity penalty, while outward crossing does. Terminal residual velocity is penalized even below the running speed allowance, because reaching the target while still flying past it is undesirable. The terminal score considers position, velocity, and short coasting escape risk. No trajectory is rejected solely because its cube is already outside; recovery remains possible. Set all five new weights to zero to recover the original score and seeded CEM sequences exactly.

## Paired actual-physics evaluation

Four saved independent stationary-start scenarios (23,99,139,159), seeds 42/43/44, five simulated seconds per run: 12 runs for each score. Both start from the exact same saved state and target, use the same frozen Medium checkpoint, and use the original CEM settings. Success means remaining within 2 cm for the final 0.5 seconds, measured at every 2 ms physics step. Escape statistics use every physics step; an outside excess >1 micrometer is a numerical tolerance only. Maximum velocity is planar cube speed measured in MuJoCo, not a forecast.

| Metric | Original | Workspace-aware |
|---|---:|---:|
| Success rate | 16.7% (2/12) | 0.0% (0/12) |
| Mean final target distance | 7.64 cm | 15.09 cm |
| Runs escaping workspace | 1/12 | 1/12 |
| Outside fraction of all physics samples | 4.34% | 6.81% |
| Mean of per-run peak cube speeds | 1.186 m/s | 1.415 m/s |
| Maximum cube speed | 1.754 m/s | 2.144 m/s |

| Case / seed | Original final cm | New final cm | Success old/new | Escaped old/new | Peak speed old/new m/s |
|---|---:|---:|---|---|---|
| 23/42 | 18.78 | 9.32 | False/False | True/False | 1.566/1.438 |
| 23/43 | 11.88 | 39.89 | False/False | False/False | 1.239/2.125 |
| 23/44 | 23.15 | 39.33 | False/False | False/False | 1.754/2.144 |
| 99/42 | 5.11 | 6.41 | False/False | False/False | 0.951/0.691 |
| 99/43 | 5.05 | 20.17 | False/False | False/False | 0.637/1.168 |
| 99/44 | 4.86 | 2.89 | False/False | False/False | 1.694/1.203 |
| 139/42 | 2.64 | 12.55 | False/False | False/False | 1.318/1.003 |
| 139/43 | 3.38 | 3.56 | False/False | False/False | 0.848/0.872 |
| 139/44 | 2.84 | 25.76 | False/False | False/True | 0.821/1.731 |
| 159/42 | 1.63 | 10.60 | True/False | False/False | 0.736/1.897 |
| 159/43 | 11.85 | 2.24 | False/False | False/False | 1.657/1.037 |
| 159/44 | 0.50 | 8.38 | True/False | False/False | 1.010/1.672 |

Episode 23/seed 42 is the requested original batting failure: its actual cube-center radius reaches 0.64614 m, beyond the 0.62150 m any-contact outer bound, and it spends 52.12% of physics samples outside the workspace proxy. The updated run stays within the proxy (maximum center radius 0.51224 m) and improves final distance from 18.78 cm to 9.32 cm, but still does not succeed. Conversely, episode 139/seed 44 escapes only under the new score, spending 81.68% of samples outside and also exceeding the any-contact outer bound. Three of twelve pairs improve final distance; nine worsen it. These defaults are not a demonstrated control improvement.

## Score breakdown and plots

Every selected trajectory has all weighted terms in `diagnostics[step].score_components` inside its per-run JSON. Their sum equals its logged cost (up to floating-point summation). The baseline logs only the four original terms. The saved `score_breakdown.png` stacks mean selected-score components per run; these are neural planning costs, not measures of actual physical success. `actual_workspace_paths.png` shows the actual paths, targets and analytically derived workspace boundary.

Example: episode 23/42, first updated selected trajectory:

| Component | Contribution |
|---|---:|
| target | 0.18778040 |
| terminal_target | 0.00154608 |
| command_changes | 0.00038282 |
| joint_velocity | 0.00095433 |
| workspace | 0.00000000 |
| excess_cube_speed | 0.00122686 |
| escape_velocity | 0.00000000 |
| terminal_workspace | 0.00000000 |
| terminal_cube_velocity | 0.00030941 |
| Total | 0.19219990 |

The added penalties can be zero for a forecast even when actual dynamics later differ. This score has no mechanism to repair model errors. The new runs have 68 first-step possible-exploitation flags (predicted >=5 mm, actual <=1 mm), versus 60 with the original score. These flags indicate model mismatch; they do not establish a unique cause of failure. Predicted terminal positions and velocities remain unverified planning forecasts and are not compared directly to later closed-loop outcomes after replanning changes commands.

## Timing, preservation and checks

| Planner | Mean ms | Median ms | P95 ms | Max ms | >100 ms / 600 |
|---|---:|---:|---:|---:|---:|
| original | 95.5 | 89.6 | 152.8 | 252.7 | 160 |
| workspace | 131.1 | 127.2 | 175.6 | 233.2 | 570 |

The analytic workspace score adds CPU work. Timing includes variable machine load; parts of the comparison overlapped test/viewer checks. These numbers are not an isolated timing benchmark. Simulation time still advances exactly 50 physics steps per selected action; optimization effort is never reduced. The viewer logs overruns and runs slower than real time when needed.

- Full suite: 67 tests passed in 134.88 seconds. Final focused suite: 23 tests passed after adding an additional zero-weight CEM-equivalence test and retaining the original default. Tests cover FK distance against a dense independent forward-kinematics reference, joint/angular reach, cube/pusher sizes, speed allowance, inward/outward asymmetry, terminal terms, exact baseline arithmetic, seeded CEM/warm starts with zero weights, no MuJoCo planning, score breakdown, actual escape metrics and overwrite guards.
- Workspace and original viewer modes both launched and closed successfully in smoke tests.
- All 29 existing dataset/checkpoint/XML source hashes match before and after. All 38 prior `runs/cube_cem_initial/` files match their pre-run hashes. No checkpoints, datasets or prior experiment outputs were changed.
- New artifacts are in `runs/cube_cem_workspace/`: paired per-run JSON/NPZ traces, config/provenance, summary, actual workspace paths, and score breakdown. All unsuccessful trials remain available.

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_cache/cube_workspace_full1
.venv/Scripts/python.exe -m pytest tests/test_cube_mpc.py tests/test_cube_workspace.py -q --basetemp .pytest_cache/cube_workspace_focused2
```

Use a fresh temporary-directory name for another test run. The model and dataset hashes, baseline reproduction, and paired outcomes establish what changed; these initial results do not justify promoting the experimental score to default.
