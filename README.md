# First MuJoCo arm

A native Windows simulation of a two-joint horizontal pushing arm. Physics runs
on the CPU; the interactive viewer uses OpenGL on the Intel integrated GPU.

## Setup and launch

Verified host: Windows 11 Home, build 26200, x64; Python 3.13.3;
Intel Core Ultra 5 226V; Intel Arc 130V, driver 32.0.101.8860; Git 2.54.0.
The global `python` command points at a Windows Store alias on this machine.
Use the project interpreter below, or choose it through VS Code's
**Python: Select Interpreter** command. Recommended extensions are provided.

From this folder in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m simple_arm
```

Activation is optional. If PowerShell blocks the activation script, use:

```powershell
.\.venv\Scripts\python.exe -m simple_arm
.\.venv\Scripts\python.exe validate.py
```

To recreate the environment with the installed Python launcher:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Requirements pin the two direct dependencies tested here. MuJoCo installs its
own viewer dependencies automatically; no separate physics engine installation
is required. F5 in VS Code launches the interactive module.

## Controls

Focus the viewer window. Each key press changes its joint target by 0.1 rad:

| Key | Action |
| --- | --- |
| 6 / 7 | Decrease / increase shoulder target |
| 8 / 9 | Decrease / increase elbow target |
| Enter | Reset state and targets |
| Window close | Exit |

Ctrl+C in the terminal also exits. Joint angles and cube position print once
per second. Press 7 several times to sweep the pusher into the orange cube.
Use the top-row number keys without modifiers. These controls avoid MuJoCo's
built-in shortcuts; letters retain their visualization functions, and Esc
returns to the free camera.
Targets are clipped to joint control ranges. Actual joint angles can differ
under contact load because actuator torque is limited. Viewer mouse controls:
left-drag rotates, right-drag translates, scroll zooms. The callback provided
by MuJoCo avoids global keyboard hooks and extra input libraries. Commands are
queued to the physics thread; viewer state changes use its lock.

## Model and physics

Edit `models/arm.xml`, then restart. Units are meters, kilograms, seconds, and
radians. A **body** is a rigid coordinate frame; nested body positions are
relative to their parent. A **geom** supplies collision shape, appearance, and
mass properties. A **joint** permits motion relative to the parent body: the
two hinge axes are local +Z. The base has no joint, so it is fixed to the world.
The cube's free joint permits translation and rotation in all six degrees of
freedom. An **actuator** applies joint force: these position servos use target
angle, stiffness `kp`, damping `kv`, and a torque limit of ±8 N m.

The world has +Z up and gravity downward. Link lengths are 0.30 and 0.25 m;
the elbow body's `pos` and upper link's `fromto` must change together when
resizing the upper link. Capsule `size` is radius; box `size` gives half extents.
The 0.10 kg cube is 70 mm wide and starts above the floor to demonstrate gravity.
Link and pusher masses are explicit; MuJoCo derives their inertias from geometry.
Default joint damping and contact friction are grouped near the XML top.
The 2 ms physics timestep uses `implicitfast`; ten substeps form each 20 ms action.
The loop paces those actions using a monotonic wall clock and recovers from slow
frames without an unbounded catch-up loop. This is approximate real time.

## Code and observations

```text
models/arm.xml           Physical model
simple_arm/simulation.py ArmSimulation: reset, step, observations
simple_arm/__main__.py   Viewer, keys, pacing, status output
validate.py             Repeatable headless physical checks
.vscode/                Interpreter, launch, extension recommendations
requirements.txt        Direct dependency versions
```

Programmatic use from this folder:

```python
from simple_arm import ArmSimulation

simulation = ArmSimulation()
observation = simulation.reset()
next_observation = simulation.step([0.3, -0.2])
```

`step(action)` accepts two absolute position targets in radians and returns
the next observation. It rejects incorrect shapes and nonfinite values.
Observations contain simulation time, two joint positions (rad), two joint
velocities (rad/s), cube center position (m), orientation quaternion (w,x,y,z),
linear velocity (m/s), and angular velocity (rad/s). Cube vectors use world
coordinates. Returned arrays are copies, safe to retain for future datasets.
`reset()` restores the original XML state, clears controls and velocities, and
returns an observation. `get_observation()` does not advance time.

## Validation and next milestone

Run `validate.py` for model loading, gravity, floor contact, actual arm-to-cube
contact and displacement, both joint servos, reset, observation validity, and
action validation. A timed viewer smoke run is available with
`python -m simple_arm --seconds 5`. Successful window startup alone does not
verify manual key input or visual appearance; check those interactively.

On 2026-11-03, all headless checks passed with MuJoCo 3.17.0 and NumPy 2.5.4;
the contact sweep displaced the cube 0.322 m. `pip check` passed. The native
viewer ran for five seconds, logged advancing physics, and exited with code 0.
Manual keyboard response and rendered appearance still need human verification.

Next, add a separate collector that saves `(observation, action, next_observation)`
transitions to NumPy files with model/dependency versions, action timing, and
episode boundaries. Begin with repeatable scripted pushes before training a
dynamics predictor. Keep collection independent of the viewer.

Reference: [MuJoCo Python and passive viewer documentation](https://mujoco.readthedocs.io/en/latest/python.html).

## CPU training environment

The optional development environment adds CPU PyTorch, pytest, and Matplotlib
to the same `.venv`. It uses the existing Python 3.13 interpreter. Install the
pinned versions with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
```

The development requirements include the simulation requirements and the
official PyTorch CPU index; the `+cpu` version pin selects the CPU build.
This does not change global pip settings. No CUDA setup is needed.

Run the synthetic learning demonstration and the tests:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.training_smoke
.\.venv\Scripts\python.exe -m pytest -q
```

In VS Code, select **Run CPU training smoke test** in Run and Debug, or use
the Testing sidebar to discover and run pytest tests. If VS Code previously
selected another interpreter, select `.venv\Scripts\python.exe` explicitly.

The smoke test learns the known equation `y = 2*x0 - x1 + 0.5` from synthetic
samples. A tensor is a numerical array used by PyTorch. The small multilayer
perceptron maps two inputs through a hidden layer to one prediction. Mean
squared error measures the difference from the known answer; `backward()`
computes parameter gradients, and Adam uses them to update the weights.
Separate validation samples measure improvement on examples outside training.

The run uses seed 42, CPU tensors, 512 training examples, 128 validation examples,
and 200 optimizer steps. It checks finite losses and gradients, changed weights,
and final validation error below 25% of its initial value. A saved checkpoint
is reloaded and must reproduce the same predictions. Fixed seeds aid repeatability;
exact results can still vary across dependency versions and hardware.

Outputs under ignored `runs/training_smoke/` are replaced on each run:

| File | Contents |
| --- | --- |
| `loss.png` | Training and validation error curves, saved without a GUI |
| `metrics.json` | Loss history, seed, sample counts, and dependency versions |
| `model.pt` | Model weights; reload with `make_model()` from the smoke module |

`simple_arm/training_smoke.py` is the standalone learning demonstration;
`tests/` checks its artifacts and runs the existing physics validation.
This verifies the development tools, not learned robot physics. Next, collect
and inspect robot transitions as described in `ML_plan_1.md`, before adapting
the network to predict physical state changes.

## First robot learning experiment: arm dynamics

Run the complete CPU experiment from this folder:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.arm_learning run
```

This collects 5,000 transitions (100 episodes of 5 seconds), trains a small
network for 150 epochs, and evaluates the best validation checkpoint on held-out
episodes. It runs headlessly, without real-time pacing. The interactive simulation
and its model remain unchanged. The collector moves the cube to (1.2, 1.2) m,
outside the arm's 0.55 m reach, to isolate arm motion from cube contact.

The data combines small movements, larger single-joint movements, repeated
reversals, and simultaneous joint movements. Initial joint angles vary within
each category. Targets change every 0.5 s and remain fixed within each 100 ms
transition. Zero initial velocities provide stationary starts; later transitions
include motion and reversals. Each category is split by entire episode into
approximately 70% training, 15% validation, and 15% test; default counts are
68/16/16 because each category's counts are rounded separately.

The six inputs are `[shoulder angle, elbow angle, shoulder velocity, elbow
velocity, shoulder target, elbow target]`, in radians and rad/s. The network is
`6 -> 64 -> ReLU -> 64 -> ReLU -> 4`, with a linear output predicting changes
in the first four quantities over 100 ms. Add the predicted changes to the
current state to obtain the next state. Time and cube state are not inputs.
Input and delta normalization use training-only means and standard deviations;
near-constant dimensions use a scale floor of 1e-6. Those statistics are saved
with the weights. Adam uses learning rate 0.001 and shuffled batches of 128.
Training minimizes normalized delta MSE, retaining the epoch with the lowest
validation MSE. Test results do not select the checkpoint.

You can run stages separately and adjust the experiment size:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.arm_learning collect --episodes 100 --steps 50 --seed 42
.\.venv\Scripts\python.exe -m simple_arm.arm_learning train --epochs 150
.\.venv\Scripts\python.exe -m simple_arm.arm_learning evaluate
```

Use `--dataset PATH` and `--output DIRECTORY` for separate experiments. Defaults
overwrite `data/arm_dynamics.npz` and outputs in `runs/arm_dynamics/`; preserve
earlier experiments by choosing different paths. The dataset contains states
shaped `(episodes, steps+1, 4)`, actions `(episodes, steps, 2)`, per-episode split
and scenario labels, and JSON metadata. The metadata records timing, units,
seed, dependency versions, MJCF hash, and a data fingerprint. The adjacent
`data/arm_dynamics.json` summarizes counts and state/action ranges for inspection.

Outputs in `runs/arm_dynamics/`:

| File | How to use it |
| --- | --- |
| `model.pt` | Best validation weights, normalization, and experiment metadata |
| `history.json`, `loss.png` | Check training versus validation error |
| `evaluation.json` | Per-joint one-step errors, scenario errors, and 0.1–1 s rollout errors |
| `rollout_error.png` | Compare angle-error growth against both baselines |
| `trajectory.png` | Compare both joint angles on the first held-out episode |

Evaluation reports angle RMSE in degrees and velocity RMSE in rad/s, in
shoulder/elbow order. The no-change baseline preserves the entire state. The
constant-velocity baseline advances angles by angular velocity times 0.1 s and
preserves velocity. Each rollout starts from a measured state, then feeds only
predicted states and the recorded future actions back into the predictor. Errors
at each horizon average over all available test-episode windows, without crossing
episode boundaries. The trajectory plot shows one example, not an aggregate.

The learning tests check deterministic collection, timing, episode splitting,
state/delta definitions, physical error units, normalization, checkpoint reload,
evaluation artifacts, and the ability to overfit a tiny physical training batch.
Run them with `python -m pytest -q`. Passing these tests verifies implementation;
scientific success requires beating the baselines on held-out episodes. Poor
performance is an experimental finding, not automatically a software bug.
These held-out episodes test new trajectories within the collection ranges;
they do not establish generalization to new masses, friction, or contact physics.

The first default run (seed 42, 5,000 transitions, 150 epochs) produced the
following held-out angle errors; these are measurements, not test thresholds:

| Predictor | 100 ms shoulder / elbow RMSE | 1 s shoulder / elbow RMSE |
| --- | --- | --- |
| Learned | 0.31 / 0.45 degrees | 0.44 / 0.58 degrees |
| No change | 13.16 / 18.33 degrees | 27.70 / 38.01 degrees |
| Constant velocity | 8.69 / 16.02 degrees | 136.09 / 157.86 degrees |

The constant-velocity baseline cannot anticipate servo braking or new targets,
which explains its large longer-horizon errors. Six pytest tests passed after
adding this experiment. Use your generated reports as the source for new runs.

For a longer audit of the existing checkpoint, run:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.rollout_stress
```

This evaluates 48 new five-second episodes, including rapid reversals and
near-limit targets, without retraining or correcting predicted states. See
`ROLLOUT_STRESS_REPORT.md` for results and limitations, and
`runs/rollout_stress/` for full metrics, raw trajectories, and plots. These
harder tests reveal substantially larger errors than the original evaluation.

## Broader training data

The optional `broad` collection profile retains the four original motion
categories and adds irregular full-range targets, fast reversals, near-limit
holds, and near-limit reversals. New target durations range from 0.1 to 1.3 s,
depending on category. Some commands reach the exact actuator limits. Added
categories vary initial angles across the workspace, with limit categories
starting at 90–99.5% of range. Two thirds of episodes start with randomly
sampled angular velocities (shoulder ±5 rad/s, elbow ±7 rad/s). These moving
starts deliberately expose transients, including motion toward limits.

The same model architecture is trained from scratch to isolate the effect of
broader experience. Targets remain absolute joint positions. All learning
targets come from MuJoCo, not the old model's erroneous predictions. The cube
stays out of reach. Splits remain by whole episode and stratified by category;
normalization and checkpoint selection still use training and validation only.

Reproduce the larger run without overwriting the original dataset/checkpoint:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.arm_learning run --profile broad --episodes 800 --epochs 200 --seed 73 --dataset data/arm_dynamics_broad.npz --output runs/arm_dynamics_broad
```

This collects 40,000 transitions, split into 560/120/120 training/validation/test
episodes. To compare both models on identical fresh five-second stress cases:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.rollout_stress --seed 2027 --output runs/rollout_stress_original_fresh
.\.venv\Scripts\python.exe -m simple_arm.rollout_stress --checkpoint runs/arm_dynamics_broad/model.pt --dataset data/arm_dynamics_broad.npz --seed 2027 --output runs/rollout_stress_broad_fresh
```

Each checkpoint is paired with its own training dataset for provenance and
coverage reporting, but both stress evaluations use identical fresh actions
and initial states. Seed 2027 is distinct from the original stress audit's
seed 2026 and is not used for training or selecting checkpoints. Broadening
coverage does not itself establish reliability for MPC or stop impacts.

See `BROADER_TRAINING_REPORT.md` for the completed run's measured comparison.
The broader model markedly improves fresh five-second reversal/near-limit
predictions, though small predicted limit overshoots remain.

## Learned model predictive control

Run the reproducible headless benchmark, using the frozen broader checkpoint:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.mpc
```

Watch either controller on one of the same scenarios:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.mpc --viewer mpc --scenario moving_target
.\.venv\Scripts\python.exe -m simple_arm.mpc --viewer direct --scenario near_limits
```

The viewer closes at the scenario's end; window close or Ctrl+C exits earlier.
The MPC short-scenario viewer was launched successfully and exited cleanly.
Interactive visual inspection is still useful. Automated evaluation is headless.

### Algorithm and interfaces

`simple_arm/mpc.py` reuses `ArmSimulation`, the training code's `load_predictor`,
and the unchanged six-input/four-delta state convention. It validates checkpoint
state order, 100 ms timing, positive normalization scales, and matching MJCF hash.
The predictor runs in inference mode on CPU, batching all candidates at each
prediction step. No model weights or training datasets are changed.

At every control tick, `LearnedMPC.act(observed, goal, previous)` samples candidate
command sequences, rolls out predicted states recursively, scores the resulting
trajectories, and returns the first command of the best sequence with diagnostics.
Only that command executes for 100 ms. The next plan begins at the newly observed
MuJoCo angles and velocities. Feedback between plans is deliberate; there are
no measured-state corrections inside a candidate's predicted rollout. The planner
holds the currently observed goal constant over its horizon; it does not preview
scheduled future goal changes.

Defaults are 256 candidates, eight model steps (0.8 s), seed 42, and two PyTorch
CPU threads. Candidates include the constant goal, constant previous command,
the shifted prior best plan, random constant commands, and smooth random
perturbations around the goal/prior plan. This is random shooting with a warm
start: no optimization gradients or additional training. Including the goal
sequence lets the planner choose the direct baseline when it scores best.

Commands use the intersection of actual actuator ranges with joint ranges inset
by a configurable 0.08 rad margin. Joint limits are read by joint name from MJCF;
actuator ranges are read from the existing simulation. Only commands are clipped
into the feasible range. Predicted states are never clipped. Any candidate with
a predicted joint-limit violation receives infinite cost. The shared predictor
fails explicitly if it produces nonfinite states; the cost function also rejects
nonfinite supplied trajectories.
If all candidates are rejected, the controller explicitly reports a fallback and
holds the previous feasible command. This is a legal fallback, not a guarantee
that momentum cannot carry the real arm toward a limit.

### Objective and configuration

For predicted states after each action, minimize the sum of intermediate costs
plus terminal costs:

```text
sum_k [ w_position * ||q[k] - goal||^2
      + w_velocity * ||velocity[k]||^2
      + w_change   * ||command[k] - command[k-1]||^2 ]
+ w_terminal_position * ||q[last] - goal||^2
+ w_terminal_velocity * ||velocity[last]||^2
```

For the first change penalty, `command[-1]` is the previous **executed command**,
not the observed joint angles or a previous unexecuted candidate. Positions and
commands use radians; velocities use rad/s. Costs are unnormalized physical-unit
squared errors, summed over both joints. There is no timestep factor; changing
the horizon changes the relative contribution of accumulated and terminal costs.

| Config field | Default | Contribution to dimensionless cost |
| --- | --- | --- |
| `position_weight` | 1.0 | 1 rad position error costs 1 per step; units rad^-2 |
| `velocity_weight` | 0.03 | 1 rad/s residual speed costs 0.03 per step; units s^2/rad^2 |
| `change_weight` | 0.08 | 1 rad command change costs 0.08; units rad^-2 |
| `terminal_position_weight` | 8.0 | Extra final-pose penalty; units rad^-2 |
| `terminal_velocity_weight` | 0.1 | Extra final-speed penalty; units s^2/rad^2 |
| `command_margin` | 0.08 rad | Distance from nominal joint limits for candidate commands |

Joint-limit violations are a hard rejection rather than a tunable finite penalty.
Use `--horizon`, `--candidates`, and `--seed` for planning settings. A JSON object
passed through `--weights PATH` overrides `Config` fields, for example:

```json
{"velocity_weight": 0.05, "change_weight": 0.1}
```

JSON fields take precedence over matching CLI settings. Use a separate output
directory for configuration comparisons:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.mpc --horizon 10 --candidates 256 --output runs/mpc_horizon10
```

### Benchmark and metrics

Five scenarios cover short/large moves, targets changing during motion, near-limit
goals, and direction reversals. Both controllers use identical initial state,
physics, target schedules, and 100 ms command intervals. The cube is out of reach.
The baseline simply sends the desired joint position to the existing servos,
clipped only to actuator limits. Scenario definitions are saved with the report
and available in `scenarios()` for extension.

Metrics use 2 ms physics samples: per-joint tracking RMSE, final absolute error,
maximum directional overshoot past each segment's target, final residual velocity,
nominal limit exceedance (>0.001 rad), and active joint-limit constraints. Settling
requires both joints within 0.02 rad and both speeds within 0.05 rad/s for the
remainder of that target segment, with at least 0.3 s remaining. Unsettled targets
have null settling time. Tracking errors include all commanded target changes.

Controller computation includes candidate generation, learned rollouts, and
selection, but excludes physics, plots, and file output. Times use wall clock;
they vary with system load. Reported deadline misses exceed 100 ms. In viewer mode,
the loop sleeps for the remaining interval; it runs slower if planning overruns.
The headless benchmark does not simulate computation delays in the physical plant.

New outputs are confined to `runs/mpc_initial/` by default:

- `report.json`: configuration, scenario definitions, per-scenario and aggregate
  metrics, versions, checkpoint fingerprint, and initial-plan audits.
- One comparison plot per scenario: actual/desired joint angles, actuator
  commands, velocities, tracking errors, and controller computation times.
- Per-controller NPZ trajectories, including 2 ms states, and JSON planner
  diagnostics including selected indices, predicted next states, and fallbacks.

The audit replays each initial selected plan and constant-goal sequence in MuJoCo
without feedback, and compares their objective costs with model predictions. It
is an offline diagnostic only; real physics never selects planner candidates.
Later replans are checked through their executed one-step prediction errors.
The audit is limited to the five initial plans, not a proof against model
exploitation across arbitrary future plans.

### Initial results and limitations

With default settings, both controllers settled all five final targets with
negligible final error/velocity, zero measured overshoot, and no joint-limit
violations or activated limit constraints. Pooled tracking RMSE was **20.70 /
22.95 degrees** (shoulder/elbow) for MPC versus **20.26 / 22.52 degrees** for
direct control. Mean final-target settling time was 0.807 versus 0.804 seconds.
MPC therefore provides no tracking advantage in this initial benchmark.

Planning averaged 5.6 ms (95th percentile 8.3 ms, max 41.6 ms), with no 100 ms
deadline misses or all-candidates-rejected fallbacks in the measured run.
These timings include normal laptop scheduling variability and are not hard
real-time guarantees.

MPC chose a first command different from the goal on 9 of 260 ticks. It is
using predicted costs, but the existing low-level servos already solve most of
this unconstrained joint-position task well. The large/near-limit initial selected
plans had lower objective cost than direct commands in both learned predictions
and MuJoCo replays; no initial-plan ranking reversal was found. That does not
establish an overall closed-loop benefit or rule out later model exploitation.

See `MPC_INITIAL_REPORT.md` for the per-scenario comparison and interpretation.
Tests cover normalization, recursive batch rollouts, legal deterministic command
generation, exact cost terms, model-dependent selection, rejected-plan fallback,
actual-state feedback, and reproducible trajectories. Run `python -m pytest -q`.
Hard-stop impacts, object contact, and arbitrary out-of-distribution commands
remain outside this experiment's claims. Further controller tuning should use
new validation scenarios rather than presenting this benchmark as untouched.

## Arm–cube contact learning

The separate contact experiment is implemented in `simple_arm/contact_simulation.py`
and `simple_arm/contact_learning.py`. It derives a new MuJoCo model from the
original XML without modifying it. The cube can translate in XYZ and rotate
about Z; locked roll/pitch prevent tipping while free Z preserves real gravity,
floor support, and friction. Arm geometry, actuators, cube mass, and friction
remain unchanged. No cube-targeting MPC is included.

Read `ARM_CUBE_EXPERIMENT.md` for the setup fixed before training, state/action
definitions, normalization and loss details, measured results, and failures.
The 13-value state contains arm angles/velocities, cube XYZ, sin/cos yaw, world
linear velocity, and yaw rate. The MLP takes that state plus two joint position
targets, using current pusher-relative X/Y as an invertible coordinate change.
It predicts 12 residual quantities; orientation updates compose yaw rotations.
Contact labels never enter the network inputs.

Reproduce collection and the two first-pass interval models:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_learning collect
.\.venv\Scripts\python.exe -m simple_arm.contact_learning train --stride 5 --epochs 100
.\.venv\Scripts\python.exe -m simple_arm.contact_learning train --stride 1 --epochs 100 --output runs/arm_cube_dynamics_20ms
```

Both intervals use the same 360 four-second episodes, split 252/54/54 by whole
episode. Commands are held for 100 ms. There are 72,000 transitions at 20 ms or
14,400 at 100 ms. Both first-pass models learn useful short-term contact responses,
but drift substantially in long recursive rollouts. The shorter interval alone
does not fix this.

Refine a separate 100 ms copy using one-second recursive training windows and
checkpoint selection on four-second validation rollouts:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_learning refine --epochs 80 --output runs/arm_cube_dynamics_recursive
```

The selected model is `runs/arm_cube_dynamics_recursive/model.pt`. It uses the
same three-hidden-layer, 128-unit SiLU MLP, not an extra network. The original
first-pass models and all arm-only/MPC artifacts remain available.

Generate independent evaluation episodes and compare frozen models:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_learning collect --episodes 120 --seed 92 --dataset data/arm_cube_unseen.npz
.\.venv\Scripts\python.exe -m simple_arm.contact_learning evaluate --dataset data/arm_cube_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics/model.pt --output runs/arm_cube_unseen_firstpass
.\.venv\Scripts\python.exe -m simple_arm.contact_learning evaluate --dataset data/arm_cube_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics_20ms/model.pt --stride 1 --output runs/arm_cube_unseen_20ms
.\.venv\Scripts\python.exe -m simple_arm.contact_learning evaluate --dataset data/arm_cube_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics_recursive/model.pt --output runs/arm_cube_unseen_recursive
.\.venv\Scripts\python.exe -m pytest -q
```

`--all-episodes` is only for an independently generated dataset with a different
seed; all 120 fresh episodes are used for evaluation, never training. Without it,
evaluation uses the dataset's test split and checks checkpoint/data provenance.
Use distinct dataset/output paths for additional experiments; rerunning commands
with the same paths can overwrite this experiment's outputs.

The refined model achieved 100 ms cube XY displacement-vector RMSE of **2.90 mm**
versus **9.15 mm** stationary and **4.19 mm** constant velocity. Four-second
endpoint RMSE was **9.88 cm** versus **16.49 cm** for the cube baselines. On
moving episodes, mean predicted displacement direction cosine was **0.882**.
Every recursive step uses the previous predicted state with no truth corrections.

Important failures remain: stationary cubes accumulate false motion, yaw is weak,
and some predicted pushes slide far beyond the actual stopping point. Full
contact-conditioned metrics and successful/failed trajectory plots are in
`runs/arm_cube_unseen_recursive/`. Contact conditions use ground-truth labels
only for evaluation; onset, ongoing contact, release, and sliding masks can
overlap. The report explains the difference between intermittent ongoing contact
and rare uninterrupted contact, as well as the scope of the success claim.
All 23 tests and dependency validation passed. Cube-manipulation MPC must wait
for further predictive-model evaluation and improvement.

### Watch simulator motion versus model prediction

Launch a 3D overlay replay of the independently evaluated contact episodes:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --episode worst
```

Original arm/cube colors show recorded MuJoCo ground truth. Translucent magenta
shows the learned arm and cube prediction in the same coordinate frame. Both
start from the same initial state and receive the same command sequence; every
subsequent predicted state comes from the previous prediction. The viewer reads
the saved evaluation trajectories, without rerunning physics or training. It
interpolates between the 100 ms samples only for smooth display, so contact timing
within those samples is not resolved. Ghost geometry is visual only.

Focus the viewer: **6/7** previous/next episode, **8** pause/resume, **9** restart.
Mouse controls retain the standard MuJoCo camera behavior. Playback loops with a
brief hold at the end; the terminal reports episode index, time, and cube error.
Default speed is half real time. Use `--speed 1` for real time, `--episode 12` for
a specific zero-based replay index, or `--run runs/arm_cube_unseen_firstpass` to
inspect another saved evaluation. Close the viewer window to exit. This is a
recorded comparison, not an interactive controller or cube MPC.

### Broader two-joint contact data

The new `simple_arm.contact_varied` experiment varies both joints together,
independent command timing and speeds, coordinated movements, abrupt changes,
reversals, cube placement/yaw and initial cube motion. It preserves the original
collector and models. See [ARM_CUBE_VARIED_EXPERIMENT.md](ARM_CUBE_VARIED_EXPERIMENT.md)
for sampling bounds, coverage requirements, reproduction and results.

The completed experiment has 1,200 episodes / 48,000 transitions, with median
shoulder/elbow excursion of about 66 degrees each. On 120 independent episodes,
the refined model's four-second cube XY endpoint RMSE is **18.2 cm**, versus
**39.4 cm** for the old contact model, **24.2 cm** stationary and **34.9 cm**
constant velocity. Abrupt contact and stationary-cube drift remain weaknesses.

### Explore still cubes and fresh random scenarios

The viewer now defaults to the latest varied-model evaluation. To inspect its
largest average trajectory error among cases where the actual cube stays within
2 mm of its starting position at every saved sample:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --episode worst-still
```

This selects episode 77 in the current saved evaluation: actual maximum cube
excursion is 1.80 mm while the prediction drifts roughly 11 cm. This selection
checks the whole sampled path, rather than just final displacement. Browse only
these low-motion episodes with **6/7**. Use `--motion-threshold 0.01` to allow up
to 1 cm of actual movement instead.

For fresh randomized initial conditions and command sequences, run:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --random
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --random --condition still
```

Random mode varies initial joint configuration/velocities, both joint commands,
movement family, cube position and yaw. General random mode also varies initial
cube velocities; still mode starts the cube at rest and accepts only episodes
whose actual cube excursion stays within the threshold at every 2 ms physics
step. Invalid starts and actual joint-limit excursions are retried, with at most
100 attempts per episode. This exploratory mode does not require a successful
push or minimum joint excursion, unlike the contact-enriched training collector.
Physics properties remain fixed. It selects no episodes based on prediction
error and is an exploration tool, not a new benchmark result.

Each four-second episode is first simulated in MuJoCo, then independently rolled
out through the frozen network from its initial state using the same commands.
Both trajectories play together, with no corrections or clipping of predictions.
The magenta ghost remains visual only. The terminal displays actual maximum
cube motion, predicted displacement and instantaneous prediction error.

Episodes advance automatically after the final-frame hold. **7** advances early
or generates the next episode; **6** revisits earlier generated episodes; **8**
pauses and **9** restarts the current episode. Generating an episode can briefly
pause playback. Seeds are printed: the default starts at 2026 and increments for
each new episode. Use `--seed 2027` to reproduce that episode directly, keeping
the same condition and threshold. Use `--speed 1` for real-time playback or
`--checkpoint PATH` to select another compatible 100 ms contact model.
No datasets, checkpoints or previous evaluations are written by random mode.

### Learning contact versus near misses

The contact-boundary experiment adds groups of overhead passes, pusher near
misses, light contact and clear pushes. Each group shares its arm commands and
initial arm state, changing only cube placement. New examples are combined with
the broad dataset while preserving original episode splits. Geometry and the
36,620-parameter network remain fixed. See
[ARM_CUBE_CONTACT_BOUNDARY_EXPERIMENT.md](ARM_CUBE_CONTACT_BOUNDARY_EXPERIMENT.md)
for category definitions, reproduction commands, independent comparisons and
the drift-reduction / pushing-regression acceptance criteria.

The completed run combines **1,280 new episodes** with the previous 1,200 broad
episodes. The refined model reduces average maximum no-contact drift from
**29.1 to 15.9 mm (45.5%)**, while broad-push endpoint RMSE increases from
**13.0 to 13.6 cm (4.5%)**. It meets the predeclared 30% drift-reduction and
10% pushing-regression limits, but still predicts false motion in most near
misses. All 36 tests pass; existing datasets/checkpoints are preserved.

Watch the exact episode 77 discussed previously with the new prediction:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_contact_boundary/reference_new --episode 77
```

Maximum predicted cube motion in that case falls from 114 mm to 40 mm, with
actual motion still only 1.8 mm. To explore fresh low-motion cases with the new
checkpoint (rather than the viewer's existing default model):

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --random --condition still --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt
```

### Quick collection parallelism benchmark

```powershell
.\.venv\Scripts\python.exe -m simple_arm.benchmark_collection
```

Replays 16 accepted contact-boundary episodes (all four categories and movement
families), twice each, sequentially and with 2/4 Windows worker processes. Each
worker uses the real collector's 2 ms physics, contact and geometry-distance
checks. Startup/warm-up is timed separately, and result hashes must match exactly.
The benchmark creates no datasets or checkpoints. Use `--episodes 32 --repeats 3`
for a longer measurement.

Initial median batch times were 3.08 s sequential, 1.74 s with two workers and
0.91 s with four: **1.77x / 3.40x steady throughput**. Startup/warm-up took
2.09/2.81/3.44 s respectively. These short measurements cover accepted-episode
simulation/checking; placement search, rejected candidates, dataset writing and
training are excluded. The production collector remains sequential.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_varied_unseen_recursive
```
