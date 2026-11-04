# Arm–cube experiment

## Setup fixed before training

The original MJCF has 0.30/0.25 m links, a downward capsule pusher, Z-axis
hinges limited to ±1.57/±2.1 rad, and position servos with kp=35, kv=5,
±8 N m torque limits. Physics runs at 2 ms. The original 70 mm, 0.10 kg free
cube has geometry-derived inertia and floor friction 0.7; existing tests
demonstrate arm contact and displacement. Arm-only collection deliberately
placed it out of reach.

The new environment derives its MJCF from the original without editing it.
The cube has independent X/Y/Z slides and a yaw hinge: roll/pitch are locked,
but vertical motion remains free so gravity generates physical floor normal
forces and friction. All added joints override inherited damping/armature to
zero. Mass, contact geometry, gravity, and friction remain unchanged.

The complete constrained state has 13 numbers: arm q(2), arm velocity(2), cube
XYZ(3), sin/cos yaw(2), world linear velocity(3), yaw rate(1). Actions are two
absolute joint targets. A residual MLP predicts 12 increments: arm(4), XYZ(3),
wrapped yaw increment(1), linear velocity(3), yaw rate(1). Yaw updates compose
rotations, so predictions remain on the unit circle. Cube X/Y input coordinates
are relative to the current pusher tip (derived from current joint angles), an
invertible change of coordinates using known robot geometry, not contact labels.

Collect four-second episodes with procedural free motion, approaches, pushes,
sustained sweeps, reversals, and near-contact starts. Vary arm/cube poses, cube
yaw, direction, speed, and some initial velocities. Record arm–cube contact at
every 2 ms physics step. Split whole episodes, stratified by collection type.
Use shared 20 ms trajectories and commands held for 100 ms to compare 20 ms
and 100 ms models on identical physical experience. Both use a three-layer
128-unit MLP, normalized training-only inputs/targets, and contact-weighted
training loss. No cube-targeting MPC is added.

Evaluation uses held-out episodes only: separate free/onset/sustained/release/
sliding metrics, stationary and constant-velocity cube baselines, the original
arm model at 100 ms, and full four-second recursive rollouts. Longer rollouts
use only one measured initial state and future actions. Contact-conditioned
metrics use actual labels, never predicted-input contact information. Time-step
comparisons share episodes and physics, but have different transition counts;
they are an exploratory comparison, not a compute-matched ablation.

Existing models, datasets, and MPC outputs remain unchanged. New results will
be appended below after execution, including failures rather than assuming
that successful contact collection guarantees useful learned dynamics.

## Completed experiment

### Data and definitions

Training collection: seed 91, 360 four-second episodes, 72,000 transitions at
20 ms, also grouped into 14,400 transitions at 100 ms. Both first-pass models
use exactly the same episodes and 100 ms command holds. Episode split is
252 training / 54 validation / 54 test, stratified over six collection types.
240 episodes end with more than 2 cm cube displacement. Arm initial velocities
are nonzero in two thirds of episodes; cube initial velocities are zero.
Yaw, initial arm pose, approach direction, gap, and commanded speed vary.

Contact is detected using active, penetrating arm/cube geom pairs, excluding
floor contact, every 2 ms. Contact activity chatters: some windows contain both
onset and release. The final diagnostic definitions are:

- No contact: no contact anywhere in the interval, and cube speed at both ends
  below 5 mm/s.
- Onset/release: any off-to-on/on-to-off contact transition within the interval.
- Sustained: contact activity in consecutive intervals, allowing intermittent
  micro-separation. This means ongoing pushing, not uninterrupted contact.
- Sliding: no arm contact during the interval but cube endpoint speed above
  5 mm/s. Sliding includes post-release motion and is not a history-conditioned
  guarantee that the immediately preceding interval contained contact.

These masks overlap and do not partition the dataset. Final mask counts:

| Mask | 20 ms intervals | 100 ms intervals |
| --- | --- | --- |
| No contact | 60,176 | 11,811 |
| Onset | 7,665 | 2,322 |
| Ongoing contact | 9,723 | 2,089 |
| Release | 7,470 | 2,316 |
| Sliding | 1,379 | 191 |

The collection-time JSON snapshots predate the diagnostic definition change and
use a stricter, uninterrupted-contact mask (714 / 52 sustained samples). Updated
evaluation JSONs state the final definition explicitly. This diagnostic change
does not alter physics, training inputs, transition targets, or loss weights.
Strict uninterrupted contact at 100 ms was rare; do not claim this dataset
adequately tests continuous sticking contact.

### Networks and refinement

Both first-pass networks are `15 -> 128 -> SiLU -> 128 -> SiLU -> 128 -> SiLU
-> 12`, trained for 100 epochs on CPU with Adam at 0.001, batches of 256, seed
91. The 15 inputs contain the 13 state quantities and two actions. Input X/Y
positions use the current pusher-relative frame with world-aligned axes; all
remaining quantities retain their documented physical units and world frames.
No contact flags or future states are input features.

First-pass loss is normalized delta MSE with 5x weight on contact-containing
transitions. This uses contact as supervision, not as a privileged input.
Normalization uses training data only. Physical scale floors prevent nearly
constant Z quantities from being amplified by tiny measured standard deviations;
the exact scales are saved in checkpoints and constants are in the code.
Validation uses unweighted normalized one-step error. Thus training and
validation loss curves have different weighting and are not directly comparable.

The first-pass models had useful one-step predictions but poor recursive
stability. Both are preserved. A separate 100 ms copy was refined for 80 epochs
using 10-step (one-second) differentiable autoregressive training windows,
Adam at 0.0002, batches of 256, seed 93, and gradient norm clipping at 5.
State-error scales are explicit in `train_multistep`; contact steps receive 3x
loss weight. Normalization statistics remain those of the original training
split. Checkpoint selection uses complete four-second validation rollouts,
checked every five epochs; refinement epoch 15 was selected. No test data
selected the epoch. The same MLP is used; there is no additional neural network.

Refined checkpoint: `runs/arm_cube_dynamics_recursive/model.pt`.
SHA256: `5f48ed9deb0461521c9ed1862bf0112fc0bf572d12562def29c5e1179da60e25`.

### Fresh evaluation

After the refinement design was fixed, generated 120 independent episodes with
seed 92. All 120 are evaluation-only, including 80 moving-cube episodes.
First-pass 20 ms, first-pass 100 ms, and refined 100 ms predictors receive the
same physical trajectories (appropriately sampled). Exact equality of the
100 ms comparison's ground-truth states and actions was checked.

Four-second rollouts initialize once from a measured state and recursively use
predictions for every subsequent step; only action sequences are supplied.
No projection onto the floor, stationarity snapping, or state clipping is used.
Yaw composition enforces its representation's unit-circle identity, not a
physics correction. Contact-conditioned evaluation always uses actual labels.

| Model | 4 s endpoint XY displacement-vector RMSE |
| --- | --- |
| Stationary cube | 16.49 cm |
| Constant-velocity cube | 16.49 cm |
| First-pass 20 ms MLP | 34.22 cm |
| First-pass 100 ms MLP | 20.38 cm |
| Recursive-trained 100 ms MLP | **9.88 cm** |

The full-rollout baselines coincide because cube initial velocity is zero;
this is expected. Their one-step predictions differ because measured current
cube velocity is then available. The 20 ms model has 200 recursive steps rather
than 40; smaller one-step error alone does not imply better physical-time
rollouts. The interval comparison is not compute-matched: finer data produces
more training transitions and optimizer updates per epoch. No 20 ms recursive
refinement was attempted, so this does not establish 100 ms as intrinsically best.

### Refined model accuracy

On 4,800 fresh 100 ms transitions:

- Arm angle RMSE: 0.679 / 0.362 degrees; velocity RMSE: 0.129 / 0.057 rad/s.
- Cube XYZ RMSE: 1.79 / 2.29 / 0.091 mm; yaw RMSE: 1.45 degrees.
- Cube linear velocity RMSE: 0.0265 / 0.0326 / 0.0125 m/s;
  yaw-rate RMSE: 0.445 rad/s.
- XY displacement-vector RMSE: 2.90 mm versus 9.15 mm stationary and 4.19 mm
  constant velocity. Mean moving-sample direction cosine: 0.986.
- Using 2 mm displacement as the motion threshold, false-motion rate for
  stationary transitions is 13.5%, and missed-motion rate is 0.50%.

Contact-conditioned one-step XY displacement-vector RMSE:

| Actual contact condition | Samples | Learned | Stationary | Constant velocity |
| --- | --- | --- | --- | --- |
| No contact | 3,927 | 1.94 mm | 0.01 mm | 0.02 mm |
| Onset | 789 | 5.02 mm | 21.57 mm | 9.17 mm |
| Ongoing contact | 690 | 4.29 mm | 22.19 mm | 5.56 mm |
| Release | 785 | 5.17 mm | 22.17 mm | 9.32 mm |
| Sliding | 76 | 6.88 mm | 14.34 mm | 14.53 mm |

For the 80 moving episodes, four-second endpoint displacement-vector RMSE is
11.33 cm versus 20.20 cm stationary, displacement-magnitude MAE is 5.51 cm,
and mean direction cosine is 0.882. Across all episodes, magnitude MAE is
4.97 cm versus 12.04 cm stationary. Prefix-averaged rollout vector RMSE at
1 / 2 / 4 seconds is 3.70 / 6.09 / 7.93 cm; these are averages over all steps
up to each horizon, not endpoint-only errors.

At four seconds, arm angle RMSE is 1.59 / 3.04 degrees, cube yaw RMSE is 24.09
degrees, linear velocity RMSE is 0.0149 / 0.0234 / 0.0046 m/s, and yaw-rate
RMSE is 0.121 rad/s. Full per-condition arm, orientation, velocity, displacement,
and motion-detection metrics are in the machine-readable evaluations.

### What fails and what this demonstrates

The refined model predicts previously unseen pushing directions and approximate
displacements better than the simple cube baselines. The success criterion is
met statistically within these procedural pose/action families, not universally.
It is a useful contact predictor, not a reliable general physical world model.

Stationarity remains poor: all 40 stationary episodes accumulate more than
2 mm of false displacement after four seconds. Their median drift is 2.39 cm
and worst drift 23.24 cm. Some pushes continue to move the predicted cube long
after actual release/friction has stopped it; the worst-push plot exposes this.
Yaw endpoint error is worse than the stationary yaw baseline (24.09 vs 22.56
degrees). The existing arm-only network is also much more accurate for the arm
itself in this low-load setting (four-second 0.068 / 0.203 degrees), despite
not observing the cube. Successful object prediction did not improve arm prediction.

The dataset is dominated by non-contact samples despite targeted collection;
intermittent contacts, frictional stopping, yaw, and off-manifold drift remain
weaknesses. Exact mass/friction, constrained roll/pitch, and the same procedural
families are shared between training and evaluation. There is no evidence yet
for different physical properties, tipping, arbitrary collision geometry, or
cube-manipulation planning. Cube-targeting MPC was not implemented.

### Validation and artifacts

All 23 tests pass, including old arm-only/MPC tests and new tests for support,
inertia, physical pushing, frictional stopping, reproducible collection, valid
starts, episode splits, action holds, state extraction, yaw wrap/reconstruction,
no privileged contact inputs, target correctness, normalization, interval checks,
gradient consistency, and 200-step autoregression. Dependency checks pass.
Both arm-only checkpoint hashes remain unchanged.

Key outputs:

- `data/arm_cube_dynamics.npz`: shared training/validation/test experience.
- `data/arm_cube_unseen.npz`: independent evaluation-only experience.
- `runs/arm_cube_dynamics/`, `runs/arm_cube_dynamics_20ms/`: preserved first passes.
- `runs/arm_cube_dynamics_recursive/`: refined model and validation history.
- `runs/arm_cube_unseen_firstpass/`, `runs/arm_cube_unseen_20ms/`,
  `runs/arm_cube_unseen_recursive/`: fresh matched comparisons.

Evaluation outputs include `evaluation.json`, raw `test_rollouts.npz`,
`horizon_errors.png`, and best/worst moving-episode plots with actual/predicted
arm angles, cube position/velocity/yaw, XY paths, and actual contact timing over
errors. Best/worst are chosen by whole-trajectory XY error among episodes with
actual displacement >2 cm; they are not representative average examples.
Viewer overlays were not added; static plots and saved numerical trajectories
are the inspection tools for this experiment.
