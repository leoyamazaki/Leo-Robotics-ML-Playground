# Broader arm-dynamics training

The broader model substantially reduces five-second prediction errors on fresh
reversal and near-limit sequences. The original checkpoint and dataset remain
available for comparison.

## Training changes

Collected 40,000 MuJoCo transitions in 800 five-second episodes using seed 73.
The eight equally represented motion categories retain the original four and
add irregular full-range commands, fast reversals, limit holds, and limit
reversals. Command durations vary from 0.1 to 1.3 seconds by category; some
targets reach exact actuator limits. Added categories include near-limit initial
poses, and two thirds of episodes begin with nonzero angular velocities.

The resulting observed velocity range is approximately -12.10 to +12.26 rad/s
for shoulder and -20.63 to +20.53 rad/s for elbow. Position coverage approaches
both joint limits. The cube remains outside the workspace.

The architecture remains `6 -> 64 -> ReLU -> 64 -> ReLU -> 4`, trained from
scratch on CPU. Actions and state representation are unchanged. Training ran
200 epochs; epoch 200 had the best validation loss and was saved. Split counts
are 560 training / 120 validation / 120 test episodes, stratified by category.
Normalization uses training data only. The original run used fewer transitions
and epochs, so this is not an ablation isolating data diversity from training
volume or optimizer updates.

New checkpoint: `runs/arm_dynamics_broad/model.pt`.
SHA256: `2e71458992bafe881e95245576f30b6a7bc0733b965a3d33c3b4b05834fd9f4c`.
The original checkpoint hash remains
`2dd8ea900af3c43b5515ff6cee8f862f8a836cf88d3c31e9486502fd0fcc4f57`.

## Fresh five-second evaluation

Both checkpoints were evaluated on the same 48 newly generated episodes,
seed 2027, distinct from training and the previous stress audit. Actions,
ground-truth states, and group labels were confirmed exactly equal between
evaluations. Each rollout starts from one measured state and uses 50 predicted
states recursively, without corrections or clipping. Checkpoints were selected
using validation loss before these stress evaluations.

Five-second endpoint angle RMSE, in degrees, **shoulder / elbow**:

| Commands | Original model | Broader model |
| --- | --- | --- |
| Irregular moderate commands | 1.05 / 1.35 | 0.28 / 0.43 |
| Rapid moderate reversals | 12.42 / 5.80 | 0.50 / 0.26 |
| Near-limit holds | 7.76 / 11.83 | 0.36 / 0.28 |
| Near-limit reversals | 10.14 / 16.83 | 0.81 / 0.26 |

The new model's whole-trajectory angle RMSE for near-limit reversals is
1.29 / 0.46 degrees; peak individual errors are 5.65 / 2.17 degrees. Thus,
the small endpoint errors do not mean every intermediate prediction is equally
accurate. Immediately after large sign reversals, RMSE is 1.28 / 0.58 degrees.

For near-limit holds, predicted angles exceed nominal limits by more than
0.001 rad at 7 of 600 sample times. Maximum overshoot is 0.011 degrees shoulder
and 0.184 degrees elbow. Predictions were not clipped to hide these errors.
Ground-truth joint-limit constraints did not activate in the fresh stress audit;
this does not validate stop-impact dynamics or unrestricted MPC.

The broader dataset's own held-out one-step test gives 0.252 / 0.263 degrees
angle RMSE and 0.047 / 0.062 rad/s velocity RMSE. That test distribution differs
from the original dataset's, so use the identical fresh stress sequences above
for direct model comparisons.

## Artifacts and checks

- `data/arm_dynamics_broad.npz` and adjacent JSON: dataset and coverage summary.
- `runs/arm_dynamics_broad/`: checkpoint, training history, and standard evaluation.
- `runs/rollout_stress_original_fresh/`: original checkpoint's seed-2027 results.
- `runs/rollout_stress_broad_fresh/`: broader checkpoint's seed-2027 results.
- Both stress directories contain raw trajectories, metrics, and worst-case plots.
- All ten pytest tests pass; dependency consistency passes. New tests verify
  deterministic broad collection, category coverage across splits, moving starts,
  legal full-range actions, and command changes outside the old 0.5-second grid.

Reproduction commands are in the README's broader-training section. Future
training decisions informed by this audit should be checked on another reserved
seed rather than treating seed 2027 as a permanently untouched test set.
