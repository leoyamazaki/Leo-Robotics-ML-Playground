# Stationary-supervised Medium world-model experiment

## Recommendation

**Use `stationary_20/model.pt` for the next paired CEM-MPC test, retaining the original Medium model as its control.** The 20-step stationary model reduces mean maximum stationary/no-contact drift by 43.4% on the independent boundary set and 32.7% on the broad-motion set. Four-second actual-push endpoint RMSE changes from 12.19 to 12.36 cm on boundary (+1.4%) and from 14.33 to 11.94 cm on broad (-16.7%). Clear-push one-step error increases modestly (5.8% boundary, 3.4% broad). This is a better tradeoff than selecting by stationary accuracy alone.

The strict stationary goal is **not solved**: 98.75% of the 80 boundary no-contact cases and all 25 broad stationary cases still exceed 2 mm predicted movement. The recommendation is for another controlled MPC experiment, not a claim of successful closed-loop manipulation. Existing controller defaults and checkpoints are unchanged.

## Reproduce and inspect

From the project root in PowerShell:

```powershell
.venv/Scripts/python.exe -m simple_arm.stationary_experiment --output runs/arm_cube_stationary_repeat
# Configurable training-only stationary supervision
.venv/Scripts/python.exe -m simple_arm.stationary_experiment --output runs/arm_cube_stationary_custom --epochs 10 --stationary-threshold 0.002 --stationary-weight 1.0 --seed 94
# Replay the exact independent predictions; use the same --episode for comparisons
.venv/Scripts/python.exe -m simple_arm.contact_viewer --run runs/arm_cube_stationary_experiment/evaluations/boundary/baseline --episode 23
.venv/Scripts/python.exe -m simple_arm.contact_viewer --run runs/arm_cube_stationary_experiment/evaluations/boundary/stationary_20 --episode 23
# Suggested subsequent MPC inspection; not part of the training/evaluation run
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --scoring original --checkpoint runs/arm_cube_stationary_experiment/stationary_20/model.pt
```

Output directories must be new. Other flags: `--source`, `--batch-size` (256), and `--learning-rate` (0.0002). The source defaults to `runs/arm_cube_capacity_refinement/medium_256/model.pt`. The paired CEM experiment above is a suggested next step; no new simulation trajectories were generated for this study.

## Protocol and controls

Use the latest saved `data/arm_cube_contact_boundary/mixed.npz`: 2,480 four-second episodes, 99,200 transitions at 100 ms, with unchanged 1,736/372/372 train/validation/test episode splits and paired-group boundaries. All trainable conditions start directly from the same recursively refined Medium checkpoint (256 x 3 SiLU hidden layers, 138,764 parameters, previous selected refinement epoch 9). State order, absolute-position action conventions, stride 5, normalization tensors and architecture are preserved exactly. No normalization fitting or one-step pretraining is repeated.

Four conditions are evaluated:

1. `baseline`: the exact frozen source model; zero additional optimization.
2. `control_10`: additional 10-step training with the existing objective and zero stationary weight. This isolates the effect of simply training longer.
3. `stationary_10`: the same additional training plus stationary loss.
4. `stationary_20`: the same stationary loss with a 20-step (2 s) recursive horizon.

All three trainable conditions use 10 epochs, 36,456 windows per epoch, batch size 256, 143 optimizer updates per epoch / 1,430 total updates, identical window starts and batch order (dedicated seed 94), Adam 0.0002, gradient norm limit 5, and two CPU threads. They use all 21 starts 0..20 per training episode so every start supports a full 20-step rollout; the 10-step conditions use those same starts. This is a controlled common-start subset of the old 31-start recipe, not an exact rerun of it. Ten-step windows supervise transition indices 0..29, while twenty-step windows also supervise the later transitions through index 39; that added temporal exposure is part of the horizon intervention. No windows cross an episode or split.

Budgets match updates/windows/batch order, not FLOPs: 10-step variants process 3,645,600 supervised model-transition instances, while the 20-step variant processes 7,291,200. Its roughly doubled computation is reported rather than hidden. The source frozen model is supplemented by the matched original-objective continuation control to avoid attributing all extra-training effects to the stationary term.

Each checkpoint is selected using the original unweighted, state-scaled 40-step validation rollout MSE, including epoch zero. The stationary metric does not select a checkpoint on its own. All three selections finish before any independent evaluation. `last.pt` preserves final-epoch weights for inspection, but only validation-selected `model.pt` files are compared here.

## Training-only supervision

At each 100 ms interval t, define stationary(t) when **all five saved 20 ms XY samples** remain within the configurable threshold of the actual cube position at t. Default threshold: 2 mm. This catches out-and-back motion that an endpoint-only test would miss. Labels are local to each interval: a cube can move later or stop at a new position. Training has 61,540 stationary labels among 69,440 intervals; 929 training episodes change stationary/moving label within the episode.

The added term is:

```text
stationary_weight * mean_over_windows_and_steps(
    stationary(t) * mean_XY(((predicted_cube_XY(t+1) - actual_cube_XY(t)) / 0.1 m)^2)
)
```

Defaults: additional weight 1.0, threshold 0.002 m. Predicted states feed back recursively with gradients through all model steps. Each supervised step anchors the penalty to the actual beginning of its own interval, not the episode origin, the window origin, or the already-drifting prediction. Moving intervals contribute zero stationary penalty. Near-threshold true motion can be suppressed, so contact accuracy is explicitly reported.

The original contact-weighted state loss stays intact: mean squared scaled 13-state error, with 3x weight on steps that contain saved contact. Scales are `[1,1,3,3,0.1,0.1,0.01,0.5,0.5,0.5,0.5,0.1,2]`. The stationary term uses ground-truth motion only and introduces no new contact detector. Neither stationary labels nor contact labels enter inference. Existing `load_predictor` / tensor transitions are reused with no movement clamps or hand-coded inference rules.

## Training and validation

Common losses below are unweighted full 40-step state-scaled MSE, measured at each selected epoch on the full training/validation splits. The online objective is contact-weighted and uses 10 or 20 steps, so it is not numerically interchangeable with these common diagnostics. Per-epoch online total/base/stationary loss and common 40-step train/validation components are retained in `training.json`.

| Model | Selected added epoch | Common train loss | Common validation loss | Additional training time |
|---|---:|---:|---:|---:|
| baseline | 0 | 0.04145 | 0.06221 | 0 s |
| control_10 | 4 | 0.03580 | 0.05801 | 74.0 s |
| stationary_10 | 3 | 0.03583 | 0.05359 | 74.5 s |
| stationary_20 | 6 | 0.03093 | 0.05424 | 157.8 s |

Training timing starts after model/optimizer/tensor setup, and includes epoch-zero diagnostics, optimizer steps, full train/validation diagnostics, and checkpoint writes. Total additional training time is about 306 seconds. A brief numerical test run overlapped part of the control; these are single-run observed times, not repeated isolated throughput measurements.

## Independent evaluation definitions

All four models use the same complete saved independent boundary set (seed 292, 160 episodes) and broad-motion set (seed 293, 120 episodes), with no overlap with training collection seeds 191/291. Actual states and command arrays saved in every evaluation were checked exactly equal across models.

Cube errors are XY vector RMSE. One-step predictions are teacher-forced: every transition starts at its actual recorded state. Boundary episode cases use saved collector categories: 80 no-contact (40 overhead + 40 near-miss), 40 light-contact, 40 clear-push. Broad categories use the same physical episode rules as boundary collection: no saved contact, light <=0.2 s contact with <=2 cm excursion, clear >=0.1 s contact with >=5 cm excursion. Broad has 25 no-contact, zero light-contact, and 91 clear-push cases; four additional moderate pushes are included in the wider actual-push endpoint metric. All transitions within each case are included, even non-contact portions of a push episode. Supplementary interval-conditioned metrics are saved for truly local contact comparisons.

A stationary episode must stay within 2 mm of its initial XY position at every saved 20 ms sample for all four seconds. This evaluation definition always uses 2 mm, independently of the training threshold flag. False drift is the maximum predicted XY distance from that initial position over the 40-step recursive path. Actual pushes require saved contact and final actual displacement >2 cm: 40 boundary and 95 broad episodes.

## One-step cube displacement error

| Dataset / model | No-contact mm | Light-contact mm | Clear-push mm |
|---|---:|---:|---:|
| boundary / baseline | 1.57 | 2.19 | 4.78 |
| boundary / control_10 | 1.55 | 2.14 | 4.65 |
| boundary / stationary_10 | 1.23 | 1.72 | 4.91 |
| boundary / stationary_20 | 1.25 | 1.74 | 5.05 |
| broad / baseline | 4.20 | n/a | 6.01 |
| broad / control_10 | 3.92 | n/a | 5.86 |
| broad / stationary_10 | 3.37 | n/a | 6.07 |
| broad / stationary_20 | 3.42 | n/a | 6.22 |

Broad light-contact episode error is n/a because no such episodes meet the saved-data category definition; it is not treated as zero. Per-interval light-contact metrics are available in JSON for both datasets. The 20-step model improves boundary no/light-contact error by approximately 21%/20%, while clear-push one-step error gets 5.8% worse. Broad no-contact improves 18.5%, with clear-push one-step error 3.4% worse.

## Four-second drift and push tradeoffs

| Dataset / model | Mean max stationary/no-contact drift mm | Worst drift mm | Stationary/no-contact >2mm | Actual-push 4s endpoint RMSE cm | Clear-push 4s endpoint RMSE cm |
|---|---:|---:|---:|---:|---:|
| boundary / baseline | 24.64 | 107.17 | 98.75% | 12.19 | 12.19 |
| boundary / control_10 | 19.91 | 80.21 | 100.00% | 12.69 | 12.69 |
| boundary / stationary_10 | 13.96 | 65.36 | 98.75% | 11.77 | 11.77 |
| boundary / stationary_20 | 13.95 | 53.25 | 98.75% | 12.36 | 12.36 |
| broad / baseline | 21.05 | 47.44 | 100.00% | 14.33 | 14.20 |
| broad / control_10 | 21.26 | 44.78 | 100.00% | 12.29 | 12.13 |
| broad / stationary_10 | 15.04 | 36.92 | 100.00% | 15.42 | 15.30 |
| broad / stationary_20 | 14.18 | 45.28 | 100.00% | 11.94 | 11.78 |

Near-miss and overhead details use all 40 stationary cases in each class:

| Model | Overhead mean / P95 / worst max drift mm | Overhead >2mm | Near-miss mean / P95 / worst max drift mm | Near-miss >2mm |
|---|---|---:|---|---:|
| baseline | 17.06 / 25.51 / 32.21 | 97.5% | 32.21 / 59.36 / 107.17 | 100.0% |
| control_10 | 12.23 / 21.73 / 42.76 | 100.0% | 27.59 / 44.60 / 80.21 | 100.0% |
| stationary_10 | 6.33 / 16.53 / 24.97 | 97.5% | 21.60 / 39.63 / 65.36 | 100.0% |
| stationary_20 | 7.74 / 15.71 / 19.76 | 97.5% | 20.16 / 37.14 / 53.25 | 100.0% |

Including stationary light-contact cases, the boundary stationary cohort has 111 episodes. Its >2 mm rate is 99.10% baseline, 100% control, 99.10% stationary-10 and 99.10% stationary-20. The broad stationary cohort consists of the same 25 no-contact episodes and remains at 100%. Reducing drift magnitude has not improved the strict binary failure rate.

Relative to the original-objective continuation control, stationary-20 reduces mean no-contact drift by 29.9% boundary and 33.3% broad; actual-push endpoint error improves about 2.6%/2.9%. This shows a useful tradeoff beyond extra training alone, but also shows that much of the broad pushing gain versus frozen baseline already occurs with ordinary continuation.

Stationary-10 reduces drift strongly and improves boundary push endpoint error, but broad push error rises to 15.42 cm (7.6% worse than frozen baseline and 25.4% worse than the continued-training control). Longer rollouts add almost no further mean boundary drift reduction versus stationary-10, but improve broad drift by 5.7% and broad push endpoint error by 22.6%. Their main benefit here is the more balanced long-horizon pushing behavior, not elimination of stationary drift.

## Teacher-forced versus autoregressive

Teacher-forced (TF) error below measures instantaneous next-state predictions with true states supplied every 100 ms. AR final error measures a four-second rollout from the initial state only. These different horizons must not be interpreted as directly interchangeable errors. The integrated-TF diagnostic sums the predicted local XY increments from truth; it accumulates local bias while removing predicted-state feedback, and is explicitly not a physically consistent model rollout.

| Dataset / model, no-contact cases | TF one-step RMSE mm | Integrated-TF 4s diagnostic RMSE mm | AR 4s endpoint RMSE mm |
|---|---:|---:|---:|
| boundary / baseline | 1.57 | 20.76 | 28.21 |
| boundary / control_10 | 1.55 | 21.51 | 22.89 |
| boundary / stationary_10 | 1.23 | 14.83 | 17.18 |
| boundary / stationary_20 | 1.25 | 16.50 | 15.72 |
| broad / baseline | 4.20 | 31.10 | 22.06 |
| broad / control_10 | 3.92 | 30.89 | 22.69 |
| broad / stationary_10 | 3.37 | 23.73 | 15.41 |
| broad / stationary_20 | 3.42 | 22.18 | 14.09 |

Both instantaneous errors and accumulated drift are present. The stationary terms improve teacher-forced no-contact predictions, so the result is not solely an AR stabilization artifact. Nevertheless, small local biases accumulate: stationary-20 overhead mean maximum next-state error is about 1.25 mm with teacher forcing, versus 7.74 mm mean maximum excursion in AR mode. Near-miss teacher-forced maximum excursion remains 6.20 mm on average and up to 20.61 mm. One broad stationary example still has a 99.54 mm teacher-forced next-state error (123.36 mm baseline), despite only 45.28 mm worst AR drift across that cohort. Predicted-state feedback can change the arm/cube path enough to hide some instantaneous failures; lower AR drift does not prove correct instantaneous contact physics.

Saved metrics include all-category TF/AR position and incremental errors, and all individual prediction arrays. The two `<dataset>_teacher_forced_vs_ar.png` plots compare time-dependent next-state TF error with AR position error; the drift/push scatter plots show the two objectives together. These diagnostics support improved stationarity sensitivity, not a learned explicit contact classifier or a guarantee against model exploitation.

## Artifacts and verification

- `simple_arm/stationary_training.py`: configurable training-only interval masks, actual-anchor stationary loss, recursive training, matched budgets and full diagnostic histories.
- `simple_arm/stationary_experiment.py`: provenance checks, frozen/continued/stationary comparisons, independent case/interval metrics, TF/AR arrays and plots.
- `runs/arm_cube_stationary_experiment/`: config and source hashes; three selected/final checkpoint pairs and training histories; eight independent dataset/model evaluations; summary and five plots.
- All 29 pre-existing dataset/checkpoint/XML hashes remain unchanged. All source normalization tensors are exactly preserved; all four selected models load through the existing width-aware predictor and MPC adapter as 256-wide models.
- Fourteen relevant numerical/model tests passed: six new stationary tests, six existing capacity/refinement tests, and two existing yaw/recursive-rollout tests. No MuJoCo trajectories were collected, including in tests. The full test suite was intentionally not run because its collection fixtures generate simulation trajectories.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_stationary_training.py tests/test_capacity_screen.py -q --basetemp .pytest_cache/stationary_training_verify1
.venv/Scripts/python.exe -m pytest tests/test_contact_learning.py::test_yaw_targets_wrap_and_residual_reconstruction tests/test_contact_learning.py::test_contact_rollout_is_autoregressive -q
```

This is one lightweight training seed, selected by validation and evaluated on existing independent collection seeds. These independent sets have been inspected in earlier experiments; the results are a bounded screening result, not a fresh confirmatory holdout. Recommended next step: compare the original and stationary-20 checkpoints with identical CEM scores/seeds, exposing both predicted and actual cube motion. Do not combine a model change with a score change in that comparison.
