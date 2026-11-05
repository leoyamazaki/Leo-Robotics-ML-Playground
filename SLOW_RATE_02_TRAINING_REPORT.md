# Slow-motion 0.2 rad/s world-model fine tuning

## Outcome and recommendation

Collected a dedicated slow-contact corpus and fine-tuned the stationary-20 Medium model. The selected checkpoint contains updated weights trained on 0.2 rad/s command sequences, with older-data replay. **Keep the previous stationary-20 checkpoint for actual CEM control for now.** The new model improves stationary prediction, but worsens slow and broad-push four-second prediction and reduces paired five-second CEM success from 9/12 to 6/12. Both controllers stay in reach at the same 0.2 rad/s command limit.

The user-requested slow-trained checkpoint is preserved at `runs/arm_cube_slow_02_expanded/refined/model.pt`. It is an experimental candidate, not a replacement justified by these control results.

## Use and reproduce

Recommended current slow viewer configuration:

```powershell
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --checkpoint runs/arm_cube_stationary_experiment/stationary_20/model.pt --command-rate-limit 0.2
```

Inspect the newly trained model with identical controller settings:

```powershell
.venv/Scripts/python.exe -m simple_arm.cube_mpc view --checkpoint runs/arm_cube_slow_02_expanded/refined/model.pt --command-rate-limit 0.2
```

Training does not remove the need for the runtime command-rate flag. The model predicts physics; CEM chooses commands. The 0.2 value bounds target slew, not actual joint speed. Physics and actuators remain unchanged.

Reproduce in a new directory:

```powershell
.venv/Scripts/python.exe -m simple_arm.slow_training --rate 0.2 --groups 320 --independent-groups 40 --workers 4 --epochs 10 --output runs/arm_cube_slow_02_repeat
.venv/Scripts/python.exe -m simple_arm.slow_control_evaluation --candidate runs/arm_cube_slow_02_repeat/refined/model.pt --rate 0.2 --output runs/arm_cube_slow_02_control_repeat
```

All output/data paths must be new. Flags also include `--source` for another compatible Medium checkpoint. The collection rate is configurable. The dataset remains in each new experiment directory, leaving all existing training datasets intact.

## Slow data collection

Generate four-second trajectories with the original simulation, state order and 100 ms position-target actions. Start every arm and cube at rest. Commands ramp independently at 65-100% of the 0.2 rad/s cap, include holds and occasional smooth reversals, and reduce reference speed near the end. Desired reference acceleration is 0.2 rad/s?; joint-command box clipping can produce sharper deceleration. Every change from the initial command and every subsequent 100 ms target change is <=0.02 rad.

Each independently seeded arm configuration yields a paired group of four cube placements: stationary overhead, stationary near-miss, a push, and a grazing/push-release case. New collection uses geometry to position training examples and saved MuJoCo contact diagnostics to check coverage. Neither the geometry placement rules nor contact labels enter neural inference. Pushing examples require actual movement; stationary examples require no actual contact and <=2 mm motion. Related placements share an episode-group split.

Expanded collection: **1280 episodes / 51,200 transitions**, [896, 192, 192] train/validation/test episodes, seed 394. Separate independent slow evaluation: 160 episodes, seed 395; none are used for training or selection. Each starts at rest; half the episodes remain stationary. Collection took 82.5 s for training and 14.8 s for independent evaluation with four processes.

Maximum collected command rate is 0.199949 rad/s. Maximum actual joint speed observed in saved 20 ms states is 0.264 rad/s; maximum cube speed is 0.186 m/s. The independent slow set peaks at 0.259 rad/s actual joint speed. No state or velocity was clipped to make these measurements appear slow.

Coverage requirements do not impose a large joint-excursion threshold: slow movements, resting cubes and gentle pushes are deliberately retained. All accepted trajectories remain inside the original actual joint limits. Saved collection JSON includes per-group sampling/rejection counts and the measured command cap.

## Fine tuning and leakage controls

Initialize from `runs/arm_cube_stationary_experiment/stationary_20/model.pt` (256 x 3 hidden layers). Preserve all normalization tensors and the 13-state/15-input/12-output architecture. The new `initialization.pt` contains exactly those source weights with explicit source file hash and source metadata; its prospective corpus metadata records the forthcoming fine-tuning corpus. Newly trained checkpoints clear the initialization-only flag.

Training and validation each mix 50% slow episodes with 50% replay sampled without replacement from the corresponding original mixed-dataset split. Training: 896 new slow + 896 original train episodes. Validation: 192 new slow + 192 original validation episodes. The 192 new slow test episodes are retained unused; independent seed-395 data supplies the reported slow test. No original test episode enters fine tuning. Original and new group IDs are namespaced, and saved merged group IDs were checked never to cross splits.

Use the existing 20-step recursive loss and stationary supervision: actual-anchor, per-interval stationary masks (2 mm), stationary weight 1, existing state scales/contact weights, Adam 0.0001, gradient norm limit 5, batch size 256, two CPU threads, seeded window shuffle 95. Normalization is not refitted. Windows start at 0..20 with no episode boundary crossing. Select by the original unweighted full 40-step state-scaled validation loss, including epoch zero. Independent predictions and control outcomes are not checkpoint selectors.

The expanded run performs 1,470 optimizer updates across 10 epochs and selects epoch 10. Training took 100.8 s including common train/validation diagnostics and checkpoint writes. Full common validation state MSE improves from 0.03243 to 0.03146. Common training state MSE finishes at 0.01355. Full online/base/stationary and common train/validation histories are in `refined/training.json`.

Two small pilots were retained in `runs/arm_cube_slow_02/`: 320 new slow episodes, with 10-epoch fine tuning at 0.0001 and 0.00001. Both overfit and selected epoch zero. The expanded corpus was tried in response to validation results; it starts again from the original stationary-20 source, not pilot final weights. The expanded seed-394 corpus shares deterministic physical groups with the pilot, but the final source model never trained on pilot weights. Independent seed 395 never enters training or selection. The pilots are not portrayed as independent confirmation of the expanded result.

## Independent prediction results

Compare the previous stationary-20 baseline and selected slow-refined model on identical saved commands and truth. All values below are XY vector errors; drift is mean maximum predicted movement over four seconds on truly stationary/no-contact episodes.

| Dataset / model | Overall one-step RMSE mm | Mean max false drift mm | Stationary/no-contact >2 mm | Actual-push 4 s endpoint RMSE cm |
|---|---:|---:|---:|---:|
| slow_independent / baseline | 1.29 | 10.64 | 98.75% | 5.12 |
| slow_independent / slow_refined | 1.17 | 6.45 | 87.50% | 5.79 |
| boundary / baseline | 2.81 | 13.95 | 98.75% | 12.36 |
| boundary / slow_refined | 2.78 | 10.47 | 95.00% | 12.10 |
| broad / baseline | 6.00 | 14.18 | 100.00% | 11.94 |
| broad / slow_refined | 5.97 | 13.12 | 100.00% | 13.48 |

Slow held-out stationary drift improves 39.4%, and the >2 mm rate falls from 98.75% to 87.5%. Boundary drift improves 25.0%; broad drift improves 7.5%. However slow-push endpoint error worsens 13.0%, broad-push error worsens 12.9%, and boundary-push error improves only 2.1%. Better stationarity does not establish better pushing.

| Dataset / model | No-contact one-step mm | Light-contact one-step mm | Clear-push one-step mm |
|---|---:|---:|---:|
| slow_independent / baseline | 0.55 | n/a | 1.86 |
| slow_independent / slow_refined | 0.40 | n/a | 1.73 |
| boundary / baseline | 1.25 | 1.74 | 5.05 |
| boundary / slow_refined | 1.21 | 1.75 | 5.00 |
| broad / baseline | 3.42 | n/a | 6.22 |
| broad / slow_refined | 3.04 | n/a | 6.23 |

Boundary categories use saved collector labels. Slow/broad cases use actual saved-contact duration and cube excursion thresholds, so light-contact episode cohorts are empty there (n/a, not zero error). Slow independent data has 80 no-contact cases, 62 clear pushes by the stricter >=5 cm category, and 80 actual pushes by >2 cm final movement. JSON also reports local interval-conditioned metrics.

Teacher-forced predictions improve on the slow set (1.29 ->1.17 mm overall) while four-second autoregressive endpoint error worsens (37.28 ->41.24 mm over all slow episodes). Errors are accumulating despite better local predictions. The integrated teacher-forced displacement diagnostic improves from 32.70 to 27.23 mm, reinforcing that full predicted-state feedback remains a problem. Saved arrays include teacher-forced next states, autoregressive predictions and accumulated-local-error diagnostics; plots compare recursive RMSE over 0.1..4 s.

## Actual CEM-MPC comparison at 0.2 rad/s

Use the same original score, horizon 15, 256 candidates, five CEM iterations, original knot interpolation, rate projection, target scenarios (23,99,139,159), and seeds 42/43/44. Execute five simulation seconds per run, with identical initial states in each pair. Only the world-model checkpoint changes. Success requires staying within 2 cm for the final 0.5 seconds, measured at 2 ms.

| Metric | Previous stationary-20 | Slow-refined |
|---|---:|---:|
| Success rate | 75.00% | 50.00% |
| Mean final target distance | 1.78 cm | 2.08 cm |
| Escape run rate | 0.00% | 0.00% |
| Maximum actual cube speed | 0.20 m/s | 0.20 m/s |

| Case / seed | Final cm old / new | Success old / new |
|---|---:|---|
| 23/42 | 5.62 / 5.97 | False / False |
| 23/43 | 5.78 / 5.95 | False / False |
| 23/44 | 5.68 / 5.99 | False / False |
| 99/42 | 0.26 / 0.19 | True / True |
| 99/43 | 0.57 / 0.09 | True / True |
| 99/44 | 0.32 / 0.42 | True / True |
| 139/42 | 0.11 / 0.23 | True / True |
| 139/43 | 0.40 / 0.32 | True / True |
| 139/44 | 0.40 / 0.26 | True / True |
| 159/42 | 0.54 / 1.67 | True / False |
| 159/43 | 0.83 / 1.78 | True / False |
| 159/44 | 0.84 / 2.12 | True / False |

Both models fail all episode-23 trials within five seconds, while both succeed on episodes 99 and 139. The slow-trained model loses the three episode-159 successes. Two of those new runs finish within 2 cm but fail the final-half-second residence criterion; final position alone is not success. Neither model has workspace escapes. Peak actual joint speed is about 0.277 rad/s in both controller groups, so the 0.2 command constraint is still not a hard physical speed bound.

These outcomes support the runtime limit as a way to obtain slow motion, but do not support replacing the previous model with this fine-tuned candidate. Further training should address accumulated pushing error and settling, while maintaining these stationary gains.

## Artifacts and checks

- `simple_arm/slow_collection.py`: deterministic paired collection, smooth/rate-bounded references, grouped splits and physical-motion measurements.
- `simple_arm/slow_training.py`: balanced old/new replay, frozen normalization, fine tuning, validation selection, independent predictions, and source audits.
- `simple_arm/slow_control_evaluation.py`: paired actual control comparison with both models at 0.2 rad/s.
- `runs/arm_cube_slow_02_expanded/`: raw slow training/independent data, merged replay data and provenance, initialization, selected/final checkpoints, histories, six model/dataset evaluations and recursive-error plots.
- `runs/arm_cube_slow_02_control/`: 24 actual control traces, metrics, planning diagnostics, summary and actual-distance plot. The model is frozen throughout control evaluation.
- All 40 source files audited before the expanded experiment (prior datasets/checkpoints/XML) retain their SHA-256 hashes. The original 35 source files remain unchanged. Normalization tensors match exactly and the selected model loads through the existing 256-wide MPC adapter. New weights were verified to differ from the original source.
- Thirty-nine relevant tests passed, covering slow reference bounds from the initial command, determinism, real collected contacts/stationary starts, replay/split/group isolation, overwrite guards, stationary training, all-candidate rate projection, and existing controller/workspace behavior.
- The viewer with the new checkpoint, 0.2 limit and episode 99 passed a three-second launch/close smoke test. It logged deadline overruns under concurrent evaluation; simulation updates still span exactly 100 ms.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_slow_training.py tests/test_stationary_training.py tests/test_cube_command_rate.py tests/test_cube_mpc.py tests/test_cube_workspace.py -q --basetemp .pytest_cache/slow_final_verify1
```

This is a lightweight single-seed fine-tuning study with mixed outcomes, not evidence that learning slower examples alone makes the planner prefer slow commands. Runtime command-rate enforcement remains necessary.
