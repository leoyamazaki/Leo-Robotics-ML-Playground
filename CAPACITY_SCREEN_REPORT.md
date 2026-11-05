# Arm–cube network capacity screen

## Protocol

This is a short one-step capacity screen, not full retraining or recursive
training. No simulation trajectories were collected or regenerated.

The existing `data/arm_cube_contact_boundary/mixed.npz` contains 99,200 transitions
at 100 ms across 2,480 episodes. Its exact original episode/group splits were
preserved: 1,736/372/372 episodes, or 69,440/14,880/14,880 transitions for
training/validation/test. All models were evaluated on the same 372 test episodes,
including 48 clear-push, 48 overhead and 48 pusher-near-miss boundary episodes.
There are 145 test episodes with no arm contact and <=2 mm maximum cube excursion;
these form the general no-contact drift metric.

All newly trained models have 15 inputs, three SiLU hidden layers, 12 residual
outputs, the same geometry-relative state features and sin/cos yaw composition.
Only hidden width changes. Input and residual normalization are reused directly
from the existing one-step checkpoint, whose exact dataset provenance and
training-only means are checked before training. Contact labels remain loss
weights/evaluation diagnostics, never prediction inputs.

- Widths 128/256/512: **36,620 / 138,764 / 539,660 parameters**.
- CPU, two PyTorch threads, float32, batch size 256, Adam 0.001 with unchanged
  default optimizer settings. The objective is the existing normalized residual
  MSE with 5x weight for transitions containing contact.
- **12 epochs and 3,264 optimizer updates for every new model**, with no early
  stopping. A separate seeded shuffle generator gives all widths identical batch
  orders; initialization and shuffling both use seed 91. Model initialization
  consumes different numbers of random draws at different widths, which is why
  the batch-order generator is separate.
- Select the checkpoint by minimum unweighted one-step validation loss within
  the cap. All three selected epoch **11**. Test trajectories never select epochs.
- Equal optimization budgets mean equal samples, epochs and updates. They do
  not mean equal FLOPs or wall time; larger models cost more per update.

The 128-wide short run is an additional matched-budget control. Existing
128-wide references are loaded frozen: the historical one-step model trained
150 epochs (selected 136), and its recursively refined model trained a further
80 epochs (selected refinement 25). Their methodology and optimization budgets
differ substantially from this screen.

## Training cost and generalization

Losses below are post-training-checkpoint normalized residual MSE, evaluated
with the same unweighted reduction on train and validation. The actual training
objective retains contact weighting. Online and post-epoch weighted losses are
also saved in the training histories and summary.

| Model | Train loss | Validation loss | Weighted train loss | Training wall time | Mean epoch |
|---|---:|---:|---:|---:|---:|
| Short 128 control | 0.33091 | 0.29889 | 1.28534 | 9.70 s | 0.81 s |
| Medium 256 | 0.26400 | 0.26479 | 1.01727 | 16.50 s | 1.38 s |
| Large 512 | 0.21236 | 0.24093 | 0.79680 | 45.38 s | 3.78 s |
| Existing 128 one-step | 0.07824 | 0.21050 | 0.27968 | Not recorded | Not recorded |
| Existing 128 refined | 0.06914 | 0.21362 | 0.26344 | Not recorded | Not recorded |

New-model wall times cover the capped epoch loops, including post-epoch
train/validation loss passes and best-checkpoint writes. They exclude common
data loading/normalization, model setup, plots and test evaluation. These are
single runs, not repeated isolated timing benchmarks. Historical timings were
not recorded and were not reconstructed from file timestamps or retraining.

Medium lowers validation loss by **11.4%** relative to the matched short control;
large lowers it by **19.4%**. Large improves on medium by 9.0%, with a larger
train/validation gap. Both remain above the much longer-trained existing
one-step checkpoint. This short screen does not show converged performance or
establish whether larger models will generalize better after full training.

## Identical held-out trajectories

Cube XY errors are vector RMSE in the horizontal plane. The 1/2/4 s columns
measure errors exactly at those horizons, not error averaged over a prefix.
Every rollout starts from the observed initial state, then uses only its
previous predicted states and the recorded future commands. Saved actual
states and commands were checked identical across all five evaluations.

| Model / training method | One-step XY | 1 s endpoint | 2 s endpoint | 4 s endpoint |
|---|---:|---:|---:|---:|
| Short 128, one-step only | 5.00 mm | 7.46 cm | 11.86 cm | 17.89 cm |
| Medium 256, one-step only | 4.49 mm | 4.77 cm | 9.18 cm | 13.05 cm |
| Large 512, one-step only | 4.17 mm | 3.87 cm | 7.75 cm | 12.33 cm |
| Existing 128, long one-step | **3.45 mm** | **2.77 cm** | **5.73 cm** | **8.79 cm** |
| Existing 128, recursively refined | **3.33 mm** | **2.65 cm** | **5.76 cm** | **9.25 cm** |

Medium and large reduce four-second error by **27.1% and 31.1%** relative to
the matched control. Large improves on medium by only 5.6% for this aggregate
metric, despite taking 2.75x as much training time. The references still perform
better. That comparison mixes training duration and method with width; it is
not evidence that higher capacity hurts after full training.

The stationary cube baseline has 14.09 cm four-second error and constant velocity
23.97 cm on these same episodes. Both larger short models beat stationary
prediction in aggregate, while the short 128 control does not.

### No contact and genuine pushes

Maximum drift uses the whole four-second path relative to the initial cube
position. No-contact classification uses ground-truth contacts solely for
evaluation and requires actual cube excursion <=2 mm across saved states.

| Model | No-contact mean max drift | 95th percentile | Clear-push 4 s error | Broad-push 4 s error |
|---|---:|---:|---:|---:|
| Short 128 | 132.4 mm | 306.5 mm | 18.57 cm | 19.00 cm |
| Medium 256 | 78.3 mm | 153.7 mm | 14.56 cm | 16.83 cm |
| Large 512 | 64.5 mm | 128.3 mm | 11.29 cm | 17.63 cm |
| Existing 128 one-step | 22.6 mm | 53.4 mm | **7.03 cm** | **13.90 cm** |
| Existing 128 refined | **13.8 mm** | **31.1 mm** | **8.44 cm** | **14.72 cm** |

Clear pushes are the 48 original test boundary examples, not newly collected
episodes. Broad pushes are the broad-source test episodes with actual endpoint
displacement >2 cm. Large improves clear-push error by **22.5%** relative to
medium, and mean displacement-direction cosine is 0.788 versus 0.514. However,
large is slightly worse on broad pushes. Neither width dominates every regime.

All three short models exceed 2 mm predicted motion on every no-contact episode.
The existing refined model does so on 98.6%. Lower drift magnitudes do not mean
that any model reliably predicts a stationary cube. In the focused overhead /
near-miss subsets, mean max drift is 72.5/95.8 mm for medium, 52.9/67.5 mm for
large, and 7.7/22.1 mm for the refined reference. Recursive stabilization and/or
more one-step optimization remain important.

## Recommendation

**Medium is the practical first candidate for full one-step training followed by
recursive refinement on this laptop.** It captures about 87% of large's aggregate
four-second error reduction over the matched control, for about 36% of large's
training time. It also improves validation loss and broad-push performance over
the short control, so the capacity signal is not confined to training loss.

**Large also warrants follow-up when clear-push accuracy is the priority.** It
has the best short-budget validation loss, one-step error, aggregate rollout error
and clear-push predictions. Its extra cost is not justified by a demonstrated
win over the existing model yet. Give the selected candidate a fuller one-step
fit before recursive refinement; this screen does not prove how either width
will behave under the recursive objective.

No existing model is replaced or promoted by this screen. No recursive training
was run. Results use one initialization/shuffle seed and one fixed test split;
they justify follow-up experiments, not a claim that capacity is the sole
remaining bottleneck. Repeated width selection on this test split should be
followed by a fresh independent evaluation before broader claims.

## Artifacts and reproduction

All new outputs are under `runs/arm_cube_capacity_screen/`:

- `config.json`, `summary.json`: exact budget, provenance, losses, timings and
  aggregated physical metrics, with training methods labeled.
- `short_128/`, `medium_256/`, `large_512/`: new `model.pt`, epoch-by-epoch
  `training.json` and train/validation `loss.png`.
- `evaluations/<model>/`: full state/contact/category metrics, saved actual and
  autoregressive trajectories, horizon plots and representative push plots.
- `endpoint_comparison.png`, `drift_comparison.png`: common held-out comparisons.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.capacity_screen --epochs 12 --output runs/arm_cube_capacity_screen_repeat
.\.venv\Scripts\python.exe -m pytest tests/test_capacity_screen.py tests/test_contact_learning.py::test_contact_rollout_is_autoregressive -q
```

The output directory must be new. Dataset and baseline checkpoint paths are
recorded in the configuration. Width-aware checkpoints remain compatible with
the existing predictor, saved replay viewer and future recursive-training
function. Older checkpoints without a width field still default to 128.

Six neural-network tests passed, covering parameter counts, capped updates,
normalization/provenance, width-aware and legacy loading, differentiable
prediction consistency and autoregressive state feedback. Tests use numerical
fixtures; collection tests were not run for this benchmark. All pre-existing
datasets and checkpoints were checked unchanged by hash.
