# Recursive refinement of medium and large world models

## Follow-up protocol

Refine both saved capacity-screen models directly: medium (256 x 3, 138,764
parameters) and large (512 x 3, 539,660 parameters). Both source checkpoints were
selected at epoch 11 of the same 12-epoch one-step screen. No additional
one-step training, collection, trajectory regeneration or normalization fitting
is performed. The original 99,200-transition dataset and its original
train/validation/test episode/group splits remain unchanged.

This is another bounded screening run: **10 recursive epochs per model**,
**211 optimizer updates per epoch / 2,110 updates total**. Both models use exactly
the same batch-window order, via a dedicated shuffle generator seeded 93, batch
size 256 and two CPU threads. Increased width costs more per update; equal update
counts are not equal FLOP or wall-time budgets.

The objective is the existing contact model's recursive recipe:

- Start each window from its recorded state; recursively predict ten successive
  100 ms transitions using only the previous prediction and the recorded action.
- Optimize state-scaled MSE with 3x weight for steps containing actual contact,
  Adam 0.0002 and gradient norm limit 5. State scales remain
  `[1,1,3,3,0.1,0.1,0.01,0.5,0.5,0.5,0.5,0.1,2]` for the existing 13-state order.
- Use all 31 one-second windows from each of 1,736 training episodes. No window
  crosses an episode or dataset split. Contact is a supervision weight only.
- Select by unweighted, state-scaled full four-second autoregressive validation
  loss. Evaluate every epoch, including epoch zero. A dedicated shuffle stream
  and per-epoch validation make this follow-up fair between widths; historical
  refinement used the global seeded stream and validated every five epochs.

Epoch zero is eligible: a run that never improves validation retains its source
weights. Test data never select a checkpoint. Prediction states are not clipped,
and neither validation nor test rollouts receive later simulator corrections.

Evaluate both selected checkpoints against their saved one-step versions on the
same 372 held-out episodes used by the capacity screen. Saved actual states and
commands must match exactly. Report 1/2/4-second endpoint errors separately from
prefix averages, one-step errors, no-contact drift, clear pushes and broad pushes.
The existing 128-wide refined reference remains a historical comparison with
much longer one-step and recursive training, not a matched-width control.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python.exe -m simple_arm.capacity_refinement --epochs 10 --output runs/arm_cube_capacity_refinement_repeat
```

The output directory must be new. The source checkpoints and saved screen
evaluations are read from `runs/arm_cube_capacity_screen/`. Outputs are under
`runs/arm_cube_capacity_refinement/` for the completed run:

- `config.json`: source checkpoint and dataset hashes, budget and settings.
- `medium_256/`, `large_512/`: refined checkpoints, validation histories,
  `refinement_training.json` timing/update records, and validation-loss plots.
- `evaluations/`: full physical/category metrics, saved predictions and actual
  trajectories, and representative plots for each selected model.
- `summary.json`, `endpoint_comparison.png`: before/after/historical comparisons.

The timing starts after model/data/optimizer setup and includes initial
validation, optimizer epochs, per-epoch validation and best-checkpoint writes.
Mean epoch time excludes the initial epoch-zero validation. File hashes protect
the original dataset, capacity-screen checkpoints and historical checkpoints.

Seven neural-network tests passed, including both width-aware recursive updates,
equal update caps, normalization preservation, unchanged source files, legacy
loading and autoregressive predicted-state feedback. Tests use numerical
fixtures and do not collect simulation trajectories.

## Results

Both models ran the complete **10-epoch / 2,110-update** budget, selecting
**refinement epoch 9** by validation. Medium's full validation rollout loss fell
from 0.13999 to 0.06221; large's from 0.09156 to 0.06123. These are state-scaled
recursive losses, not the normalized one-step loss used in the previous report.

| Model | Additional refinement wall time | Mean epoch time | Best epoch |
|---|---:|---:|---:|
| Medium 256 | 74.3 s | 7.42 s | 9 |
| Large 512 | 206.4 s | 20.63 s | 9 |

Large took **2.78x** medium's refinement time with equal numbers of optimization
updates. These are observed single-run timings, not repeated isolated throughput
benchmarks. Neither model underwent the full historical 80-epoch refinement.

### Held-out accuracy

All rows use the same 372 test episodes and commands. Errors are cube XY vector
RMSE; horizons are endpoint errors, not prefix-averaged errors.

| Model / method | One-step | 1 s endpoint | 2 s endpoint | 4 s endpoint |
|---|---:|---:|---:|---:|
| Medium, one-step screen | 4.49 mm | 4.77 cm | 9.18 cm | 13.05 cm |
| **Medium, refined** | **4.16 mm** | **3.17 cm** | **5.86 cm** | **8.00 cm** |
| Large, one-step screen | 4.17 mm | 3.87 cm | 7.75 cm | 12.33 cm |
| **Large, refined** | **3.75 mm** | **2.62 cm** | **5.67 cm** | **8.40 cm** |
| Historical 128, refined | 3.33 mm | 2.65 cm | 5.76 cm | 9.25 cm |

Refinement reduces four-second error by **38.7% for medium** and **31.8% for
large**, compared with their own saved one-step versions. Large has lower
one-step and shorter-horizon errors; medium has lower four-second aggregate
error. The historical model remains best for one-step error. Its training budget
and source optimization are much larger, so comparisons with it do not isolate
hidden-layer width.

### Drift and interaction tradeoffs

No-contact drift is maximum XY excursion over the whole predicted four-second
path, averaged across the same 145 truly low-motion, no-contact test episodes.
Clear pushes are the original 48 boundary clear-push test episodes. Broad pushes
are the original broad-source test episodes with endpoint displacement >2 cm.

| Model / method | Mean max no-contact drift | Clear-push 4 s error | Broad-push 4 s error |
|---|---:|---:|---:|
| Medium, one-step | 78.3 mm | 14.56 cm | 16.83 cm |
| **Medium, refined** | **20.8 mm** | **8.40 cm** | **12.24 cm** |
| Large, one-step | 64.5 mm | 11.29 cm | 17.63 cm |
| **Large, refined** | **16.8 mm** | **7.12 cm** | **13.42 cm** |
| Historical 128, refined | 13.8 mm | 8.44 cm | 14.72 cm |

Recursive refinement cuts average maximum false drift by roughly **74% for
both widths**, but every no-contact episode still exceeds the strict 2 mm
predicted-motion threshold. The 95th-percentile drifts fall from 153.7 to 47.7 mm
for medium and from 128.3 to 36.5 mm for large. This is substantial stabilization,
not elimination of false contact consequences.

Clear-push error improves by **42.3% for medium** and **36.9% for large**.
Mean predicted displacement direction cosine on clear pushes improves from
0.514 to 0.749 for medium and 0.788 to 0.826 for large. Large offers about 15%
lower clear-push error and 19% lower drift than refined medium, while medium is
about 9% better on broad pushes. The largest model is not uniformly better.

## Interpretation and next step

Recursive training made a larger practical difference than the initial width
increase alone. **Medium remains the practical default follow-up candidate**:
it has the best aggregate four-second result here and costs much less to refine.
**Large is useful when clear-push fidelity, one-step accuracy or lower drift is
the priority**, with the measured additional CPU cost.

These results strengthen the case that recursive loss helps stabilize the
learned dynamics. They do not establish a capacity ranking at convergence:
the source models had only 12 one-step epochs and both runs stopped after ten
recursive epochs, close to their best checkpoints. Both use one seed, and the
test split has now been reused across screening stages. An eventual selected
model should be checked on independent trajectories before stronger claims.
False motion remains common, so aggregate accuracy alone does not establish
readiness for cube MPC. No model defaults or existing checkpoints were replaced.

Normalization tensors and widths were verified unchanged from each source;
saved actual states and commands were checked exactly equal before/after.
All **26 pre-existing datasets and checkpoints** in the preservation audit had
unchanged hashes. Seven NN-only tests passed. No simulation data was collected.

The new bar plots `drift_comparison.png` and `clear_push_comparison.png` show the
behavior tradeoffs alongside `endpoint_comparison.png`. To watch the predictions:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_capacity_refinement/evaluations/medium_256
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_capacity_refinement/evaluations/large_512
```

Use the same zero-based `--episode` in the two runs to compare identical actual
motion and commands. Fresh exploratory predictions can use
`--random --condition still --checkpoint runs/arm_cube_capacity_refinement/medium_256/model.pt`
or the corresponding `large_512` checkpoint; this is optional exploration, not
part of the fixed-data experiment above.
