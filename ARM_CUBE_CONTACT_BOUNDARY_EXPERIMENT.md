# Contact versus near-miss learning

## Protocol fixed before training

The current simulation has only about 2 mm clearance between the horizontal
links and the supported cube. An overhead sweep can look like a push while
producing no physical contact. The learned model has shown false cube motion
in such cases. This experiment changes training experience while keeping
geometry, friction, mass, architecture, optimizer settings and prediction
interval fixed. It adds no contact rules to the learned predictor.

Generate 320 groups of four four-second episodes, seed 291, using identical arm
initial states and commands within each group but different cube placements:

| Category | Physics-based acceptance |
|---|---|
| Overhead | No arm contact; actual maximum XY excursion <=2 mm; a horizontal link passes within 1–4 mm vertically above the cube; pusher stays >5 mm away |
| Pusher near miss | No contact; excursion <=2 mm; minimum pusher clearance 0.5–5 mm |
| Light contact | Active contact duration >0 and <=200 ms; excursion <=20 mm |
| Clear push | Contact duration >=100 ms; excursion >=50 mm |

All diagnostics use actual MuJoCo states at 2 ms intervals. Contact requires
an active penetrating arm/cube contact, excluding the floor. Signed geometry
distances and closest-point segments verify overhead passes and pusher clearance.
The categories are supervised-data selection and evaluation diagnostics only;
they are not model inputs. Cube maximum excursion is checked across the entire
episode, never inferred solely from final displacement.

Reuse smooth, abrupt, coordinated and reversal command families. Each group has
a varied 1–2 second approach followed by a position-target hold for the remainder
of its four seconds, exposing contact release, frictional stopping and drift.
Initial arm configurations span the legal ranges with a 0.2 rad margin; optional
initial arm velocities are within +/-0.3 rad/s. Commands retain the generator's
0.15 rad margin and 0.1–1.5 rad/s ramps. Actual joint-limit excursions are rejected.
Cube yaw is randomized across the full circle; focused cubes start at rest.

A simulated cube-free pusher path guides nearby cube-placement candidates.
Offsets bracket non-contact and small overlap around the capsule/box boundary;
actual interacting simulation determines the final class. Hard cases are retried
with a maximum of 100 candidate simulations per category per group and 100
trajectory attempts per group. Failure stops collection with diagnostics. These
are intentionally selected counterfactual examples, not a representative random
deployment distribution.

Run a 40-group pilot with seed 290 before the full collection. Split full
groups 224/48/48, keeping all four related episodes together. Merge the resulting
1,280 episodes with all 1,200 existing broad episodes while preserving their
original split assignments. The mixed set has 2,480 episodes and 99,200
100 ms transitions: 69,440/14,880/14,880 train/validation/test transitions.
The broader data retain moving-cube initial states and full four-second commands.

## Learning and independent comparison

Use the existing 15–128–128–128–12 SiLU MLP (36,620 parameters), residual targets,
sin/cos yaw and CPU inference. Train from scratch for 150 epochs with Adam 0.001,
batch size 256 and seed 91; select the best normalized one-step validation loss.
Normalize with combined training data only, retaining the existing physical scale
floors and 5x contact weights. Refine for 80 epochs with Adam 0.0002 and seed 93,
ten-step recursive training, 3x contact weights, gradient norm limit 5, and
checkpoint selection by full four-second validation rollout loss every five epochs.
Keep both checkpoints. No choice uses independent evaluation results.

Evaluate the previous varied/refined checkpoint, new one-step checkpoint and new
refined checkpoint on exactly the same independent 40 boundary groups (seed 292)
and 120 broad-motion episodes (seed 293). Also retain stationary/constant-velocity
cube baselines and original arm-model baseline predictions. The new independent
seeds and collections are never used for fitting or normalization.

Report one-step/contact-conditioned state errors, prefix-averaged and endpoint
errors at 1/2/4 seconds, drift distributions, false motion >2 mm, missed motion,
displacement direction/magnitude and arm/yaw errors. Main predeclared targets:

- At least 30% reduction in mean maximum predicted cube drift, averaged equally
  across the overhead and pusher-near-miss categories.
- No more than 10% regression in four-second cube endpoint RMSE on broad pushes,
  defined by actual endpoint displacement >2 cm.

Both targets must pass to call the experiment successful on these criteria.
Report failures without tuning on the test sets. Generalization remains limited
to fixed geometry/mass/friction and the sampled motion families.

## Reproduction

All new artifacts refuse overwrites. Existing models/data remain unchanged.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary collect --groups 40 --seed 290 --output data/arm_cube_contact_boundary/pilot.npz
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary collect
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary merge
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary train
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary refine
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary collect --groups 40 --seed 292 --output data/arm_cube_contact_boundary/independent.npz
.\.venv\Scripts\python.exe -m simple_arm.contact_varied collect --episodes 120 --seed 293 --dataset data/arm_cube_contact_boundary/broad_independent.npz
```

Evaluate each checkpoint with `contact_boundary evaluate --independent`, selecting
the boundary or broad independent dataset with `--dataset`, the model with
`--checkpoint`, and a fresh `--output` directory. Comparison directories under
`runs/arm_cube_contact_boundary/` are `boundary_old`, `boundary_firstpass`,
`boundary_new`, `broad_old`, `broad_firstpass` and `broad_new`. Then:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/independent.npz --checkpoint runs/arm_cube_dynamics_varied_recursive/model.pt --independent --output runs/arm_cube_contact_boundary/boundary_old
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/independent.npz --checkpoint runs/arm_cube_contact_boundary/firstpass/model.pt --independent --output runs/arm_cube_contact_boundary/boundary_firstpass
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/independent.npz --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt --independent --output runs/arm_cube_contact_boundary/boundary_new
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/broad_independent.npz --checkpoint runs/arm_cube_dynamics_varied_recursive/model.pt --independent --output runs/arm_cube_contact_boundary/broad_old
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/broad_independent.npz --checkpoint runs/arm_cube_contact_boundary/firstpass/model.pt --independent --output runs/arm_cube_contact_boundary/broad_firstpass
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_contact_boundary/broad_independent.npz --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt --independent --output runs/arm_cube_contact_boundary/broad_new
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary compare
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_contact_boundary/boundary_new --episode 0
```

Independent boundary episode indices follow `4 * group + category`: categories
0/1/2/3 are overhead/near-miss/light-contact/clear-push respectively. Use episode
1, 2 or 3 to inspect the other conditions with the same arm commands. **6/7**
browse, **8** pauses, **9** restarts. `--episode worst-still` selects the largest
trajectory error among genuinely low-motion saved cases. To compare the old
model, change only the run directory to `boundary_old` and keep the episode index.
Use `--random --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt
--condition still` for fresh exploratory still-cube examples.

## Validation

Tests cover category thresholds, analytic clearance bracketing, dense physical
overhead diagnostics, deterministic grouped splits, preservation of original
splits, no diagnostic features, dataset fingerprints, overwrite protection and
rejection of group leakage. Existing tests cover training-only normalization,
checkpoint loading, wrapped yaw residuals, recursive predicted-state feedback,
random viewer scenarios and the full arm-only/MPC functionality.

## Completed results

The pilot produced 40 valid groups. Full collection produced all **320 groups /
1,280 episodes**, with 320 examples of each category, from 522 trajectory
attempts. There were 268 candidate simulation/initialization failures and 533
category mismatches. The focused set contains 51,200 transitions at 100 ms;
the combined set contains **99,200 transitions** across 2,480 episodes, with
1,736/372/372 training/validation/test episodes. All initial arm states and commands
were checked identical within each group; all categories were rechecked against
their dense diagnostics. Every original broad-data state, action, contact label
and split assignment was checked unchanged in the merged dataset.

All focused overhead and near-miss examples had zero measured XY cube motion.
Near-miss pusher gaps were approximately 1–4 mm. Light-contact cube excursions
ranged from 0.033 to 18.37 mm and contacts lasted 2–86 ms. Clear-push excursions
ranged from 5.01 cm to 1.12 m, with 0.10–3.36 s active contact. The 160 independent
boundary episodes were generated separately using seed 292. The 120 independent
broad episodes (seed 293) included 95 with endpoint cube displacement over 2 cm.

One-step training selected **epoch 136** from 150. Recursive refinement selected
**epoch 25** from 80, reducing full validation rollout loss from 0.08243 to
0.06288. Normalization means for inputs and residual targets were independently
verified against the actual combined training split. Architecture and parameter
count remain unchanged. Same epoch counts mean more total optimizer updates on
the larger dataset; this is not an equal-compute ablation.

### Predeclared acceptance results

Mean maximum drift averages the two no-contact categories equally. Broad-push
RMSE uses only the 95 independently generated episodes with actual endpoint
displacement over 2 cm. Each comparison was checked to use byte-identical saved
actual states and commands across all three models.

| Model | Mean maximum no-contact drift | Broad-push 4 s endpoint RMSE | Drift target | Pushing target |
|---|---:|---:|---|---|
| Previous varied/refined | 29.12 mm | 13.03 cm | Reference | Reference |
| New one-step | 23.34 mm | 15.77 cm | Fail: 19.8% reduction | Fail: 21.1% regression |
| New refined | **15.87 mm** | **13.61 cm** | **Pass: 45.5% reduction** | **Pass: 4.5% regression** |

The refined checkpoint meets both practical criteria on these independent sets.
One-step fitting alone did not. This supports a useful reduction in accumulated
false motion, without eliminating the error or improving every task.

### No-contact drift and contact prediction

| Category | Old mean max drift | New mean max drift | New median / 95th percentile | New fraction exceeding 2 mm |
|---|---:|---:|---:|---:|
| Overhead | 20.31 mm | 8.39 mm | 7.16 / 19.89 mm | 92.5% |
| Pusher near miss | 37.93 mm | 23.36 mm | 14.12 / 65.49 mm | 100% |

The previous model exceeded 2 mm on every episode in both categories. Even after
training, most genuinely stationary cubes still accumulate false predicted
motion. A reduction in magnitude should not be interpreted as reliable binary
contact/no-contact discrimination.

| Category | Old 4 s endpoint XY RMSE | New 4 s endpoint XY RMSE | Stationary baseline |
|---|---:|---:|---:|
| Overhead | 24.32 mm | 10.04 mm | 0 mm |
| Pusher near miss | 50.13 mm | 34.25 mm | 0 mm |
| Light contact | 50.60 mm | 34.46 mm | **2.32 mm** |
| Clear push | 132.49 mm | **128.66 mm** | 168.11 mm |

The constant-velocity cube baseline coincides with stationary prediction in XY
on the focused set because every cube starts at rest. It remains distinct on the
broad set. Light contact is still substantially overpredicted; stationary
prediction is much better for these tiny actual displacements. Clear pushes
remain better predicted than a stationary cube, while still having large errors.

For all independent boundary episodes, new one-step cube XY RMSE is **2.57 mm**
(old 2.71 mm). Autoregressive endpoint RMSE at 1/2/4 s is **3.57/6.76/6.89 cm**.
Prefix-averaged errors up to those horizons are 2.05/3.91/5.56 cm; these are
different metrics. One-step arm angle RMSE is 0.59/0.71 degrees, angular velocity
0.080/0.092 rad/s, cube yaw 1.24 degrees, cube XYZ velocity 0.027/0.046/0.008 m/s,
and cube yaw rate 0.644 rad/s. Four-second yaw error is 16.31 degrees.

Across all 120 fresh broad episodes, endpoint XY RMSE changes from 11.63 to
12.14 cm, versus 21.59 cm stationary and 33.24 cm constant velocity. New one-step
XY RMSE is 4.44 mm. Four-second arm errors are 1.61/1.98 degrees and yaw error
30.58 degrees. Full state errors, contact-conditioned scores, horizon results
and baseline comparisons are in each run's evaluation JSON. `comparison.json`
records the acceptance criteria and both checkpoints' pass/fail results.

### Revisit the original episode 77

The previously inspected seed-192 evaluation was also replayed through the new
checkpoint as a qualitative reference, without refitting or checkpoint selection.
Its exact episode 77 has actual maximum cube excursion **1.80 mm**. Maximum
predicted excursion falls from **113.72 mm to 39.96 mm**; endpoint error falls
from 114.30 mm to 38.43 mm. This familiar example improves but remains wrong.
Its reference replay is separate from the new independent acceptance sets.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_boundary evaluate --dataset data/arm_cube_varied_unseen.npz --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt --independent --output runs/arm_cube_contact_boundary/reference_new
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_contact_boundary/reference_new --episode 77
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_varied_unseen_recursive --episode 77
```

The last two commands watch new versus old predictions for identical simulator
motion. For fresh scenarios using the new model:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --random --condition still --checkpoint runs/arm_cube_contact_boundary/recursive/model.pt
```

The viewer's existing default run/checkpoint remain unchanged; explicitly select
the new run or checkpoint as above. Independent boundary episode 148 is the
largest new overhead drift (23.95 mm); episode 5 is the largest new near-miss drift
(148.27 mm). Use `--run runs/arm_cube_contact_boundary/boundary_new --episode 5`
to inspect that remaining failure, and `boundary_old` for its old prediction.

### Limits and next questions

- The focused overhead sampler accepted its first valid interior upper-link
  placement in every group. Those cubes are at radius 0.18 m from the base;
  they do not establish coverage across arbitrary positions beneath both links.
  Near-miss examples also include some lower-link overpasses, and the original
  broad data remain in training, but wider overhead placement is a useful next
  controlled extension. The independent set shares this sampling structure.
- Many near misses still falsely move the cube, and some errors remain over
  10 cm. Arm-state drift and contact-boundary approximation can both contribute;
  this experiment does not isolate their relative contributions.
- Passing the 10% regression allowance does not mean pushing improved. The new
  model is a tradeoff: less false drift, slightly worse broad pushing. Neither
  model is ready to justify cube MPC from prediction accuracy alone.
- Mass, friction and geometry were fixed, and comparisons use one fresh seed
  per suite. There is no claim of exhaustive state/action coverage, unseen
  physical-property generalization, or a solved contact representation.

All **36 tests passed**. Saved replay of episode 77 and fresh random still-cube
mode launched successfully with the new model. All 16 pre-existing dataset and
checkpoint files in the preservation audit had unchanged SHA256 hashes. New
checkpoints:

- One-step: `runs/arm_cube_contact_boundary/firstpass/model.pt`, SHA256
  `0d27b3eddf87c031bd7c3eeaf2e89849c3e9f3344285201c745e5978b2d0b2e3`
- Refined: `runs/arm_cube_contact_boundary/recursive/model.pt`, SHA256
  `fdef318b6d971ce2ca3efdb2f5c6b7cd0958a3ba3e53a57412e4950c9f87610d`
