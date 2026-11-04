# Varied two-joint contact experiment

The earlier contact dataset mainly swept the shoulder while holding the elbow.
This experiment broadens both joint motion and the cube's initial conditions.
It preserves the physical model and all earlier datasets/checkpoints. No cube
MPC is introduced.

## Protocol

- 1,200 independent four-second episodes, seed 191, split by complete episode
  into 840 training, 180 validation and 180 test episodes. A separate 120-episode
  evaluation uses seed 192. All model comparisons use the identical fresh data.
- The arm's physical joint/actuator limits come from MJCF. Initial angles and
  commanded targets retain a 0.15 rad margin. Actual limit excursions are checked
  every 2 ms; offending episodes are rejected. Initial arm velocities are sampled
  up to +/-0.5 rad/s on 70% of candidates, with zero rates close to limits.
- Equal smooth, abrupt, coordinated and reversal families. Smooth trajectories
  have independent timing/rates/pauses for each joint; coordinated trajectories
  have correlated rates and same/opposing directions. Ramps use 0.1–1.5 rad/s.
  Abrupt targets jump within +/-1.2 rad of the initial pose with 100–800 ms holds.
  These are position commands, not velocity control: actual velocities can exceed
  the commanded ramp rates during transients or collisions.
- Cube positions: 70% near prospective pusher paths, 30% sampled through random
  reachable arm configurations. This is broad workspace sampling, not uniform
  Cartesian area sampling. Cube yaw is sampled across the full circle. Half the
  episodes start the cube at rest; the other half start with 0.02–0.15 m/s planar
  speed and +/-1 rad/s yaw rate. Velocities are applied after gravity settling.
- Cube mass 0.1 kg and friction 0.7 are fixed. Roll/pitch remain constrained;
  vertical position/velocity result from gravity and support, not arbitrary
  randomized values. Arm/cube overlap and cube/base overlap are excluded initially.
- All accepted episodes move both joints at least 15 degrees. Targeted episodes
  must have contact in at least 22.5% of 100 ms windows and cube displacement over
  2 cm. Candidates are retried up to 50 times. This intentionally enriches contact
  and is a selection bias: evaluation measures this procedural family, not all
  possible motions or a uniformly random deployment distribution.
- Require aggregate contact coverage >=15% and moving-cube episodes >=50%, plus
  >=75% episodes with both joints moving at least 15 degrees. Run a 120-episode
  pilot first. Dataset JSON records actual coverage, rejected candidates, initial
  conditions, movement families, velocities, positions and command-change ranges.

## Learning and comparison

Reuse the 13-state/2-action input and 12-component residual output, including
sin/cos yaw and wrapped yaw increments. Contact is diagnostic/supervision only;
no future contact or simulator truth enters the model inputs. Use the existing
15–128–128–128–12 SiLU MLP, CPU Adam, training-only normalization and physical scale
floors. First train 150 epochs with contact-weighted one-step loss; then refine
80 epochs with ten-step recursive loss. Select using validation loss only, keeping
the initial and refined checkpoints separately.

Compare the old refined model, new one-step model and new refined model against
the same fresh episodes. Include stationary and constant-velocity cube baselines,
original arm-only predictions, one-step contact-conditioned errors, 1/2/4-second
autoregressive errors, endpoint direction/magnitude, false/missed motion and
separate results for movement families and moving/resting cube starts.
Prefix-averaged errors and endpoint errors are distinct metrics.

## Reproduction

Run in the project directory with the existing virtual environment. All commands
refuse to overwrite existing new datasets or run directories; choose fresh paths
when repeating experiments. Old collectors/training commands remain available.

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_varied collect --episodes 120 --dataset data/arm_cube_varied_pilot_v2.npz
.\.venv\Scripts\python.exe -m simple_arm.contact_varied collect
.\.venv\Scripts\python.exe -m simple_arm.contact_varied train
.\.venv\Scripts\python.exe -m simple_arm.contact_varied refine --output runs/arm_cube_dynamics_varied_recursive
.\.venv\Scripts\python.exe -m simple_arm.contact_varied collect --episodes 120 --seed 192 --dataset data/arm_cube_varied_unseen.npz
.\.venv\Scripts\python.exe -m simple_arm.contact_varied evaluate --dataset data/arm_cube_varied_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics_recursive/model.pt --output runs/arm_cube_varied_unseen_old
.\.venv\Scripts\python.exe -m simple_arm.contact_varied evaluate --dataset data/arm_cube_varied_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics_varied/model.pt --output runs/arm_cube_varied_unseen_firstpass
.\.venv\Scripts\python.exe -m simple_arm.contact_varied evaluate --dataset data/arm_cube_varied_unseen.npz --all-episodes --checkpoint runs/arm_cube_dynamics_varied_recursive/model.pt --output runs/arm_cube_varied_unseen_recursive
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_varied_unseen_recursive
.\.venv\Scripts\python.exe -m simple_arm.contact_viewer --run runs/arm_cube_varied_unseen_recursive --episode worst
```

The overlay shows recorded actual MuJoCo motion in its original colors and a
translucent magenta prediction. Display interpolation between 100 ms predictions
does not add physics or correct the model. 6/7 browse, 8 pauses, 9 restarts.

## Validation

Tests cover initial velocities, invalid initial conditions, deterministic legal
commands, full reproducible collection, contact/motion coverage, episode splits,
100 ms command holds and compatibility with existing normalization/training.
Existing tests cover yaw residuals, missing privileged contact inputs, recursively
predicted state feedback, checkpoint loading and all arm-only/MPC behavior.

## Completed experiment results

The final 1,200-episode collection passed all gates: 48,000 transitions at 100 ms,
with 33,600/7,200/7,200 train/validation/test transitions. Both joints travelled
over 15 degrees in every episode, and both moved by more than 0.005 rad in 83.1%
of 100 ms intervals. Median shoulder/elbow excursion was 65.77/66.37 degrees;
95th percentiles were 111.01/121.71 degrees. Actual angular-velocity 95th
percentiles were 2.03/2.12 rad/s, with rare peaks of 10.35/12.18 rad/s from abrupt
commands. This is substantially more demanding than the previous shoulder sweeps.

Contact occurred in 22.71% of 100 ms windows; 914/1,200 episodes (76.17%) moved
the cube over 2 cm. Condition counts were 35,503 no-contact, 10,085 onset, 9,126
ongoing contact, 10,122 release and 1,597 sliding windows. Masks overlap. There
were 916 initialization rejections, 9 insufficient-joint-motion rejections and
2,359 insufficient-contact/displacement rejections. No joint-limit rejections or
accepted violations occurred. The accepted data are consequently conditioned on
successful interaction; these rejection statistics are part of the experiment.

Initial cube X spanned -0.214 to 0.556 m and Y -0.546 to 0.555 m. Exactly half
the cubes started with planar/yaw motion; all had randomized yaw. Since every
accepted episode moves both joints, the new dataset does not cover completely
idle-arm or single-joint-only episodes; the old experiments remain available as
references. Variation is broad within the documented envelope, not exhaustive
coverage of all physically possible state/action combinations.

Training used the planned 150 one-step epochs, selecting epoch **148** using
validation loss. Recursive refinement ran 80 epochs, selecting epoch **30** by
four-second validation rollout loss: 0.10253 versus 0.12798 before refinement.
The initial and refined models are preserved. Adam learning rates were 0.001 and
0.0002, batch size 256, with training seeds 91 and 93 respectively; collection
seeds are independent of training seeds. The architecture has 36,620 parameters.

### Independent evaluation: 120 episodes, seed 192

These episodes have 22.02% contact windows and 90 episodes with over 2 cm cube
displacement. Every comparison was checked to use exactly identical saved actual
states and command sequences. No checkpoint selection used these episodes.

| Predictor | One-step cube XY vector RMSE | Four-second endpoint XY vector RMSE |
|---|---:|---:|
| Previous refined contact model | 11.27 mm | 39.41 cm |
| New one-step-trained model | 5.86 mm | 19.35 cm |
| New recursively refined model | **5.75 mm** | **18.18 cm** |
| Stationary cube baseline | 16.81 mm | 24.17 cm |
| Constant-velocity cube baseline | 8.00 mm | 34.93 cm |

The refined model reduces endpoint RMSE by 53.9% relative to the old model and
24.8% relative to stationary prediction. Mean displacement-direction cosine for
actual motion over 2 mm is 0.815, and displacement-magnitude MAE is 5.41 cm.
The independent constant-velocity baseline is now different from stationary
prediction because half the cubes start moving; friction rapidly dissipates that
motion, which constant velocity does not model.

| Movement family / initial cube state | New refined endpoint RMSE | Stationary | Constant velocity |
|---|---:|---:|---:|
| Smooth | 12.38 cm | 24.66 cm | 34.39 cm |
| Abrupt | 31.11 cm | 33.01 cm | 37.44 cm |
| Coordinated | 7.49 cm | 18.50 cm | 34.38 cm |
| Reversals | 12.01 cm | 17.21 cm | 33.38 cm |
| Cube initially at rest | 19.40 cm | 27.03 cm | 27.03 cm |
| Cube initially moving | 16.87 cm | 20.91 cm | 41.34 cm |

One-step contact-conditioned XY errors:

| Condition | Windows | Learned | Stationary | Constant velocity |
|---|---:|---:|---:|---:|
| No contact / not sliding | 3,583 | 1.67 mm | 0.015 mm | 0.025 mm |
| Contact onset | 976 | 10.82 mm | 32.12 mm | 16.56 mm |
| Ongoing contact | 880 | 9.88 mm | 34.51 mm | 13.73 mm |
| Contact release | 971 | 11.37 mm | 34.43 mm | 16.64 mm |
| Sliding | 160 | 11.13 mm | 35.06 mm | 13.79 mm |

Autoregressive XY errors averaged across the entire prefix up to 1/2/4 seconds
are 2.69/7.16/11.12 cm. These prefix averages must not be confused with the
endpoint errors at 1/2/4 seconds of 5.37/11.35/18.18 cm. One-step arm angle RMSE is 1.13/1.01
degrees and angular-velocity RMSE 0.175/0.131 rad/s. One-step cube yaw RMSE is
3.17 degrees; linear-velocity XYZ RMSE is 0.057/0.067/0.017 m/s and yaw-rate
RMSE 0.801 rad/s. Full metrics are saved in the evaluation JSON files.

On the separate 180-episode test split of the training collection, new one-step
and refined endpoint errors are 13.59 cm and **11.84 cm**, versus 19.53 cm for
stationary and 34.03 cm for constant velocity. Refined direction cosine is 0.845.
The independent seed-192 set is harder; report both rather than selecting the
more favorable result.

### Failures and interpretation

- Abrupt movements remain the weakest family. Worst replay episode **65** sends
  the actual cube from roughly (0.259, -0.454) to (0.888, 0.550) m, while the
  model ends near (0.703, -0.334) m: a large contact consequence/direction error.
  Small accumulated arm/contact timing errors can change later collisions.
- False motion is still common: at four seconds, 28/29 cases with actual cube
  displacement <=2 mm have predicted displacement >2 mm. One-step false-motion
  fraction is 13.5%; one-step missed-motion fraction is 1.9%. Do not interpret
  aggregate pushing accuracy as a reliable stationary-state model. Predicted
  displacement in these 29 nearly stationary cases has median 9.5 mm and maximum
  11.3 cm: the false-motion percentage uses a strict 2 mm threshold.
- Four-second yaw RMSE remains 32.96 degrees. Arm endpoint errors are 2.94/1.39
  degrees, versus 0.62/0.25 degrees from the original arm-only predictor used by
  the cube baselines. Broader contact learning has not improved every variable.
- Best replay episode **86** finishes within about 1.3 mm of the actual cube
  endpoint. This illustrates a success, not typical performance.
- Only one held-out seed and the same procedural movement families are tested.
  There is no claim about unseen mass/friction, hard stops, full 3D rotation,
  arbitrary command frequencies or deployment-safe control. No predictions were
  clipped or corrected using future simulator states.

The network now predicts useful directions and approximate magnitudes for much
broader two-joint interactions and beats both simple cube baselines overall.
Large abrupt-contact errors and false drift remain clear reasons to continue
predictive-model evaluation before cube manipulation MPC.

### Artifacts and verification

The main checkpoint is `runs/arm_cube_dynamics_varied_recursive/model.pt`.
Independent replay/results are in `runs/arm_cube_varied_unseen_recursive/`;
corresponding `old` and `firstpass` directories contain matched comparisons.
Test-split evaluations are in `runs/arm_cube_varied_test_firstpass/` and
`runs/arm_cube_varied_test_recursive/`. To reproduce those extra evaluations:

```powershell
.\.venv\Scripts\python.exe -m simple_arm.contact_varied evaluate --checkpoint runs/arm_cube_dynamics_varied/model.pt --output runs/arm_cube_varied_test_firstpass
.\.venv\Scripts\python.exe -m simple_arm.contact_varied evaluate --checkpoint runs/arm_cube_dynamics_varied_recursive/model.pt --output runs/arm_cube_varied_test_recursive
```

All **29 tests passed**, including an evaluation-provenance guard against potential
episode overlap. The new 3D replay launched successfully in a five-second
smoke run. VS Code also has a **Watch varied contact predictions** launch option.
All four original arm/contact datasets and four arm/contact checkpoints were
verified unchanged by SHA256. New model SHA256 hashes:

- One-step: `7ca77ba76c9f81dbc17cf4034cbbdacb99391f89ceed7b2219f9598e7fa30be4`
- Refined: `f7df639b8c15bb6a92e98668ab44744386d36a1adc5f01d14b502e1717e4cdd9`
