# Five-second autoregressive stress evaluation

The existing model handles unfamiliar moderate command sequences well, but
rapid reversals and near-limit commands expose substantial errors. The earlier
one-second results do not establish reliability over arbitrary MPC candidates.

## Method

- Existing checkpoint, no retraining; weights verified unchanged by SHA256:
  `2dd8ea900af3c43b5515ff6cee8f862f8a836cf88d3c31e9486502fd0fcc4f57`.
- Seed 2026, 12 newly generated episodes per group, 48 episodes total.
- One measured initial state followed by 50 predictions at 100 ms intervals.
  Each prediction consumes the previous predicted angles and velocities plus
  the next command. No measured-state corrections or prediction clipping.
- Ground truth uses the same MJCF and 2 ms physics timestep. The cube remains
  outside the arm workspace; all commands respect actuator limits.
- Training targets changed every 0.5 seconds. Stress groups use irregular holds
  of 0.1–0.7 s, rapid reversals every 0.1–0.3 s, limit holds of 0.7–1.2 s, and
  limit reversals every 0.1–0.4 s.
- Limit groups begin at 94–99% of joint range and command the range endpoints.
  Training initial angles were within ±0.9 / ±1.2 rad; training targets were
  approximately within ±1.2 / ±1.6 rad. Limits are ±1.57 / ±2.1 rad.

## Results

Endpoint angle RMSE across 12 episodes per group, in degrees, **shoulder / elbow**.
These measure errors at the given instant, not preceding trajectory averages.

| Commands | 1 second | 3 seconds | 5 seconds |
| --- | --- | --- | --- |
| Irregular, within training target range | 1.00 / 1.00 | 0.71 / 0.86 | 0.60 / 0.42 |
| Rapid reversals, within training target range | 11.33 / 6.22 | 7.10 / 6.91 | 4.61 / 6.15 |
| Limit-to-limit holds | 6.59 / 10.09 | 7.32 / 12.68 | 5.56 / 10.14 |
| Limit-to-limit rapid reversals | 19.07 / 15.70 | 17.31 / 17.72 | 28.01 / 18.12 |

Whole-trajectory angle RMSE is respectively 1.48 / 1.18, 8.92 / 6.41,
6.56 / 9.47, and 18.47 / 16.41 degrees. Peak individual errors in the
limit-reversal group reach 99.46 degrees shoulder and 45.53 degrees elbow.

During limit holds, predictions exceed nominal ranges at 95 of 600 sample
times, using a 0.001 rad exceedance threshold. Maximum overshoot is
4.46 / 3.15 degrees. Ground truth does not exceed nominal ranges.

Angle errors on the first prediction after a large target sign reversal are
20.97 / 22.73 degrees in the limit-reversal group, compared with 2.85 / 2.26
degrees for irregular moderate commands. The JSON includes velocity errors
and near-limit subset metrics as well.

## Interpretation

Error need not increase monotonically: stable servos can bring predicted and
actual states back toward a shared target. Small endpoint errors can hide
substantial intermediate failures.

In-range targets do not ensure familiar velocities or state/action combinations.
About 2–3% of moderate-group predicted inputs are outside at least one training
input's observed min/max range; all limit-group inputs are. This bounding-box
check is a rough coverage indicator, not a confidence estimate.

No MuJoCo joint-limit constraint activated during these runs, checked every
2 ms. This tests near-limit servo behavior, not impacts against mechanical
stops. It is an open-loop prediction audit, not a closed-loop MPC trial.

Before unrestricted MPC, expand training experience with varied command durations,
reversals while moving, and near-limit states/actions. Retain this checkpoint as
a baseline, and evaluate retrained models on fresh reserved sequences. Test
stop impacts separately before claiming those dynamics are learned. An initial
MPC trial should restrict ranges and horizons to a validated operating envelope.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m simple_arm.rollout_stress
.\.venv\Scripts\python.exe -m pytest -q
```

Outputs under `runs/rollout_stress/`: `report.json`, `trajectories.npz`,
`horizon_errors.png`, and a plot per group showing its worst whole-trajectory
angle MSE case. Raw arrays preserve every actual state, predicted state, action,
and group label. Nine tests passed, including fifty-step autoregression and
limit-exceedance checks.
