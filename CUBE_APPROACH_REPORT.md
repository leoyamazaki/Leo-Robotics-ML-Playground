# Smooth approach MPC experiment

Optional `--scoring approach` adds a gated tip-to-cube distance cost to the original objective. Defaults are `--approach-weight 0.1` and `--approach-radius 0.08` meters. Original scoring remains the default. Model inputs stay base-relative; the controller computes world XY tip geometry only for its cost. No training is performed.

Run a paired comparison (use a fresh output directory):

```powershell
.venv/Scripts/python -m simple_arm.cube_mpc compare-approach --device cuda --output runs/cube_cem_approach_new
```

The comparison defaults to `runs/arm_cube_online/recursive/model.pt`, uses cases 23/99/139/159 and seeds 42/43/44, and executes ten simulated seconds per run. CPU is the CLI default; this measured comparison uses an RTX 2060 SUPER. The current saved independent dataset has seed 592; historical benchmark guards require seed 292. Both modes use identical current saved states and targets, recorded with dataset/checkpoint hashes in config.json. Results are not a reproduction of the historical seed-292 experiment.

Viewer:

```powershell
.venv/Scripts/python -m simple_arm.cube_mpc view --checkpoint runs/arm_cube_online/recursive/model.pt --scoring approach --approach-weight 0.1 --approach-radius 0.08 --device cuda
```

Results and all trajectories are in `runs/cube_cem_approach_cuda_20261107`. Each run JSON records individual selected cost components, goal gates, predicted horizon displacement, first-step predicted versus actual displacement, actual physics contact duration and first-contact time, minimum actual tip-centroid distance, and latency. Contact means active cube contact with the pusher or arm links, sampled after every 2 ms physics step. Predicted motion is not counted as contact. Success requires error at most 2 cm throughout the final 0.5 seconds.

Verification: 36 focused tests passed, including seven approach tests plus non-overwrite protection, frozen residual model loading/execution, exact original cost preservation, neutral radius, observed-state gate, geometry, infeasible fallback, and command limits. Five legacy asset-dependent tests were excluded from the focused run; an initial run confirmed failures from the absent historical Medium checkpoint and seed-292 dataset mismatch. Existing viewer/device changes are preserved.

The preliminary CPU attempt in `runs/cube_cem_approach_20261107` was stopped before any completed run. Early CUDA timing samples overlapped that CPU attempt and include startup overhead; latency is descriptive, not an isolated speed benchmark.


Measured outcome: approach increased success from 6/12 to 7/12, but mean final error worsened from 1.93 cm to 3.83 cm. It improved final error in 4/12 pairs and regressed in 8/12. Mean actual contact rose from 1.84 s to 2.21 s. Case 99 seed 43 became successful; case 99 seed 44 and every case 159 run regressed, with case 159 seed 43 reaching 13.51 cm final error. These results do not support promotion.

Mean planning latency was 79.2 ms original and 80.4 ms approach; p95 was 87.6 ms and 86.6 ms. There were 18/1200 and 16/1200 plans over 100 ms. Early CUDA samples overlapped the abandoned CPU attempt and include startup overhead, so timing is descriptive. No fallbacks or actual joint-limit excursions occurred. Maximum executed command rate was 0.20000041 rad/s, within the 1e-6-radian execution tolerance.

The approach term helps in some cases but has substantial directional regressions. A 30-step horizon with five knots and 100 ms model steps remains a separate next experiment; it was not run here. Learned terminal values remain deferred.
