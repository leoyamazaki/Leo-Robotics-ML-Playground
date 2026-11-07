# Regenerated 0.2 rad/s dataset

Completed on 2026-11-07. This is a **new collection**, not a recovery of the lost
original dataset. The simulation geometry, cube mass/friction, four-second episode
length, 2ms physics, 20ms saved states and 100ms training transitions are unchanged.
Every joint position-target transition, including the first command relative to
the initial arm pose, is limited to **0.2 rad/s**. Actual joint speeds are measured,
not constrained or used as a rejection criterion.

## Generated artifacts

All paths below are relative to `data/arm_cube_contact_boundary/`.

|Archive|Seed|Episodes|Purpose|
|---|---:|---:|---|
|`broad.npz`|491|1,200|300 episodes each: smooth, slew-limited abrupt goals, coordinated motion and reversals|
|`focused.npz`|591|1,280|320 matched groups of overhead, near-miss, light-contact and clear-push episodes|
|`mixed.npz`|491 + 591 parent collections|2,480|Combined training/validation/test dataset|
|`independent.npz`|592|160|40 fresh boundary groups, evaluation only|
|`broad_independent.npz`|493|120|Fresh broad movements, evaluation only|

The mixed set contains **99,200 non-overlapping 100ms transitions**, split into
69,440 training, 14,880 validation and 14,880 test transitions. Episode counts are
1,736/372/372. All four episodes in each focused group share initial arm state and
commands, and remain in one split. Source split assignments are preserved by the
merge. Independent collections are excluded from this mixture.

A separate pilot used 120 broad episodes (seed 490) and 40 focused groups (seed
590). Its 280 episodes are saved under `data/arm_cube_rate_02_pilot/` and are not
added to the production training set.

## Changes needed for slow commands

Rate-limited trajectories use longer sweeps and direction holds so the arm can
cover useful distances in four seconds. Abrupt desired goals pass through the
same slew limiter; they never become abrupt actuator commands. Arm starts are
stationary. Broad data retain the original half-resting/half-moving cube starts
and randomized yaw, and the 70/30 targeted-path/workspace placement schedule.

The existing acceptance thresholds were retained: both broad joints move at
least 15 degrees; targeted broad episodes contain contact in at least 22.5% of
intervals and move the cube more than 2cm. Focused categories still require:

- Overhead: no contact, <=2mm excursion, verified close vertical link pass and pusher clearance >5mm.
- Near miss: no contact, <=2mm excursion and pusher clearance 0.5–5mm.
- Light contact: >0 and <=200ms contact, <=20mm excursion.
- Clear push: >=100ms contact and >=50mm excursion.

Thus these remain intentionally interaction-enriched datasets, not unfiltered
uniform-random samples. Single-joint-only and idle-arm episodes were not added
to this regeneration. Online fresh-data training is not implemented or started
as part of this collection.

## Measured coverage

|Dataset|Contact intervals|Cube endpoint displacement >2cm|Peak saved shoulder speed|Peak saved elbow speed|
|---|---:|---:|---:|---:|
|Broad|35.49%|72.33%|0.2618 rad/s|0.2855 rad/s|
|Focused|17.62%|25.00%|0.2686 rad/s|0.2713 rad/s|
|Mixed|26.27%|47.90%|0.2686 rad/s|0.2855 rad/s|
|Independent boundary|18.45%|25.00%|0.2495 rad/s|0.2652 rad/s|
|Independent broad|35.10%|71.67%|0.2549 rad/s|0.2724 rad/s|

The maximum commanded rate was 0.20000000000000018 rad/s, equivalent to 0.2
within floating-point tolerance. Speed statistics use the saved 20ms states;
they are not claims about exact maxima between samples. Focused acceptance
diagnostics still use every 2ms physics step.

Collection took approximately **103 seconds with eight worker processes**.
The five production NPZ archives total about **61.7 MB**. Broad collection
rejected 90 initializations and 69 insufficient-contact candidates, with no
joint-motion or joint-limit rejections. Focused collection used 702 trajectory
attempts, with 1,279 candidate initialization failures and 299 category mismatches.

## Verification and provenance

`rate_collection.json` records seeds, source hashes, archive hashes, timings,
split counts and speed/coverage statistics. `verification.json` confirms archive
hashes, slew limits including initial commands, all focused category predicates,
paired command equality, preserved source arrays/splits and distinct seeds.

A mixed-file metadata summary was corrected to report actual speed across both
sources, rather than inheriting the focused-only maximum. States, actions,
contacts, targets and split assignments were unchanged; the manifest records the
correction and corresponding source hashes.

|Archive|SHA-256|
|---|---|
|`broad.npz`|`b1b2c0c08fe5959d72a91aaa00f0e6bd1d0533386b73fdd4f168694c72078466`|
|`focused.npz`|`1722b59547bb1a555f1b0fa9bf029a57b35c2b49cfe53e25524bd90ebc1ae48d`|
|`independent.npz`|`d58dfd34cc0d82e6e911e79335dc507b83b7c93a30aa90803514d9c63a173437`|
|`broad_independent.npz`|`ec264298b2f7f36af1b07c817707599a54d3132b4f8783a803cf573c9641324c`|
|`mixed.npz`|`7af25eef51b61a769a662e2c79c7e57ed72649ca5912204becd8343f9773d520`|

## Reproduction and training handoff

```powershell
.\.venv\Scripts\python.exe -m simple_arm.rate_collection pilot --workers 8
.\.venv\Scripts\python.exe -m simple_arm.rate_collection collect --workers 8
```

`--root PATH` selects a different destination. Collectors refuse overwrites;
the workflow can reuse completed source archives after checking seed, size,
rate protocol and recorded hashes. Progress uses an in-place tqdm bar on TTYs
and stage summaries otherwise.

The production dataset is now ready for the commands in
[RESIDUAL_TRAINING.md](RESIDUAL_TRAINING.md). No network training was started in
this collection step, and controller defaults/checkpoints remain unchanged.
These binary datasets remain Git-ignored: copy the complete `data/` folder when
moving machines and retain it separately from the code repository.
