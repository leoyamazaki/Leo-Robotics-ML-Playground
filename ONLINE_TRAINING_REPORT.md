# Million-transition online residual experiment

Completed 2026-11-07 on the RTX 2060 Super using CUDA. The full run generated
**25,000 unique four-second episodes / 1,000,000 unique 100ms transitions** and
completed one-step training, recursive refinement and paired held-out evaluation.
Generation plus training took **530.4 seconds (8.84 minutes)**; evaluation followed.

## Protocol actually run

The architecture remains the exact 4,746,252-parameter, base-relative residual
MLP. Both losses were **unweighted**, with no contact or stationary multipliers.
Expected sampling was 90% uniform random replay / 10% focused training data.
Random generation rejected invalid physical starts only and retained valid
misses, idle episodes and unsuccessful pushes. Command slew stayed <=0.2rad/s
including the first command. Measured dense actual joint speed peaked at
0.2936rad/s, without being used as a rejection threshold.

One-step training consumed 20,000 fresh episodes over 20 collection epochs,
256 updates per epoch, Adam 0.001. It selected **epoch 17** by fixed one-step
validation MSE (0.0341528). Recursive training started from that winner and
consumed 5,000 additional episodes over 20 epochs, 64 updates per epoch,
fresh Adam 0.0002 and gradient norm cap 5. It selected **epoch 20**: validation
40-step state-scaled MSE fell from **0.0108018 at epoch zero to 0.00221639**.
Scores across stages have different definitions and should not be compared
directly. Epoch here denotes a fresh-data/update block, not a pass over a fixed
dataset. Samples/windows were drawn from a 5,000-episode FIFO buffer, batch 256.

Normalization was fitted once on the initial 1,000 training episodes plus
111 focused training episodes, using existing physical floors. It was preserved
bitwise through refinement. Initialization/sampling seeds were 91/93. Training
episodes used seed 791; fixed random validation/test used seeds 792/793, 120
episodes each. The focused supplement's original grouped train/validation/test
splits remained separate. Independent boundary and broad data were evaluated
without selecting weights or changing controller defaults.

## What random sampling produced

Across all one million random transitions:

- **98.2796%** met the existing stationary-interval definition.
- **0.5052%** contained arm/cube contact.
- The six command families had 4,215 / 4,154 / 4,112 / 4,156 / 4,162 / 4,201 episodes
  (idle, shoulder-only, elbow-only, independent goals, coordinated, reversals).
- 1,244 invalid physical starts were retried; no accepted episode had a recorded
  joint-limit excursion. Contact/motion outcomes were never used for acceptance.

The fixed random validation sample happened to contain no contact intervals;
its contact metric is explicitly unavailable. Fixed focused validation provides
the contact-sensitive selection component. Random test contained contact; all
reports retain cohort counts. More unique transitions therefore supplied much
more stationary/free-motion coverage, with relatively little new contact coverage.
The 10% focused supplement remains an explicit sampling choice, not an automatic
consequence of collecting a larger corpus.

## Paired results

Four-second cube XY endpoint RMSE, using identical test cases for both models:

|Evaluation set|One-step checkpoint|Recursive checkpoint|
|---|---:|---:|
|Random test, 120 episodes|2.837cm|**0.467cm**|
|Focused test, 192 episodes|5.866cm|**4.178cm**|
|Independent boundary, 160 episodes|5.057cm|**4.465cm**|
|Independent broad, 120 episodes|13.108cm|**12.639cm**|

Mean maximum predicted XY drift on stationary episodes:

|Evaluation set|One-step|Recursive|
|---|---:|---:|
|Random test|11.819mm|**1.310mm**|
|Focused test|20.935mm|**2.681mm**|
|Independent boundary|12.804mm|**2.342mm**|
|Independent broad|20.173mm|**4.192mm**|

Refinement improved rollout stability and all four aggregate four-second endpoint
scores, but **stationary one-step XY RMSE worsened on every set**. On independent
boundary data it rose from 0.056mm to 0.109mm; on broad data, 0.071mm to 0.171mm.
Random-test overall one-step XY RMSE also rose from 0.096mm to 0.104mm.
These regressions are retained in the report rather than hidden by rollout gains.
Independent broad endpoint error remains large, despite the small aggregate
random-test error. No matched smaller-network or fixed-data control was trained,
so this run does not isolate the benefit of size, fresh data, or replay.

Desktop CPU inference used two threads, 10 warm-ups and 50 timing samples per
model/batch size. Recursive-model median full-transition latency was **0.384ms**
at batch 1 and **10.684ms** at batch 256. These are not laptop measurements or
closed-loop controller benchmarks.

## Artifacts, monitoring and continuation

Run: `runs/arm_cube_online/`.

- `one_step/model.pt`, `recursive/model.pt`: validation-selected inference models.
- Each stage's `last.pt`: CPU-portable weights, Adam, sampling state, replay
  archive references, heldout hashes, history and preceding best checkpoint.
- `data/`: all consumed episodes in immutable compressed chunks, plus fixed
  validation/test archives; approximately **242.7 MB** in total.
- `status.html`, `status.json`: quiet live status; final state is complete.
- `history.json`, `loss.png` inside each stage: detailed metrics and curves.
- `evaluation/REPORT.md`, `evaluation/summary.json`: full paired metrics,
  1/2/4-second prefix/endpoint errors, false/missed motion and timing samples.
- `verification.json`: every archive SHA-256, unique sequential episode IDs,
  heldout seeds/hashes, target slew, 100ms command holds and normalization checked.
- `source.zip`: matching source, model XML and environment/handoff instructions.

The selected recursive checkpoint also passed the existing MPC adapter's
convention/physics checks and produced finite predictions. Controller defaults
were not switched. Tests: **34 passed**, including exact CPU continuation,
worker-order reproducibility, unweighted losses, complete replay windows and
CUDA-to-CPU continuation. Four historical MPC tests requiring the original
seed-292 benchmark/checkpoint were excluded; regenerated evaluation uses seed 592.
The real CUDA smoke run and full production run both completed evaluation.

See [ONLINE_TRAINING.md](ONLINE_TRAINING.md) for training, status, resume and
evaluation commands. Copy the **complete run folder**, matching code, and the
focused supplement to continue on the laptop. `runs/` is Git-ignored; retaining
the code repository alone does not retain this experiment's state.

The future parameter-matched three-hidden-layer 1,536-wide plain MLP remains
untrained. It should use a matched generation, sampling and optimization budget.
