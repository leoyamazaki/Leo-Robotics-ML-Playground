# 200 additional single-step epochs at 5x

Continuation from the previous 12-epoch 5x final checkpoint. Same original mixed dataset, normalization, architecture, ground-truth single-step inputs, and 5x stationary XY objective. Fresh Adam at 0.001, batch 256, seed 91, two CPU threads. No collection or recursive training. The final checkpoint has 212 stationary-weighted epochs in its lineage.

Training collections: 1,200 broad varied episodes (smooth/abrupt/coordinated/reversal, seed 191) and 1,280 focused boundary episodes (overhead/near-miss/light-contact/clear-push, seed 291). Their unchanged training splits provide 840+896 episodes = 69,440 transitions. Validation/test remain 14,880 transitions each. Independent boundary seed 292 and broad seed 293 are evaluation only. No slow-0.2 collection is included.

Training took 206.38s for 54,400 updates. Diagnostic best unweighted-validation epoch: 30 (relative to this continuation). The fixed final epoch 200 is the primary comparison. Independent results never select a checkpoint.

## boundary

|Model|Stationary mm|Moving-contact mm|Clear-push mm|False 2mm %|Missed contact 2mm %|AR 4s endpoint cm|Push 4s cm|
|---|---:|---:|---:|---:|---:|---:|---:|
|frozen_medium|1.8945|11.4092|5.0179|8.4442|2.3923|10.5878|13.3568|
|previous_5x_12|1.5129|8.7452|3.8648|3.6307|1.4354|7.5656|13.4213|
|continued_final_200|1.1504|6.7991|3.0706|1.3964|5.7416|6.1233|11.5724|
|continued_best_validation|1.1499|7.5639|3.3425|2.8914|3.3493|6.8124|12.9744|

### Final epoch 200 changes versus previous 12-epoch 5x

- stationary one-step RMSE change: -0.3625 mm (-23.9621%).
- moving one-step RMSE change: -2.1472 mm (-23.7502%).
- moving_contact one-step RMSE change: -1.9462 mm (-22.2540%).
- all 4s endpoint change: -1.4422 cm; positive is a regression.
- no_contact 4s endpoint change: -1.5308 cm; positive is a regression.
- light_contact 4s endpoint change: -2.0484 cm; positive is a regression.
- clear_push 4s endpoint change: -1.8488 cm; positive is a regression.

## broad

|Model|Stationary mm|Moving-contact mm|Clear-push mm|False 2mm %|Missed contact 2mm %|AR 4s endpoint cm|Push 4s cm|
|---|---:|---:|---:|---:|---:|---:|---:|
|frozen_medium|2.8747|10.8967|6.1509|18.5196|1.4541|19.7858|21.8815|
|previous_5x_12|2.1220|8.2623|4.9653|5.8101|1.5660|14.6887|16.2844|
|continued_final_200|1.1034|7.3424|4.0263|1.3408|5.1454|12.1938|13.6618|
|continued_best_validation|1.3344|7.7262|4.6031|2.1788|3.4676|13.3230|14.9337|

### Final epoch 200 changes versus previous 12-epoch 5x

- stationary one-step RMSE change: -1.0186 mm (-48.0018%).
- moving one-step RMSE change: -1.2969 mm (-14.8052%).
- moving_contact one-step RMSE change: -0.9199 mm (-11.1335%).
- all 4s endpoint change: -2.4949 cm; positive is a regression.
- no_contact 4s endpoint change: -3.1759 cm; positive is a regression.
- light_contact 4s endpoint change: unavailable cm; positive is a regression.
- clear_push 4s endpoint change: -2.4576 cm; positive is a regression.

## Artifacts and checks

Each continued checkpoint has separate predictions and full metrics, including counts, 0.5mm/2mm false/missed motion, contact displacement magnitude/direction, light/clear case errors, interval errors, stationary/near-miss/overhead drift, and teacher-forced versus autoregressive errors. See summary.json and evaluations/. Learning and four-second curves are saved as PNG files.

All 51 pre-existing datasets/checkpoints/XML files retain their SHA-256 hashes: True. Normalization is bitwise unchanged; all evaluation states/actions exactly match the previous experiment. MuJoCo stepping was prohibited throughout; recursive prediction was additionally prohibited during training. No controller checkpoint was switched.

## Contact detection tradeoff

- boundary: missed actual contact motions >2mm increased from 1.44% to 5.74% (+4.31 percentage points).
- broad: missed actual contact motions >2mm increased from 1.57% to 5.15% (+3.58 percentage points).

These increases remain within 5 percentage points of the previous 5x model, while stationary, moving-contact, and clear-push one-step RMSE improve on both independent datasets. This is a continuation comparison, not a new matched 1x/5x schedule experiment. No controller checkpoint was changed.

## Saved artifacts

- [Final epoch 200 checkpoint](runs/arm_cube_one_step_stationary_5x_200/training/final.pt)
- [Best validation checkpoint, epoch 30](runs/arm_cube_one_step_stationary_5x_200/training/best.pt)
- [History and timing](runs/arm_cube_one_step_stationary_5x_200/training/training.json)
- [Full independent metrics](runs/arm_cube_one_step_stationary_5x_200/summary.json)
- [Provenance and collection details](runs/arm_cube_one_step_stationary_5x_200/config.json)
- [Verification](runs/arm_cube_one_step_stationary_5x_200/verification.json)
- [Validation learning curves](runs/arm_cube_one_step_stationary_5x_200/validation_overall.png)
- [Boundary four-second errors](runs/arm_cube_one_step_stationary_5x_200/boundary_four_second_errors.png)
- [Broad four-second errors](runs/arm_cube_one_step_stationary_5x_200/broad_four_second_errors.png)
