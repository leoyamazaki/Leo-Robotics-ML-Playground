# One-step stationary-weighted Medium experiment

Recommendation: **weight_5**. This is a one-step comparison, not a deployment endorsement.

Each model starts from runs/arm_cube_capacity_screen/medium_256/model.pt and uses 12 additional epochs, batch size 256, fresh Adam at learning rate 0.001, seed 91, and two CPU threads. Original 99,200-transition data and episode splits 1,736/372/372 remain unchanged. No collection or recursive training occurs.

Features, architecture, normalization, 12 displacement targets, and 100 ms interval are preserved. Only stationary squared normalized residuals in columns 4 and 5 receive the multiplier. The denominator stays 12, then the original 5x contact-transition weight is applied. Small actual displacements within tolerance remain the target. Histories include epoch zero and post-epoch train/validation weighted objective, unweighted overall error, stationary/moving XY error, and unrelated-output error.

All primary comparisons use the fixed final epoch. Best unweighted-validation checkpoints include epoch zero and remain secondary diagnostics. Independent data never select checkpoints.

A multiplier above 1x must improve stationary XY RMSE on both datasets; on each, moving-contact and clear-push one-step RMSE must stay within 10% of 1x, and missed motion at 2mm may increase at most 5 percentage points. Among eligible weights, maximize average relative stationary improvement, breaking ties toward the smaller weight; if none qualify, retain 1x.

Stationary intervals require planar speed <=5 mm/s and excursion <=0.5 mm across the actual start and all five subsequent saved samples (unless overridden in config.json). XY errors below are vector RMSE. Rates use strictly greater than the reporting threshold; missed motion requires actual contact. Empty cohorts are unavailable.

## boundary

|Model|Stationary mm|Moving mm|Moving contact mm|Clear push mm|False 0.5mm %|False 2mm %|Missed 2mm %|4s endpoint cm|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|initialization|1.8945|11.4441|11.4092|5.0179|68.2274|8.4442|2.3923|10.5878|
|weight_1|2.1644|9.4319|9.1626|3.9901|58.8303|6.5878|0.4785|8.6404|
|weight_5|1.5129|9.0406|8.7452|3.8648|38.5083|3.6307|1.4354|7.5656|
|weight_20|1.0231|9.2728|9.0714|4.0350|18.8106|1.5771|3.8278|7.2540|
|weight_100|0.4949|10.5710|10.7184|4.6321|6.9821|0.6243|9.5694|7.4595|

Full metrics and predictions are saved separately per model and dataset. Teacher-forced XY errors are displacement errors; no prediction is fed back during training. Contact-motion direction requires >2 mm actual displacement and is measured against the actual displacement vector. Zero-length predictions have undefined angle and fail the joint magnitude/direction accuracy measure.

### Cohorts, case errors, and interval errors

|Model|Stationary n|Moving n|Moving contact n|Light case n / RMSE mm|Clear case n / RMSE mm|No-contact interval n / mm|Light interval n / mm|Active-push interval n / mm|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|initialization|6087|313|268|1600 / 2.6180|1600 / 5.0179|5984 / 2.1472|76 / 8.8922|209 / 12.5969|
|weight_1|6087|313|268|1600 / 3.0907|1600 / 3.9901|5984 / 2.3542|76 / 10.0948|209 / 9.8725|
|weight_5|6087|313|268|1600 / 2.2025|1600 / 3.8648|5984 / 1.7658|76 / 7.6497|209 / 9.6762|
|weight_20|6087|313|268|1600 / 1.5414|1600 / 4.0350|5984 / 1.3612|76 / 5.7409|209 / 10.1941|
|weight_100|6087|313|268|1600 / 0.8919|1600 / 4.6321|5984 / 0.9707|76 / 5.1816|209 / 12.1015|

Boundary case labels are saved collector kinds. Broad light contact requires >0 and <=0.2s contact and <=2cm excursion; broad clear push requires >=0.1s contact and >=5cm excursion. Interval light contact means 1..10 active substeps out of 50; active push means contact and >2mm net motion. These interval cohorts may overlap. Whole-episode rollout stationarity uses <=2mm excursion from the episode's initial position, distinct from the six-sample transition label.

### False and missed motion, including evaluation-only zero XY

|Model|Threshold mm|Stationary n|False count / %|Contact-motion n|Missed count / %|
|---|---:|---:|---:|---:|---:|
|initialization|0.5|6087|4153 / 68.2274|251|0 / 0.0000|
|initialization|2|6087|514 / 8.4442|209|5 / 2.3923|
|weight_1|0.5|6087|3581 / 58.8303|251|0 / 0.0000|
|weight_1|2|6087|401 / 6.5878|209|1 / 0.4785|
|weight_5|0.5|6087|2344 / 38.5083|251|1 / 0.3984|
|weight_5|2|6087|221 / 3.6307|209|3 / 1.4354|
|weight_20|0.5|6087|1145 / 18.8106|251|3 / 1.1952|
|weight_20|2|6087|96 / 1.5771|209|8 / 3.8278|
|weight_100|0.5|6087|425 / 6.9821|251|16 / 6.3745|
|weight_100|2|6087|38 / 0.6243|209|20 / 9.5694|
|zero_xy_reference|0.5|6087|0 / 0.0000|251|251 / 100.0000|
|zero_xy_reference|2|6087|0 / 0.0000|209|209 / 100.0000|

### Predicted displacement magnitudes and contact direction

|Model|Stationary predicted mean / median / p95 / max mm|Contact n|Predicted / actual mean mm|Magnitude RMSE mm|Direction n / mean angle deg|Wrong direction %|Within 30deg and >2mm %|
|---|---:|---:|---:|---:|---:|---:|---:|
|initialization|1.0754 / 0.7095 / 3.1908 / 25.4527|209|20.7817 / 22.5309|11.9602|209 / 10.1081|0.4785|91.8660|
|weight_1|0.9450 / 0.5978 / 2.3956 / 55.9333|209|22.3281 / 22.5309|9.0840|209 / 9.9617|0.9569|96.1722|
|weight_5|0.6487 / 0.3994 / 1.5734 / 42.2504|209|21.5971 / 22.5309|8.8400|209 / 9.8778|1.4354|96.6507|
|weight_20|0.4149 / 0.2712 / 0.9752 / 26.4883|209|20.2837 / 22.5309|9.3062|209 / 10.7421|1.9139|92.3445|
|weight_100|0.2181 / 0.1371 / 0.5640 / 11.7299|209|18.4773 / 22.5309|11.3757|209 / 12.0748|1.9139|85.6459|
|zero_xy_reference|0.0000 / 0.0000 / 0.0000 / 0.0000|209|0.0000 / 22.5309|33.1602|0 / unavailable|0.0000|0.0000|

Wrong-direction rate divides opposite predictions by all contact-motion samples. Zero predictions have undefined direction; the joint >2mm/30deg measure counts them as failures. The zero-XY reference is never part of inference or deployment.

### Four-second diagnostics

|Model|Cohort|Episodes|TF one-step mm|AR prefix mm|AR increment mm|AR 4s endpoint cm|TF summed 4s cm|
|---|---|---:|---:|---:|---:|---:|---:|
|initialization|all|160|3.1335|78.7016|6.7839|10.5878|3.3100|
|initialization|no_contact|80|1.9029|61.4233|4.3341|8.8764|3.0131|
|initialization|light_contact|40|2.6180|79.4322|5.9502|10.6027|3.6183|
|initialization|clear_push|40|5.0179|104.5018|10.5408|13.3568|3.5459|
|initialization|actual push (>2cm endpoint, contact)|40|unavailable|unavailable|unavailable|13.3568|unavailable|
|weight_1|all|160|2.9676|66.5274|6.2286|8.6404|3.1456|
|weight_1|no_contact|80|2.2082|39.2205|3.5670|5.1196|2.5158|
|weight_1|light_contact|40|3.0907|53.6252|5.0350|6.7179|3.2873|
|weight_1|clear_push|40|3.9901|108.4039|10.2168|14.1800|4.0142|
|weight_1|actual push (>2cm endpoint, contact)|40|unavailable|unavailable|unavailable|14.1800|unavailable|
|weight_5|all|160|2.4848|60.6554|5.8543|7.5656|2.4944|
|weight_5|no_contact|80|1.5667|29.2392|2.9632|3.5347|1.8188|
|weight_5|light_contact|40|2.2025|41.5586|4.2781|4.8817|2.2753|
|weight_5|clear_push|40|3.8648|106.2042|10.0613|13.4213|3.6187|
|weight_5|actual push (>2cm endpoint, contact)|40|unavailable|unavailable|unavailable|13.4213|unavailable|
|weight_20|all|160|2.2805|58.4363|5.7524|7.2540|2.0844|
|weight_20|no_contact|80|1.0358|26.2510|2.8001|3.1539|1.1214|
|weight_20|light_contact|40|1.5414|37.5489|4.0726|4.3979|1.3714|
|weight_20|clear_push|40|4.0350|104.2644|10.0047|13.0861|3.6032|
|weight_20|actual push (>2cm endpoint, contact)|40|unavailable|unavailable|unavailable|13.0861|unavailable|
|weight_100|all|160|2.3870|60.2083|6.0531|7.4595|1.8920|
|weight_100|no_contact|80|0.5196|22.9606|2.6890|2.7110|0.6369|
|weight_100|light_contact|40|0.8919|34.2411|3.9811|3.9753|0.8214|
|weight_100|clear_push|40|4.6321|110.7851|10.7818|13.8592|3.5823|
|weight_100|actual push (>2cm endpoint, contact)|40|unavailable|unavailable|unavailable|13.8592|unavailable|

The TF summed displacement is an error-accumulation diagnostic, not an autoregressive physical rollout.

|Model|Stationary cohort|Episodes|AR mean / p95 / max drift mm|AR >2mm %|TF next mean mm|TF summed mean mm|
|---|---|---:|---:|---:|---:|---:|
|initialization|all_stationary|111|84.3712 / 161.3259 / 231.4235|100.0000|6.6106|29.0683|
|initialization|stationary_no_contact|80|80.7564 / 159.2486 / 231.4235|100.0000|5.8737|27.2702|
|initialization|overhead|40|63.9418 / 111.3803 / 127.9390|100.0000|1.8023|20.6925|
|initialization|near_miss|40|97.5710 / 181.4552 / 231.4235|100.0000|9.9451|33.8478|
|weight_1|all_stationary|111|43.8224 / 116.6518 / 220.5374|100.0000|7.3575|23.1359|
|weight_1|stationary_no_contact|80|40.7172 / 102.4980 / 220.5374|100.0000|6.4547|21.2168|
|weight_1|overhead|40|25.5425 / 50.8046 / 68.9888|100.0000|1.6558|13.5741|
|weight_1|near_miss|40|55.8920 / 137.8742 / 220.5374|100.0000|11.2536|28.8595|
|weight_5|all_stationary|111|31.3060 / 88.0359 / 139.8036|100.0000|5.2201|16.4866|
|weight_5|stationary_no_contact|80|28.0665 / 85.5101 / 138.2683|100.0000|4.5473|15.3092|
|weight_5|overhead|40|14.0436 / 24.8559 / 26.7265|100.0000|1.0909|10.8542|
|weight_5|near_miss|40|42.0895 / 97.9839 / 138.2683|100.0000|8.0038|19.7642|
|weight_20|all_stationary|111|25.5796 / 71.0016 / 140.7950|99.0991|3.4290|10.8737|
|weight_20|stationary_no_contact|80|23.0134 / 66.6619 / 139.5159|98.7500|2.9512|10.1022|
|weight_20|overhead|40|10.6063 / 22.0070 / 31.2132|97.5000|0.6957|7.6759|
|weight_20|near_miss|40|35.4205 / 90.6069 / 139.5159|100.0000|5.2068|12.5284|
|weight_100|all_stationary|111|18.8301 / 52.1957 / 131.3050|97.2973|1.8274|6.0344|
|weight_100|stationary_no_contact|80|16.7104 / 51.6008 / 130.0223|96.2500|1.5389|5.5200|
|weight_100|overhead|40|5.5992 / 13.1216 / 22.1220|92.5000|0.3743|3.7014|
|weight_100|near_miss|40|27.8215 / 71.6482 / 130.0223|100.0000|2.7034|7.3387|

### Rollout regressions and initialization comparisons

weight_1 versus initialization:
- One-step stationary RMSE change: 0.2699 mm.
- One-step moving RMSE change: -2.0122 mm.
- One-step moving_contact RMSE change: -2.2466 mm.
- all 4s endpoint RMSE change: -1.9474 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -3.7568 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -3.8848 cm (positive is regression).
- clear_push 4s endpoint RMSE change: 0.8232 cm (positive is regression).
- all_stationary mean maximum drift change: -40.5488 mm.
- stationary_no_contact mean maximum drift change: -40.0392 mm.
- overhead mean maximum drift change: -38.3993 mm.
- near_miss mean maximum drift change: -41.6790 mm.

weight_5 versus initialization:
- One-step stationary RMSE change: -0.3816 mm.
- One-step moving RMSE change: -2.4035 mm.
- One-step moving_contact RMSE change: -2.6639 mm.
- all 4s endpoint RMSE change: -3.0222 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -5.3417 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -5.7210 cm (positive is regression).
- clear_push 4s endpoint RMSE change: 0.0645 cm (positive is regression).
- all_stationary mean maximum drift change: -53.0652 mm.
- stationary_no_contact mean maximum drift change: -52.6899 mm.
- overhead mean maximum drift change: -49.8982 mm.
- near_miss mean maximum drift change: -55.4815 mm.

weight_5 versus weight_1:
- One-step stationary RMSE change: -0.6516 mm.
- One-step moving RMSE change: -0.3913 mm.
- One-step moving_contact RMSE change: -0.4173 mm.
- all 4s endpoint RMSE change: -1.0748 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -1.5848 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -1.8361 cm (positive is regression).
- clear_push 4s endpoint RMSE change: -0.7588 cm (positive is regression).
- all_stationary mean maximum drift change: -12.5164 mm.
- stationary_no_contact mean maximum drift change: -12.6507 mm.
- overhead mean maximum drift change: -11.4989 mm.
- near_miss mean maximum drift change: -13.8025 mm.

weight_20 versus initialization:
- One-step stationary RMSE change: -0.8714 mm.
- One-step moving RMSE change: -2.1713 mm.
- One-step moving_contact RMSE change: -2.3378 mm.
- all 4s endpoint RMSE change: -3.3338 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -5.7225 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -6.2048 cm (positive is regression).
- clear_push 4s endpoint RMSE change: -0.2707 cm (positive is regression).
- all_stationary mean maximum drift change: -58.7916 mm.
- stationary_no_contact mean maximum drift change: -57.7430 mm.
- overhead mean maximum drift change: -53.3355 mm.
- near_miss mean maximum drift change: -62.1505 mm.

weight_20 versus weight_1:
- One-step stationary RMSE change: -1.1413 mm.
- One-step moving RMSE change: -0.1591 mm.
- One-step moving_contact RMSE change: -0.0912 mm.
- all 4s endpoint RMSE change: -1.3864 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -1.9657 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -2.3200 cm (positive is regression).
- clear_push 4s endpoint RMSE change: -1.0940 cm (positive is regression).
- all_stationary mean maximum drift change: -18.2428 mm.
- stationary_no_contact mean maximum drift change: -17.7038 mm.
- overhead mean maximum drift change: -14.9362 mm.
- near_miss mean maximum drift change: -20.4715 mm.

weight_100 versus initialization:
- One-step stationary RMSE change: -1.3996 mm.
- One-step moving RMSE change: -0.8732 mm.
- One-step moving_contact RMSE change: -0.6908 mm.
- all 4s endpoint RMSE change: -3.1282 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -6.1654 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -6.6273 cm (positive is regression).
- clear_push 4s endpoint RMSE change: 0.5024 cm (positive is regression).
- all_stationary mean maximum drift change: -65.5411 mm.
- stationary_no_contact mean maximum drift change: -64.0460 mm.
- overhead mean maximum drift change: -58.3426 mm.
- near_miss mean maximum drift change: -69.7495 mm.

weight_100 versus weight_1:
- One-step stationary RMSE change: -1.6696 mm.
- One-step moving RMSE change: 1.1391 mm.
- One-step moving_contact RMSE change: 1.5558 mm.
- all 4s endpoint RMSE change: -1.1808 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -2.4086 cm (positive is regression).
- light_contact 4s endpoint RMSE change: -2.7425 cm (positive is regression).
- clear_push 4s endpoint RMSE change: -0.3209 cm (positive is regression).
- all_stationary mean maximum drift change: -24.9923 mm.
- stationary_no_contact mean maximum drift change: -24.0069 mm.
- overhead mean maximum drift change: -19.9433 mm.
- near_miss mean maximum drift change: -28.0704 mm.

### Every rollout regression

All scalar rollout errors/drift measures are compared in comparisons.json, including improvements. Positive changes below are regressions; units follow the metric suffix (m or dimensionless fraction). These diagnostics do not enter the recommendation rule.

|Model|Reference|Metric|Increase|
|---|---|---|---:|
|weight_1|initialization|one_step_cases.no_contact.cube_displacement_rmse_m|0.0003053496|
|weight_1|initialization|one_step_cases.light_contact.cube_displacement_rmse_m|0.0004726661|
|weight_1|initialization|one_step_intervals.no_contact.cube_displacement_rmse_m|0.0002069989|
|weight_1|initialization|one_step_intervals.light_contact.cube_displacement_rmse_m|0.001202655|
|weight_1|initialization|stationary_drift.all_stationary.teacher_forced_next.mean_max_drift_m|0.000746873|
|weight_1|initialization|stationary_drift.all_stationary.teacher_forced_next.p95_max_drift_m|0.007491536|
|weight_1|initialization|stationary_drift.all_stationary.teacher_forced_next.worst_max_drift_m|0.0304806|
|weight_1|initialization|stationary_drift.stationary_no_contact.teacher_forced_next.mean_max_drift_m|0.0005809643|
|weight_1|initialization|stationary_drift.stationary_no_contact.teacher_forced_next.p95_max_drift_m|0.006532835|
|weight_1|initialization|stationary_drift.stationary_no_contact.teacher_forced_next.worst_max_drift_m|0.0304806|
|weight_1|initialization|stationary_drift.overhead.teacher_forced_next.p95_max_drift_m|0.001021056|
|weight_1|initialization|stationary_drift.near_miss.teacher_forced_next.mean_max_drift_m|0.00130843|
|weight_1|initialization|stationary_drift.near_miss.teacher_forced_next.p95_max_drift_m|0.01157212|
|weight_1|initialization|stationary_drift.near_miss.teacher_forced_next.worst_max_drift_m|0.0304806|
|weight_1|initialization|teacher_forced_vs_autoregressive.no_contact.teacher_forced_one_step_rmse_m|0.0003053496|
|weight_1|initialization|teacher_forced_vs_autoregressive.light_contact.teacher_forced_one_step_rmse_m|0.0004726661|
|weight_1|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_prefix_position_rmse_m|0.003902039|
|weight_1|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_4s_endpoint_rmse_m|0.008232445|
|weight_1|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.004682219|
|weight_1|initialization|push_4s.cube_endpoint_rmse_m|0.008232445|
|weight_1|initialization|clear_push_4s_rmse_m|0.008232445|
|weight_5|initialization|stationary_drift.all_stationary.teacher_forced_next.p95_max_drift_m|0.0008194996|
|weight_5|initialization|stationary_drift.all_stationary.teacher_forced_next.worst_max_drift_m|0.01679775|
|weight_5|initialization|stationary_drift.stationary_no_contact.teacher_forced_next.worst_max_drift_m|0.01679775|
|weight_5|initialization|stationary_drift.near_miss.teacher_forced_next.p95_max_drift_m|0.003641335|
|weight_5|initialization|stationary_drift.near_miss.teacher_forced_next.worst_max_drift_m|0.01679775|
|weight_5|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_prefix_position_rmse_m|0.00170241|
|weight_5|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_4s_endpoint_rmse_m|0.0006447277|
|weight_5|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.0007271841|
|weight_5|initialization|push_4s.cube_endpoint_rmse_m|0.0006447277|
|weight_5|initialization|clear_push_4s_rmse_m|0.0006447277|
|weight_20|initialization|stationary_drift.all_stationary.teacher_forced_next.worst_max_drift_m|0.001332112|
|weight_20|initialization|stationary_drift.stationary_no_contact.teacher_forced_next.worst_max_drift_m|0.0008809076|
|weight_20|initialization|stationary_drift.near_miss.teacher_forced_next.worst_max_drift_m|0.0008809076|
|weight_20|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.0005728285|
|weight_20|weight_1|one_step_cases.clear_push.cube_displacement_rmse_m|4.491951e-05|
|weight_20|weight_1|one_step_intervals.active_push.cube_displacement_rmse_m|0.0003216055|
|weight_20|weight_1|teacher_forced_vs_autoregressive.clear_push.teacher_forced_one_step_rmse_m|4.491951e-05|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_prefix_position_rmse_m|0.006283233|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_increment_rmse_m|0.0002410203|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_4s_endpoint_rmse_m|0.005023812|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.0003631078|
|weight_100|initialization|push_4s.cube_endpoint_rmse_m|0.005023812|
|weight_100|initialization|clear_push_4s_rmse_m|0.005023812|
|weight_100|weight_1|one_step_cases.clear_push.cube_displacement_rmse_m|0.000641997|
|weight_100|weight_1|one_step_intervals.active_push.cube_displacement_rmse_m|0.002228979|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.teacher_forced_one_step_rmse_m|0.000641997|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.autoregressive_prefix_position_rmse_m|0.002381194|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.autoregressive_increment_rmse_m|0.0005650897|
## broad

|Model|Stationary mm|Moving mm|Moving contact mm|Clear push mm|False 0.5mm %|False 2mm %|Missed 2mm %|4s endpoint cm|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|initialization|2.8747|10.7133|10.8967|6.1509|78.8547|18.5196|1.4541|19.7858|
|weight_1|2.8593|8.4990|7.7657|5.1113|77.0670|13.0726|0.8949|18.0129|
|weight_5|2.1220|8.7597|8.2623|4.9653|54.3296|5.8101|1.5660|14.6887|
|weight_20|1.4701|9.4271|9.0029|5.1717|27.9609|2.1788|4.0268|13.5853|
|weight_100|1.1744|11.0433|10.8193|5.9380|10.2235|0.7263|10.0671|19.1881|

Full metrics and predictions are saved separately per model and dataset. Teacher-forced XY errors are displacement errors; no prediction is fed back during training. Contact-motion direction requires >2 mm actual displacement and is measured against the actual displacement vector. Zero-length predictions have undefined angle and fail the joint magnitude/direction accuracy measure.

### Cohorts, case errors, and interval errors

|Model|Stationary n|Moving n|Moving contact n|Light case n / RMSE mm|Clear case n / RMSE mm|No-contact interval n / mm|Light interval n / mm|Active-push interval n / mm|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|initialization|3580|1220|1073|0 / unavailable|3640 / 6.1509|3666 / 3.3715|140 / 11.1060|894 / 11.8183|
|weight_1|3580|1220|1073|0 / unavailable|3640 / 5.1113|3666 / 3.7670|140 / 8.6390|894 / 8.2920|
|weight_5|3580|1220|1073|0 / unavailable|3640 / 4.9653|3666 / 3.1401|140 / 8.8058|894 / 8.9148|
|weight_20|3580|1220|1073|0 / unavailable|3640 / 5.1717|3666 / 2.8143|140 / 9.0280|894 / 9.7860|
|weight_100|3580|1220|1073|0 / unavailable|3640 / 5.9380|3666 / 2.7674|140 / 9.7859|894 / 11.8225|

Boundary case labels are saved collector kinds. Broad light contact requires >0 and <=0.2s contact and <=2cm excursion; broad clear push requires >=0.1s contact and >=5cm excursion. Interval light contact means 1..10 active substeps out of 50; active push means contact and >2mm net motion. These interval cohorts may overlap. Whole-episode rollout stationarity uses <=2mm excursion from the episode's initial position, distinct from the six-sample transition label.

### False and missed motion, including evaluation-only zero XY

|Model|Threshold mm|Stationary n|False count / %|Contact-motion n|Missed count / %|
|---|---:|---:|---:|---:|---:|
|initialization|0.5|3580|2823 / 78.8547|1038|4 / 0.3854|
|initialization|2|3580|663 / 18.5196|894|13 / 1.4541|
|weight_1|0.5|3580|2759 / 77.0670|1038|0 / 0.0000|
|weight_1|2|3580|468 / 13.0726|894|8 / 0.8949|
|weight_5|0.5|3580|1945 / 54.3296|1038|4 / 0.3854|
|weight_5|2|3580|208 / 5.8101|894|14 / 1.5660|
|weight_20|0.5|3580|1001 / 27.9609|1038|11 / 1.0597|
|weight_20|2|3580|78 / 2.1788|894|36 / 4.0268|
|weight_100|0.5|3580|366 / 10.2235|1038|36 / 3.4682|
|weight_100|2|3580|26 / 0.7263|894|90 / 10.0671|
|zero_xy_reference|0.5|3580|0 / 0.0000|1038|1038 / 100.0000|
|zero_xy_reference|2|3580|0 / 0.0000|894|894 / 100.0000|

### Predicted displacement magnitudes and contact direction

|Model|Stationary predicted mean / median / p95 / max mm|Contact n|Predicted / actual mean mm|Magnitude RMSE mm|Direction n / mean angle deg|Wrong direction %|Within 30deg and >2mm %|
|---|---:|---:|---:|---:|---:|---:|---:|
|initialization|1.4700 / 0.9370 / 4.2595 / 91.3191|894|18.5699 / 21.2076|11.4083|894 / 7.4415|0.4474|95.0783|
|weight_1|1.3421 / 0.9141 / 3.3901 / 85.1683|894|20.2966 / 21.2076|7.6742|894 / 8.0515|0.2237|95.7494|
|weight_5|0.8460 / 0.5347 / 2.1453 / 78.3435|894|19.6479 / 21.2076|8.3204|894 / 7.5334|0.3356|95.6376|
|weight_20|0.5159 / 0.3087 / 1.3261 / 55.9333|894|18.7461 / 21.2076|9.2959|894 / 7.5327|0.6711|93.6242|
|weight_100|0.2882 / 0.1794 / 0.7039 / 57.2088|894|17.1080 / 21.2076|11.4249|894 / 8.9297|1.4541|87.4720|
|zero_xy_reference|0.0000 / 0.0000 / 0.0000 / 0.0000|894|0.0000 / 21.2076|29.0431|0 / unavailable|0.0000|0.0000|

Wrong-direction rate divides opposite predictions by all contact-motion samples. Zero predictions have undefined direction; the joint >2mm/30deg measure counts them as failures. The zero-XY reference is never part of inference or deployment.

### Four-second diagnostics

|Model|Cohort|Episodes|TF one-step mm|AR prefix mm|AR increment mm|AR 4s endpoint cm|TF summed 4s cm|
|---|---|---:|---:|---:|---:|---:|---:|
|initialization|all|120|5.9443|129.8111|11.8972|19.7858|4.8943|
|initialization|no_contact|25|3.3392|50.0449|3.9285|7.7235|4.1118|
|initialization|light_contact|0|unavailable|unavailable|unavailable|unavailable|unavailable|
|initialization|clear_push|91|6.1509|136.5036|12.7069|21.1661|4.7973|
|initialization|actual push (>2cm endpoint, contact)|95|unavailable|unavailable|unavailable|21.8815|unavailable|
|weight_1|all|120|4.9454|116.0023|10.9951|18.0129|5.0647|
|weight_1|no_contact|25|3.4471|45.4478|3.6255|6.6610|4.4415|
|weight_1|light_contact|0|unavailable|unavailable|unavailable|unavailable|unavailable|
|weight_1|clear_push|91|5.1113|126.3640|12.0645|19.7920|5.0593|
|weight_1|actual push (>2cm endpoint, contact)|95|unavailable|unavailable|unavailable|19.9543|unavailable|
|weight_5|all|120|4.7813|99.9585|10.1127|14.6887|4.2613|
|weight_5|no_contact|25|2.8826|39.3359|3.4774|5.2861|3.0615|
|weight_5|light_contact|0|unavailable|unavailable|unavailable|unavailable|unavailable|
|weight_5|clear_push|91|4.9653|107.7022|10.9961|15.9529|4.2882|
|weight_5|actual push (>2cm endpoint, contact)|95|unavailable|unavailable|unavailable|16.2844|unavailable|
|weight_20|all|120|4.9193|90.4504|9.8795|13.5853|4.1354|
|weight_20|no_contact|25|2.1059|25.4041|2.8505|3.3873|2.0550|
|weight_20|light_contact|0|unavailable|unavailable|unavailable|unavailable|unavailable|
|weight_20|clear_push|91|5.1717|98.1678|10.7259|14.9632|4.2364|
|weight_20|actual push (>2cm endpoint, contact)|95|unavailable|unavailable|unavailable|15.1694|unavailable|
|weight_100|all|120|5.6591|111.1047|12.4736|19.1881|5.0231|
|weight_100|no_contact|25|2.0182|24.7681|3.0521|3.2670|1.7823|
|weight_100|light_contact|0|unavailable|unavailable|unavailable|unavailable|unavailable|
|weight_100|clear_push|91|5.9380|122.4731|13.7727|21.5748|5.2166|
|weight_100|actual push (>2cm endpoint, contact)|95|unavailable|unavailable|unavailable|21.5003|unavailable|

The TF summed displacement is an error-accumulation diagnostic, not an autoregressive physical rollout.

|Model|Stationary cohort|Episodes|AR mean / p95 / max drift mm|AR >2mm %|TF next mean mm|TF summed mean mm|
|---|---|---:|---:|---:|---:|---:|
|initialization|all_stationary|25|68.4426 / 155.1733 / 166.0766|100.0000|8.4878|32.9744|
|initialization|stationary_no_contact|25|68.4426 / 155.1733 / 166.0766|100.0000|8.4878|32.9744|
|weight_1|all_stationary|25|52.2833 / 82.5682 / 233.0387|100.0000|7.6046|33.1190|
|weight_1|stationary_no_contact|25|52.2833 / 82.5682 / 233.0387|100.0000|7.6046|33.1190|
|weight_5|all_stationary|25|33.8382 / 61.6184 / 233.6536|100.0000|5.9915|20.6823|
|weight_5|stationary_no_contact|25|33.8382 / 61.6184 / 233.6536|100.0000|5.9915|20.6823|
|weight_20|all_stationary|25|20.2540 / 40.4125 / 151.4025|100.0000|4.0167|12.3193|
|weight_20|stationary_no_contact|25|20.2540 / 40.4125 / 151.4025|100.0000|4.0167|12.3193|
|weight_100|all_stationary|25|18.6776 / 25.2284 / 147.9253|96.0000|3.4334|8.4875|
|weight_100|stationary_no_contact|25|18.6776 / 25.2284 / 147.9253|96.0000|3.4334|8.4875|

### Rollout regressions and initialization comparisons

weight_1 versus initialization:
- One-step stationary RMSE change: -0.0154 mm.
- One-step moving RMSE change: -2.2143 mm.
- One-step moving_contact RMSE change: -3.1310 mm.
- all 4s endpoint RMSE change: -1.7729 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -1.0626 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: -1.3741 cm (positive is regression).
- all_stationary mean maximum drift change: -16.1593 mm.
- stationary_no_contact mean maximum drift change: -16.1593 mm.

weight_5 versus initialization:
- One-step stationary RMSE change: -0.7527 mm.
- One-step moving RMSE change: -1.9536 mm.
- One-step moving_contact RMSE change: -2.6344 mm.
- all 4s endpoint RMSE change: -5.0971 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -2.4374 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: -5.2131 cm (positive is regression).
- all_stationary mean maximum drift change: -34.6044 mm.
- stationary_no_contact mean maximum drift change: -34.6044 mm.

weight_5 versus weight_1:
- One-step stationary RMSE change: -0.7373 mm.
- One-step moving RMSE change: 0.2607 mm.
- One-step moving_contact RMSE change: 0.4966 mm.
- all 4s endpoint RMSE change: -3.3243 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -1.3748 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: -3.8390 cm (positive is regression).
- all_stationary mean maximum drift change: -18.4451 mm.
- stationary_no_contact mean maximum drift change: -18.4451 mm.

weight_20 versus initialization:
- One-step stationary RMSE change: -1.4045 mm.
- One-step moving RMSE change: -1.2862 mm.
- One-step moving_contact RMSE change: -1.8938 mm.
- all 4s endpoint RMSE change: -6.2005 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -4.3362 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: -6.2028 cm (positive is regression).
- all_stationary mean maximum drift change: -48.1886 mm.
- stationary_no_contact mean maximum drift change: -48.1886 mm.

weight_20 versus weight_1:
- One-step stationary RMSE change: -1.3892 mm.
- One-step moving RMSE change: 0.9281 mm.
- One-step moving_contact RMSE change: 1.2372 mm.
- all 4s endpoint RMSE change: -4.4276 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -3.2737 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: -4.8287 cm (positive is regression).
- all_stationary mean maximum drift change: -32.0293 mm.
- stationary_no_contact mean maximum drift change: -32.0293 mm.

weight_100 versus initialization:
- One-step stationary RMSE change: -1.7003 mm.
- One-step moving RMSE change: 0.3301 mm.
- One-step moving_contact RMSE change: -0.0774 mm.
- all 4s endpoint RMSE change: -0.5978 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -4.4565 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: 0.4087 cm (positive is regression).
- all_stationary mean maximum drift change: -49.7650 mm.
- stationary_no_contact mean maximum drift change: -49.7650 mm.

weight_100 versus weight_1:
- One-step stationary RMSE change: -1.6849 mm.
- One-step moving RMSE change: 2.5443 mm.
- One-step moving_contact RMSE change: 3.0536 mm.
- all 4s endpoint RMSE change: 1.1751 cm (positive is regression).
- no_contact 4s endpoint RMSE change: -3.3939 cm (positive is regression).
- light_contact 4s endpoint RMSE change: unavailable cm (positive is regression).
- clear_push 4s endpoint RMSE change: 1.7828 cm (positive is regression).
- all_stationary mean maximum drift change: -33.6057 mm.
- stationary_no_contact mean maximum drift change: -33.6057 mm.

### Every rollout regression

All scalar rollout errors/drift measures are compared in comparisons.json, including improvements. Positive changes below are regressions; units follow the metric suffix (m or dimensionless fraction). These diagnostics do not enter the recommendation rule.

|Model|Reference|Metric|Increase|
|---|---|---|---:|
|weight_1|initialization|one_step_cases.no_contact.cube_displacement_rmse_m|0.0001079397|
|weight_1|initialization|one_step_intervals.no_contact.cube_displacement_rmse_m|0.0003955106|
|weight_1|initialization|stationary_drift.all_stationary.autoregressive.worst_max_drift_m|0.06696207|
|weight_1|initialization|stationary_drift.all_stationary.teacher_forced_accumulated_diagnostic.mean_max_drift_m|0.0001446104|
|weight_1|initialization|stationary_drift.all_stationary.teacher_forced_accumulated_diagnostic.worst_max_drift_m|0.04774174|
|weight_1|initialization|stationary_drift.stationary_no_contact.autoregressive.worst_max_drift_m|0.06696207|
|weight_1|initialization|stationary_drift.stationary_no_contact.teacher_forced_accumulated_diagnostic.mean_max_drift_m|0.0001446104|
|weight_1|initialization|stationary_drift.stationary_no_contact.teacher_forced_accumulated_diagnostic.worst_max_drift_m|0.04774174|
|weight_1|initialization|teacher_forced_vs_autoregressive.all.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.001704526|
|weight_1|initialization|teacher_forced_vs_autoregressive.no_contact.teacher_forced_one_step_rmse_m|0.0001079397|
|weight_1|initialization|teacher_forced_vs_autoregressive.no_contact.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.003296659|
|weight_1|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.002620936|
|weight_5|initialization|stationary_drift.all_stationary.autoregressive.worst_max_drift_m|0.06757701|
|weight_5|initialization|stationary_drift.all_stationary.teacher_forced_accumulated_diagnostic.worst_max_drift_m|0.01472866|
|weight_5|initialization|stationary_drift.stationary_no_contact.autoregressive.worst_max_drift_m|0.06757701|
|weight_5|initialization|stationary_drift.stationary_no_contact.teacher_forced_accumulated_diagnostic.worst_max_drift_m|0.01472866|
|weight_5|weight_1|one_step_intervals.light_contact.cube_displacement_rmse_m|0.0001667745|
|weight_5|weight_1|one_step_intervals.active_push.cube_displacement_rmse_m|0.0006228493|
|weight_5|weight_1|stationary_drift.all_stationary.autoregressive.worst_max_drift_m|0.000614945|
|weight_5|weight_1|stationary_drift.stationary_no_contact.autoregressive.worst_max_drift_m|0.000614945|
|weight_20|weight_1|one_step_cases.clear_push.cube_displacement_rmse_m|6.033393e-05|
|weight_20|weight_1|one_step_intervals.light_contact.cube_displacement_rmse_m|0.0003890379|
|weight_20|weight_1|one_step_intervals.active_push.cube_displacement_rmse_m|0.001494063|
|weight_20|weight_1|teacher_forced_vs_autoregressive.clear_push.teacher_forced_one_step_rmse_m|6.033393e-05|
|weight_100|initialization|one_step_intervals.active_push.cube_displacement_rmse_m|4.186736e-06|
|weight_100|initialization|teacher_forced_vs_autoregressive.all.autoregressive_increment_rmse_m|0.0005763058|
|weight_100|initialization|teacher_forced_vs_autoregressive.all.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.001288436|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_increment_rmse_m|0.001065776|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.autoregressive_4s_endpoint_rmse_m|0.004087095|
|weight_100|initialization|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.004193923|
|weight_100|initialization|clear_push_4s_rmse_m|0.004087095|
|weight_100|weight_1|one_step_overall_rmse_m|0.0007137163|
|weight_100|weight_1|one_step_cases.clear_push.cube_displacement_rmse_m|0.00082671|
|weight_100|weight_1|one_step_intervals.light_contact.cube_displacement_rmse_m|0.001146941|
|weight_100|weight_1|one_step_intervals.active_push.cube_displacement_rmse_m|0.003530545|
|weight_100|weight_1|teacher_forced_vs_autoregressive.all.teacher_forced_one_step_rmse_m|0.0007137163|
|weight_100|weight_1|teacher_forced_vs_autoregressive.all.autoregressive_increment_rmse_m|0.001478493|
|weight_100|weight_1|teacher_forced_vs_autoregressive.all.autoregressive_4s_endpoint_rmse_m|0.01175128|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.teacher_forced_one_step_rmse_m|0.00082671|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.autoregressive_increment_rmse_m|0.001708151|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.autoregressive_4s_endpoint_rmse_m|0.01782838|
|weight_100|weight_1|teacher_forced_vs_autoregressive.clear_push.teacher_forced_accumulated_4s_diagnostic_rmse_m|0.001572987|
|weight_100|weight_1|push_4s.cube_endpoint_rmse_m|0.01545961|
|weight_100|weight_1|clear_push_4s_rmse_m|0.01782838|
## Eligibility

- weight_5: eligible.
- weight_20: broad: moving_contact RMSE regression 15.93% exceeds 10%.
- weight_100: boundary: moving_contact RMSE regression 16.98% exceeds 10%; boundary: clear_push RMSE regression 16.09% exceeds 10%; boundary: missed motion increase 9.09pp exceeds 5pp; broad: moving_contact RMSE regression 39.32% exceeds 10%; broad: clear_push RMSE regression 16.17% exceeds 10%; broad: missed motion increase 9.17pp exceeds 5pp.

## Verification

Existing datasets/checkpoints unchanged: True. Identical batch-order hashes: True. Normalization and saved evaluation states/actions checked for every model. See config.json and summary.json for SHA-256 evidence.

- weight_1: 3264 updates; final epoch 12; diagnostic best epoch 8; 10.20 seconds.
- weight_5: 3264 updates; final epoch 12; diagnostic best epoch 11; 9.94 seconds.
- weight_20: 3264 updates; final epoch 12; diagnostic best epoch 11; 10.45 seconds.
- weight_100: 3264 updates; final epoch 12; diagnostic best epoch 11; 10.06 seconds.

54 relevant tests passed; no collection fixtures were run. The training tests forbid MuJoCo stepping and recursive prediction. Additional saved-artifact audits passed; see [verification.json](runs/arm_cube_one_step_stationary/verification.json).

## Artifacts

- [Full summary and metrics](runs/arm_cube_one_step_stationary/summary.json)
- [Training configuration and source hashes](runs/arm_cube_one_step_stationary/config.json)
- [All diagnostic changes against initialization and 1x](runs/arm_cube_one_step_stationary/comparisons.json)
- [Training learning curves](runs/arm_cube_one_step_stationary/train_learning_curves.png)
- [Validation learning curves](runs/arm_cube_one_step_stationary/validation_learning_curves.png)
- [Boundary four-second errors](runs/arm_cube_one_step_stationary/boundary_four_second_errors.png)
- [Broad four-second errors](runs/arm_cube_one_step_stationary/broad_four_second_errors.png)
- [Boundary false/missed motion at 2mm](runs/arm_cube_one_step_stationary/boundary_tradeoff_0.002.png)
- [Broad false/missed motion at 2mm](runs/arm_cube_one_step_stationary/broad_tradeoff_0.002.png)
