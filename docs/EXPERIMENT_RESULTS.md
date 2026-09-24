# Kết quả và cách diễn giải

## Cập nhật spacing 1 mm — 2026-09-23

Mục tiêu là kiểm tra spacing, nên mọi nhánh dùng cùng backbone U-Net 3D tự viết, patch `64×128×128`, loss, optimizer, threshold và seed. Đây không phải so sánh backbone với nnU-Net hoặc MedNeXt.

Pilot nhanh trên 100 ca/10 epoch/20 ca validation:

| Biến thể | Mean Dice | Mean HD95 (mm) | Mean precision |
| --- | ---: | ---: | ---: |
| P0 baseline | 0,0220 | 190,13 | 0,0118 |
| Spacing 1 mm | **0,0356** | **181,50** | **0,0192** |
| Brain-mask normalization | 0,0252 | 190,44 | 0,0135 |
| Intensity augmentation | 0,0251 | 190,21 | 0,0134 |

Full fold 1, seed 42:

| Nhánh | Inner Dice | Outer Dice | Surface Dice 2 mm | Precision |
| --- | ---: | ---: | ---: | ---: |
| P0 median spacing | 0,7158 | 0,6719 | 0,6811 | 0,6270 |
| Spacing 1 mm | 0,7294 | 0,6735 | 0,6834 | 0,6375 |

Full fold 2, seed 42 spacing 1 mm đạt outer Dice `0,6723`, Surface Dice `0,6632`, sensitivity `0,8079`, precision `0,6255`. P0 fold 2 (`21859`) vẫn đang chạy nên chưa thể kết luận spacing tốt hơn trên fold này.

Job `21363` bị đánh dấu failed ở bước validator chung vì validator kiểm tra membership của cả bốn profile P0--P3 trong thư mục spacing riêng. Run preprocessing thực tế ghi đủ `570/570` ca và `0 failure record`; lỗi validator không phải lỗi xử lý ảnh. Cần dùng validator profile-aware trước khi tái sử dụng job này làm gate.

## Pilot code mới P0--P3

Pilot mạnh hơn hiện có dùng 100 common labeled cases, split seed 42 thành 80 train/20 validation, 10 epochs/profile, cùng model và hyperparameter:

| Profile | Mean Dice | Mean HD95 (mm) |
| --- | ---: | ---: |
| P0 median + z-score | 0.02196 | 190.13 |
| P1 median + clip + z-score | **0.02877** | **186.24** |
| P2 median + N4 + z-score | 0.02511 | 186.86 |
| P3 median + N4 + clip + z-score | 0.01971 | 187.79 |

P1 thắng 11/20 validation cases và là lựa chọn tốt nhất trong pilot này. Tuy nhiên budget 10 epochs quá nhỏ để kết luận preprocessing nào tốt nhất. Validation pilot không chứa 0402, nên việc bổ sung 0402 không làm thay đổi các số trên.

## Baseline cũ

nnU-Net cũ đạt mean cross-validation Dice khoảng `0.773835` sau 1000 epochs/5 folds. Không được đặt số này cạnh pilot 10 epochs để kết luận pipeline cũ tốt hơn: model, training budget, folds và preprocessing đều khác. Ngoài ra split cũ bị duplicate 0402 như mô tả trong [`INCIDENT_0402.md`](INCIDENT_0402.md).

## Preprocess sửa 0402

Remote Slurm job `20932` xử lý P0--P3 thành công, không có failure record. Input image hash xác nhận dùng standalone patch. Output image SHA-256:

| Profile | SHA-256 |
| --- | --- |
| P0 | `990d5f1dc64261c0f87a8dabdf970b1e62934bdaa7b12e505dd6c88c8dcd673d` |
| P1 | `d451a1fcd2f8ffd44c038aa86e64e77929ee666239856cdfe827b731911b0ef6` |
| P2 | `1692d824cade207c3c74014c47ce5d65fcb0f57fbef546777b0b6d35b4c6c2fb` |
| P3 | `075928adfc2ab96170b3eed309e9c87d64f744a134dbb0215fa355ed6407b98c` |

## Kết luận được phép ở thời điểm này

- Dữ liệu 0402 đã được sửa đúng về provenance và đủ trong P0--P3.
- P1 là profile ưu tiên cho smoke test/benchmark kế tiếp.
- Chưa có so sánh khoa học công bằng cũ--mới. Bước cần thiết là cùng model, cùng case/split, cùng epoch/seed và chỉ thay preprocessing.

## Smoke test cũ--mới trên corrected 0402

Slurm job `20935` hoàn tất trong 20 giây, exit code `0`. Report: `experiments/preprocessing_compare_0402/run_20935/comparison.json`.

| Thuộc tính | Raw | Legacy | New P1 |
| --- | ---: | ---: | ---: |
| Grid size | 352×352×200 | 352×352×200 | 256×256×146 |
| Spacing (mm) | 0.6818×0.6818×0.8000 | native | 0.9375×0.9375×1.1000 |
| Voxel image khác 0 | 16,690,299 | **24,780,800 (toàn FOV)** | 4,939,327 |
| Label volume (mm³) | 17,009.95 | 17,009.95 | 16,987.68 |
| Label components | 2 (45,737 + 1 voxel) | 2 | 1 |
| Finite fraction | 1.0 | 1.0 | 1.0 |

Cả hai output đều khớp geometry image--label và dùng đúng input hash patch. Legacy bảo toàn label/native grid nhưng làm background thành nonzero do lỗi clipping; new P1 có label-volume delta khoảng `-0.13094%` và mất isolated component 1 voxel khi resample. Vì grid khác nhau, đây là kiểm tra correctness/QC chứ chưa phải so sánh chất lượng segmentation.

## Smoke5 sau khi sửa legacy

Job `20948` chạy 5 case đại diện và hoàn tất trong 2 phút 27 giây, exit `0`. Corrected legacy đã hết lỗi background ở cả 5 case và bảo toàn label volume tuyệt đối. New P1 pass giới hạn label-volume ±10%:

| Case | Lý do chọn | New P1 label-volume delta |
| --- | --- | ---: |
| 0274 | tumor lớn nhất | +0.3063% |
| 0402 | standalone patch | -0.1309% |
| 0454 | median volume | +0.5146% |
| 0459 | tumor nhỏ nhất | -3.8622% |
| 0596 | 22 components | -0.0759% |

0402 vẫn có warning component 2→1 do isolated voxel. Không case nào fail finite, geometry, background hoặc volume gate.

## Full rerun và protocol kế tiếp — 2026-09-09

Full new preprocessing job `20950` đã hoàn tất exit `0` trong 11 giờ 54 phút 45 giây: 570 case được xử lý cho cả P0--P3, không có failure record. Validator `20962` hoàn tất exit `0`, `valid: true`; mỗi profile có 500 train images, 500 train labels và 70 validation images, không duplicate/overlap/missing/unexpected case.

Chưa có aggregate hoàn chỉnh mới sau full rerun. Experiment đã chốt là năm nhánh `legacy_corrected` + P0--P3, corrected 5-fold, ba seed. Legacy-corrected được dùng để so sánh end-to-end; P0--P3 tạo thành ablation 2×2 clipping/N4. Prediction outer-test được resample về raw canonical geometry trước khi tính metric. Smoke `21002` đã pass; array chính `21003` bị hủy giữa chừng và còn thiếu task, nên chưa được aggregate. Xem protocol và lệnh tại [`ABLATION_PROTOCOL.md`](ABLATION_PROTOCOL.md).
