# Rerun readiness — cập nhật 2026-09-23

## Trạng thái hiện tại

| Gate | Legacy corrected | New P0--P3 |
| --- | --- | --- |
| Raw canonical 500 train + 70 validation | PASS | PASS |
| Patch 0402 đúng SHA-256 | PASS | PASS |
| Duplicate/overlap/missing path | PASS | PASS |
| Unit/regression tests | PASS, 10/10 | Preflight PASS |
| Dry-run toàn 570 cases | PASS | PASS |
| Slurm `--test-only` | PASS | PASS |
| Smoke processing | PASS, 5/5 | PASS P1, 5/5 |
| Full preprocessing | PASS, job `20949` | PASS, job `20950`, 570 cases × 4 profiles, 0 failure |
| Full output validation | PASS: 570 images/500 labels | PASS, job `20962`, `valid: true` |
| Training | Nhánh `legacy_corrected` trong array `21003` | Array `21003` bị hủy; pilot spacing 1 mm đang được đối chiếu với P0 |

Remote raw source đã transfer đủ 13,054,290,063 bytes và rsync dry-run không còn delta. Canonical view remote có 570 relative symlinks; bản 0402 materialized trước migration được giữ ở `experiments/canonical_migration_backup/materialized_0402_before_symlink`.

## Những gì đã sửa ở legacy

- Clip/z-score chỉ áp dụng trong original nonzero support; background 0 được bảo toàn.
- Crop bounds lấy từ raw oriented support, không lấy từ ảnh đã normalize.
- Crop affine dịch origin theo crop start.
- Manifest duplicate làm pipeline fail trước processing.
- Output tồn tại bị từ chối nếu không có `--overwrite` rõ ràng.
- Manifest, config, data root, output root và artifact root đều configurable.
- Split fail nếu duplicate, train/QC mismatch, fold overlap hoặc một case không được validation đúng một lần.
- Corrected split: 500 unique cases; fold sizes 102/101/99/99/99; 0402 chỉ ở fold 3.

Legacy cũ vẫn được giữ nguyên dưới artifact lịch sử. “Legacy corrected” là một run mới, không sửa số liệu cũ.

## Smoke5 evidence

Job `20948` chọn năm case: tumor nhỏ nhất, median, lớn nhất, case 22 components và patched 0402. Kết quả:

- legacy corrected: 5/5 finite, geometry match, background 0 được giữ, label volume delta 0%;
- new P1: 5/5 finite, geometry match, label-volume delta trong ±10%;
- warning duy nhất: 0402 từ 2 xuống 1 component do isolated component 1 voxel mất khi resample;
- report: `experiments/preprocessing_compare_smoke5/run_20948/validation.json`.

## Job và lệnh theo dõi

```bash
sacct -j 20949,20950,20962 --format=JobID,JobName%28,State,ExitCode,Elapsed -P
tail -f /mnt/beegfs/ydang/BRATS/experiments/unet/logs/legacy_corrected_full_20949.out
tail -f /mnt/beegfs/ydang/BRATS/experiments/unet/logs/preprocess_full_20950.out
```

Job `20949` ghi vào `experiments/legacy_corrected/run_20949/` và hoàn tất trong 58 phút 21 giây, exit `0`. Validator xác nhận 570 images, 500 labels, không duplicate/missing; corrected split và nnU-Net raw/export đã được tạo. Job `20950` ghi vào `experiments/preprocessing_full_canonical/run_20950/`, hoàn tất trong 11 giờ 54 phút 45 giây, exit `0`, xử lý đủ 570 case cho P0--P3 với 0 failure record. Cả hai không cài hoặc ghi đè dataset đang dùng.

Validator job `20962` đã chạy sau `20950`, hoàn tất exit `0` và ghi `valid: true`: mỗi profile có 500 train images, 500 labels, 70 validation images; không duplicate, overlap, missing hoặc unexpected case.

## Điều kiện mở training

Không submit training chỉ vì preprocessing job có exit code 0. Cần:

1. Legacy `validation.json` có `valid: true`, đúng 570 images/500 labels.
2. New run có 570 success records cho mỗi P0--P3, không failure record.
3. QC finite/geometry/label classes pass và warning volume/components được audit.
4. Corrected split được cài vào đúng run root, không dùng split legacy cũ.
5. Chốt protocol so sánh: cùng case IDs, split, model, epoch, seed và evaluation. nnU-Net legacy và U-Net mới không được so trực tiếp như một preprocessing ablation.

Toàn bộ data gate của run `20949`/`20950`/`20962` đã pass. Array `21003` chưa hoàn tất nên không được aggregate. Spacing 1 mm đã có kết quả đầy đủ ở fold 1 và fold 2 seed 42; P0 fold 2 đang chạy để tạo cặp so sánh cùng fold. Xem các số mới tại [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) và [`MEN_RT_PREPROCESS_REVIEW.md`](MEN_RT_PREPROCESS_REVIEW.md).
