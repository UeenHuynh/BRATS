# BraTS-MEN-RT workspace

> Cập nhật trạng thái gần nhất: **2026-09-23**. Canonical raw và P0--P3 đã pass các data gate; spacing 1 mm đang được đối chiếu với P0 trên fold 2, seed 42.

Đây là workspace chứa **hai pipeline khác nhau**:

- `src/brats_men_rt/`, `scripts/`, `configs/`: pipeline nnU-Net cũ và các artifact cũ.
- `code/brats-research/`: code mới từ `UeenHuynh/BRATS`, dùng các profile P0--P3.

Không chạy pipeline chỉ dựa vào README nằm trong một thư mục con. Hãy đọc theo thứ tự sau:

1. [`README_WORKSPACE.md`](README_WORKSPACE.md): thư mục nào là nguồn chuẩn, thư mục nào chỉ là lịch sử.
2. [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md): dữ liệu đến từ đâu và quy tắc thay thế case 0402.
3. [`docs/INCIDENT_0402.md`](docs/INCIDENT_0402.md): lỗi trùng/ghi đè đã xảy ra trong pipeline cũ và ảnh hưởng tới split.
4. [`docs/VALIDATION_AND_RERUN.md`](docs/VALIDATION_AND_RERUN.md): luồng kiểm tra bắt buộc trước khi chạy lại.
5. [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md): kết quả hiện có và giới hạn khi so sánh cũ--mới.
6. [`docs/RERUN_READINESS.md`](docs/RERUN_READINESS.md): trạng thái go/no-go và job đang chạy.
7. [`code/brats-research/README.md`](code/brats-research/README.md): cách chạy riêng code mới sau khi các data gate ở bước 4 đã pass.
8. [`docs/ABLATION_PROTOCOL.md`](docs/ABLATION_PROTOCOL.md): protocol 5 nhánh và lệnh tự submit/kiểm tra.

Tài liệu cũ được phân loại tại [`planning/README.md`](planning/README.md) và [`reports/README.md`](reports/README.md). Không đọc một report legacy riêng lẻ mà bỏ qua cảnh báo trạng thái.

## Trạng thái hiện tại

- Raw canonical: 500 train + 70 validation, không trùng case ID.
- `BraTS-MEN-RT-0402-1` dùng standalone patch, không dùng ảnh lỗi trong training v2.
- 0402 đã được preprocess lại thành công cho P0--P3 và cài vào đủ bốn profile.
- Kiểm tra P0--P3: mỗi profile có 500 train images, 500 train labels và 70 validation images.
- Smoke test cũ--mới trên corrected 0402 đã pass; đồng thời xác nhận lỗi background/crop trong preprocessing cũ.
- Regression tests legacy pass 10/10; corrected split có 500 validation assignments duy nhất.
- Smoke5 corrected legacy--new P1 pass 5/5 ở job `20948`.
- Legacy-corrected full `20949`: COMPLETED, validator pass 570 images/500 labels.
- New P0--P3 full `20950`: COMPLETED, exit `0`, 570/570 case và 0 failure record.
- Validator `20962`: COMPLETED, `valid: true`; mỗi P0--P3 đủ 500 train images, 500 labels và 70 validation images.
- Protocol ablation 5 nhánh × 5 folds × 3 seeds: smoke `21002` pass; array chính đã bị hủy giữa chừng và còn thiếu nhiều task, chưa được aggregate.
- Pilot 100 ca/10 epoch: spacing 1 mm đạt Dice validation `0,0356`, P0 `0,0220`; brain-mask normalization `0,0252`; intensity augmentation `0,0251`.
- Spacing 1 mm full fold 1/seed 42: inner Dice `0,7294` so với P0 `0,7158`; outer Dice `0,6735` so với P0 `0,6719`. Đây là tín hiệu nhỏ, chưa đủ để chốt preprocessing.
- Spacing 1 mm full fold 2/seed 42 đã hoàn tất sạch (`21858`); P0 fold 2 (`21859`) đang chạy để so sánh cùng fold. Kết quả spacing fold 2 hiện là Dice `0,6723`.
- Các số spacing được ghi ở [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md) và [`docs/MEN_RT_PREPROCESS_REVIEW.md`](docs/MEN_RT_PREPROCESS_REVIEW.md). Không dùng kết quả fold 2 resume cũ làm xác nhận chính thức.

## Lệnh kiểm tra nhanh

```bash
python transfer/scripts/validate_canonical_dataset.py
python transfer/scripts/validate_preprocessed_dataset.py \
  --dataset-root datasets/preprocessed \
  --manifest code/brats-research/manifests/local_dataset.csv \
  --expected-train-count 500 \
  --expected-validation-count 70
```

Chỉ tiếp tục preprocessing/training khi cả hai lệnh trả exit code `0` và JSON có `"valid": true`.
