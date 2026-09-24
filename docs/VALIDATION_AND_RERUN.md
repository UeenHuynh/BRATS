# Validation gate và luồng chạy lại

## Luồng quyết định

```text
archive/raw nguyên trạng
        |
        v
build canonical view + manifest
        |
        v
canonical validator -- FAIL --> sửa source/override/manifest rồi kiểm tra lại
        |
       PASS
        v
preprocess vào experiments/<run>/staging
        |
        v
QC geometry + membership -- FAIL --> giữ staging để audit, không install
        |
       PASS
        v
install không ghi đè + validate đủ P0--P3
        |
        v
training/benchmark với split duy nhất, không duplicate
```

## Gate 1: canonical raw

```bash
python transfer/scripts/build_canonical_raw_view.py
python transfer/scripts/validate_canonical_dataset.py \
  --output experiments/canonical_validation.json
```

Pass khi có 570 unique rows, đúng 500 train/70 validation, mọi path tồn tại, và SHA-256 image 0402 là `5969ef...981fcf`. Sau transfer hoặc download, thêm `--verify-archives`.

## Gate 2: preprocessed membership

```bash
python transfer/scripts/validate_preprocessed_dataset.py \
  --dataset-root datasets/preprocessed \
  --manifest code/brats-research/manifests/local_dataset.csv \
  --expected-train-count 500 \
  --expected-validation-count 70 \
  --output experiments/preprocessed_validation.json
```

Validator mới fail thật bằng exit code `1`; không chỉ ghi warning. Pass khi cả P0--P3 khớp manifest, mỗi train image có label và không có case thừa.

## Khi nào phải preprocess lại

| Tình huống | Phạm vi chạy lại |
| --- | --- |
| Chỉ raw của một case được thay bằng patch, config không đổi | Chạy riêng case đó cho mọi profile, sau đó validate toàn bộ |
| Orientation, clipping, N4, crop, resampling hoặc target spacing đổi | Chạy lại toàn cohort cho profile bị ảnh hưởng |
| Chỉ manifest/split đổi, raw và preprocessing deterministic không đổi | Không cần preprocess; tạo lại split và training |
| Checksum output sai/thiếu, run metadata không chứng minh input hash | Chạy lại case/profile liên quan vào staging |
| Duplicate case hoặc train/validation overlap | Dừng; sửa canonical manifest và tạo lại split trước |

## Quy tắc staging/install

- Output thử nghiệm luôn ở `experiments/<tên_run>/`, không ghi thẳng vào dataset chính.
- Ghi `run.json`, resolved config, manifest dùng lúc chạy và checksum input/output.
- Không dùng `--overwrite` để che lỗi provenance.
- Chỉ `install` khi tất cả profile cần thiết thành công và destination chưa tồn tại.
- Sau install phải chạy lại Gate 2.

Remote 0402 dùng `transfer/remote_configs/ydang/run_preprocess_0402.sbatch`. File này đã áp dụng staging, fail-fast, kiểm tra đủ output và từ chối ghi đè.

## So sánh preprocessing cũ--mới

Smoke test công bằng phải dùng cùng raw canonical 0402:

- Cũ: RAS/canonical, clip 0.5--99.5 percentile, nonzero z-score, crop margin 8 voxel, không resample.
- Mới P1: RAS, clip 0.5--99.5 percentile trên head foreground, z-score, giữ full FOV, rectangular support, resample về cohort-median spacing.

Chạy hai pipeline vào hai staging root độc lập. So sánh input hash, output geometry, finite fraction, nonzero mean/std, label voxel/physical volume và connected components. Không so voxel trực tiếp nếu hai output khác grid. Smoke test kiểm tra tính đúng của processing, không thay thế benchmark model cùng split/epoch/seed.

Lưu ý: mô tả dòng “Cũ” phía trên là **ý định cấu hình**. Implementation legacy tại thời điểm incident clip toàn ảnh nên làm background khác 0 và vô hiệu hóa crop ở phần lớn case. Vì lỗi ảnh hưởng 485/500 train rows trong QC cũ, không được tái sử dụng artifact đó; phải dùng implementation corrected, regression tests và output root mới.

Phần legacy corrected đã được triển khai ngày 2026-09-08 và pass smoke5. Theo dõi full-run và training gate tại [`RERUN_READINESS.md`](RERUN_READINESS.md). Đoạn mô tả lỗi phía trên vẫn giữ để giải thích vì sao artifact legacy cũ bị invalidated.
