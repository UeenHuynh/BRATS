# ydang small preprocessing benchmark

Trước khi chạy, đọc `docs/VALIDATION_AND_RERUN.md` ở workspace root. Cấu hình trong thư mục này là bản remote-specific; tài liệu provenance có thẩm quyền nằm ở `docs/DATA_PROVENANCE.md`.

This pilot compares P0--P3 on the same deterministic seed-42 split:

- 40 common labeled cases selected by the benchmark.
- 32 training and 8 validation cases.
- 3 epochs per profile.
- 3D U-Net, `64 x 128 x 128` patches, batch size 2.
- RTX A6000 requested through Slurm `LocalQ`.

The run is a pipeline and ranking smoke test. Its sample size and training budget
are too small for a final scientific preprocessing conclusion.

## Follow-up pilot

`pilot_preprocessing_3d_100c_10e.yaml` and
`run_pilot_preprocessing_100c_10e.sbatch` run a stronger comparison with 100
common cases, an 80/20 train-validation split, and 10 epochs per profile. The
seed and all other model, optimizer, patch, and evaluation settings remain the
same as the smoke test.

## Corrected case 0402

`run_preprocess_0402.sbatch` processes the standalone Synapse 0402 patch in an
isolated staging directory. It installs the result into P0--P3 only after every
profile succeeds and verifies that no destination file already exists.

Job sửa đã hoàn tất là `20932`. Không submit lại script install này khi bốn destination đã tồn tại; dùng một output root staging mới cho các lần smoke test.

Smoke test legacy--new P1 đã hoàn tất ở job `20935`; report nằm tại `experiments/preprocessing_compare_0402/run_20935/comparison.json`. Script yêu cầu A6000 vì các job CPU-only `20933` và `20934` bị node `mantis-10` fail trước khi tạo stdout; bản thân profile P1 không chạy N4/TotalSegmentator.

Theo dõi job:

```bash
squeue -u ydang
tail -f /mnt/beegfs/ydang/BRATS/experiments/unet/logs/<job-log>.out
```

## Full rerun status — 2026-09-09

- Full preflight: PASS trên 570 cases và 4 profiles.
- Corrected legacy smoke5: job `20948`, PASS 5/5.
- Corrected legacy full preprocessing: job `20949`, COMPLETED/validated (570 images, 500 labels).
- New P0--P3 full preprocessing: job `20950`, COMPLETED, exit `0`, 570 cases × 4 profiles, 0 failure.
- New full membership validator: job `20962`, COMPLETED, exit `0`, `valid: true`.
- Matched ablation được cấu hình tại `ablation_5arm_5fold_3seed.yaml`. Smoke `21002` pass; array chính `21003` gồm 75 task đã submit và đang chạy. Lệnh theo dõi nằm tại `docs/ABLATION_PROTOCOL.md` ở workspace root.
