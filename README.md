# BraTS-MEN-RT Research

Research code for preprocessing and tumor-segmentation benchmarks on the BraTS-MEN-RT dataset.

## Current results

The latest full 3D experiment trained `P3_MEDIAN_N4_CLIP_ZSCORE` for 30 epochs and evaluated 100 validation cases.

| Run | Model / profile | Epochs | Validation cases | Mean Dice | Mean HD95 (mm) | Sensitivity | Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 3D | `unet3d` / `P3_MEDIAN_N4_CLIP_ZSCORE` | 30 | 100 | **0.3314** | **110.09** | 0.6352 | 0.2606 |
| Full 2D | `unet2d` / `P3_MEDIAN_N4_CLIP_ZSCORE` | 30 | 100 | 0.0000 | N/A | 0.0000 | 0.0000 |

The full 3D result is the current usable baseline. The reported 3D Dice is the mean across validation cases; its median is 0.2251 (SD 0.2974), which indicates substantial case-to-case variability.

Detailed outputs are intentionally not tracked because they include large medical-image datasets and generated artifacts. The reference result files are:

- `outputs/full_3d/seed42_20260906T035516Z/benchmark.json`
- `outputs/full_2d/seed42_20260905T013521Z/benchmark.json`

## Current issues and limitations

1. **Full 2D training/evaluation is invalid.** Its final validation Dice, sensitivity, and precision are all zero, while HD95 is the floating-point maximum value. This usually means all predictions were empty or the evaluation path did not produce valid foreground masks. Do not compare or use this run as a baseline until the 2D data/model/evaluation pipeline is debugged.
2. **3D segmentation quality remains modest.** Mean Dice of 0.3314 and HD95 of 110.09 mm are insufficient for a strong segmentation result. Precision (0.2606) is especially low relative to sensitivity (0.6352), suggesting substantial false-positive volume.
3. **High variability across cases.** The 3D Dice median (0.2251) is much lower than its mean, and the standard deviation is 0.2974. Performance should be inspected per case and stratified by tumor volume/site before drawing broader conclusions.
4. **Limited experiment coverage.** The full result covers one preprocessing profile, one seed, and 30 epochs. The previous multi-profile benchmarks were short pilot runs, so they are not sufficient for robust profile selection.
5. **Data and generated outputs are local-only.** MRI inputs, preprocessing products, checkpoints, and logs are not committed to GitHub because of their size and sensitivity. Reproduction requires access to the corresponding dataset and local output paths.

## Recommended next steps

1. Debug the full 2D prediction and evaluation pipeline using a small, known-positive subset; verify logits, thresholding, and mask shape/spacing alignment.
2. Add qualitative overlays and per-case metric reports for the 3D run to identify failure modes.
3. Tune postprocessing/thresholding and loss weighting to reduce false positives, then run multiple seeds.
4. Re-run the four preprocessing profiles with matched full training budgets before selecting a final profile.

## Running benchmarks

The benchmark configuration files are in `configs/`. The Slurm workflows are in `scripts/`:

- `scripts/run_unet_benchmark_cheaha.sbatch`
- `scripts/tune_then_full_unet_cheaha.sbatch`

Run the benchmark module after preparing a dataset at the configured `dataset_root`:

```bash
uv run python -m brats.experiments.benchmark --config configs/benchmark_3d.yaml
```
