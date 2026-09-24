# Protocol ablation 5 nhánh — lệnh chạy trên `ydang`

> Protocol gốc chốt ngày **2026-09-09**. Full preprocessing `20949`, `20950` và validator `20962` đã pass. Smoke `21002` pass; array `21003` sau đó bị hủy và chưa đủ kết quả để aggregate. Các pilot spacing 1 mm được ghi riêng trong [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).

## Câu hỏi và thiết kế

Protocol dùng năm nhánh:

| Nhánh | Vai trò |
| --- | --- |
| `legacy_corrected` | pipeline cũ đã sửa, dùng làm đối chứng end-to-end |
| `P0_MEDIAN_ZSCORE` | không clipping, không N4 |
| `P1_MEDIAN_CLIP_ZSCORE` | clipping, không N4 |
| `P2_MEDIAN_N4_ZSCORE` | N4, không clipping |
| `P3_MEDIAN_N4_CLIP_ZSCORE` | clipping và N4 |

P0--P3 là ablation 2×2 để tách tác dụng clipping/N4. `legacy_corrected` là đối chứng pipeline; không diễn giải nó như một ablation intensity thuần vì grid/crop khác P0--P3. Artifact legacy cũ bị lỗi background/duplicate 0402 tuyệt đối không được đưa vào training mới.

Mỗi nhánh dùng corrected 5 folds trên đủ 500 case, folds `102/101/99/99/99`, ba seed `42/2026/3407`. Trong phần outer-train, 10% case cố định được giữ làm inner validation; outer-test không được dùng để chọn checkpoint. Model, loss, optimizer, augmentation patch và training budget giống nhau giữa năm nhánh.

Runner cố định Python/NumPy/PyTorch/CUDA seed, tắt cuDNN benchmark và bật cuDNN deterministic. PyTorch/CUDA không cung cấp deterministic backward tuyệt đối cho `MaxPool3D` trên GPU này; ba seed được dùng để lượng hóa biến thiên còn lại thay vì tuyên bố bitwise reproducibility.

Cả inner-validation prediction dùng để chọn checkpoint và outer-test prediction đều được resample bằng physical geometry về raw canonical mask trước khi tính metric. Primary endpoint là Dice theo case, trung bình qua ba seed trước khi tổng hợp 500 case. Secondary endpoints gồm HD95, Surface Dice 2 mm, sensitivity và precision. Aggregator tạo paired bootstrap 95% CI, Wilcoxon normal approximation và Holm-adjusted p-value.

## Đồng bộ runner, config và tài liệu lên remote

Chạy ở máy local, từ workspace root:

```bash
cd /media/neeyu/hien_3/uyen/BRATS
bash transfer/scripts/sync_ablation_to_ydang.sh
bash transfer/scripts/sync_ablation_to_ydang.sh --execute
```

Script đầu tiên chỉ dry-run. Bản `--execute` chỉ cập nhật runner, config và các tài liệu liên quan; không chuyển dataset, không xóa file và không submit job.

## Preflight bắt buộc

SSH vào remote:

```bash
ssh -F /dev/null -i /home/neeyu/.ssh/id_ed25519_brats \
  -o IdentitiesOnly=yes ydang@130.191.49.60
```

Sau đó chạy:

```bash
cd /mnt/beegfs/ydang/BRATS/code/brats-research
source /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/brats.env
export PYTHONPATH="$PROJECT_ROOT"

"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/ablation_5arm_5fold_3seed.yaml \
  preflight
```

Chỉ submit khi lệnh kết thúc bằng:

```text
PASS: 5 arms, 5 folds, 3 seeds, 500 cases
```

Kiểm tra Slurm mà chưa submit thật:

```bash
mkdir -p /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs
sbatch --test-only \
  /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/run_ablation_5arm_array.sbatch
```

Chạy smoke end-to-end 2 epoch trên tập nhỏ trước array chính:

```bash
SMOKE_JOB_ID=$(sbatch --parsable \
  /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/run_ablation_smoke.sbatch)
echo "$SMOKE_JOB_ID"
squeue -j "$SMOKE_JOB_ID"
```

Sau khi job xong, bắt buộc kiểm tra:

```bash
sacct -j "$SMOKE_JOB_ID" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS -P
test -f /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/_smoke/P1_MEDIAN_CLIP_ZSCORE/fold_0/seed_42/DONE.json
test ! -s /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/smoke_${SMOKE_JOB_ID}.err
```

Chỉ submit array chính khi smoke ở trạng thái `COMPLETED`, exit `0`, có `DONE.json` và stderr rỗng. Smoke output nằm dưới `_smoke/` nên aggregator không đọc nhầm thành kết quả chính.

## Submit

Chạy toàn bộ 75 task, tối đa bốn GPU task đồng thời:

```bash
JOB_ID=$(sbatch --parsable \
  /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/run_ablation_5arm_array.sbatch)
echo "$JOB_ID"
```

Để giảm rủi ro tài nguyên, có thể chạy seed 42 trước trên đủ năm nhánh × năm folds:

```bash
JOB_ID=$(sbatch --parsable \
  --array=0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72%4 \
  /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/run_ablation_5arm_array.sbatch)
echo "$JOB_ID"
```

Không chạy đồng thời hai array chứa cùng task ID: runner từ chối task đã có `DONE.json` nhưng hai task trùng đang chạy có thể ghi cùng checkpoint.

## Tự kiểm tra khi đang chạy

Thay `<JOB_ID>` bằng ID in ra lúc submit:

```bash
squeue -j <JOB_ID> -o "%.18i %.30j %.10T %.10M %.24R"
sacct -j <JOB_ID> --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS -P
```

Theo dõi một task:

```bash
tail -f /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs/task_<JOB_ID>_0.out
```

Tìm stderr có nội dung:

```bash
find /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/logs \
  -type f -name '*.err' -size +0c -print
```

Đếm task hoàn tất và lấy danh sách array ID còn thiếu:

```bash
cd /mnt/beegfs/ydang/BRATS/code/brats-research
source /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/brats.env
export PYTHONPATH="$PROJECT_ROOT"

"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/ablation_5arm_5fold_3seed.yaml \
  status
```

`status` trả exit `1` khi còn thiếu và in `MISSING_TASK_IDS=...`. Có thể submit lại đúng các ID đó bằng `sbatch --array=<danh_sách>%4 ...`. Runner chỉ coi task hoàn tất khi có `DONE.json` và từ chối ghi lên thư mục task partial. Trước khi retry, đổi tên thư mục partial để giữ bằng chứng, ví dụ:

```bash
mv /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/<ARM>/fold_<FOLD>/seed_<SEED> \
   /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/<ARM>/fold_<FOLD>/seed_<SEED>_failed_<JOB_ID>
```

## Tổng hợp sau khi đủ 75/75

```bash
"$BRATS_VENV_DIR/bin/python" -m brats.experiments.ablation \
  --config /mnt/beegfs/ydang/BRATS/transfer/remote_configs/ydang/ablation_5arm_5fold_3seed.yaml \
  aggregate

"$BRATS_VENV_DIR/bin/python" -m json.tool \
  /mnt/beegfs/ydang/BRATS/experiments/ablation_5arm_5fold_3seed/aggregate.json | less
```

Không chạy `aggregate` trên kết quả thiếu: command sẽ fail và liệt kê các `DONE.json` chưa có. Không đổi threshold, postprocessing hoặc chọn lại fold sau khi đã xem outer-test results.
