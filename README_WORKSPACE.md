# Cấu trúc BRATS workspace

> Trạng thái kiểm chứng: **2026-09-23**. Remote raw canonical đã đủ 500+70; legacy full `20949`, new P0--P3 full `20950` và validator `20962` đều pass. Spacing 1 mm đang được đối chiếu với P0 trên fold 2/seed 42.

## Nguồn dữ liệu và code có thẩm quyền

| Đường dẫn | Vai trò | Có nên sửa trực tiếp? |
| --- | --- | --- |
| `data/*.zip` | Archive gốc đã tải từ Synapse | Không |
| `data/extracted/` | Bản giải nén nguyên trạng, gồm training v2, validation v1 và patch 0402 | Không |
| `datasets/raw_canonical/` | View chuẩn, không trùng; tạo bằng symlink | Tạo lại bằng script, không sửa tay |
| `code/brats-research/manifests/local_dataset.csv` | Manifest chuẩn của code mới | Tạo lại bằng script |
| `datasets/preprocessed/P*` | Đường dẫn ổn định tới bốn profile P0--P3 | Chỉ cài output đã qua validation |
| `code/brats-research/` | Pipeline mới | Có, đây là Git repo riêng |
| `src/brats_men_rt/`, `scripts/`, `configs/` | Pipeline cũ để audit/tái lập | Không coi là nguồn data chuẩn |
| `experiments/` | Staging, log, run metadata và kết quả | Có; không dùng làm raw source |
| `transfer/` | Script kiểm tra/chuyển máy và cấu hình SSH `ydang` | Có |

`datasets/raw/brats_men_rt` là symlink tương thích trỏ về raw đã giải nén. `datasets/preprocessed/P*` hiện là symlink tương đối tới thư mục bundle có tên `data1 ` (có dấu cách ở cuối). Luôn quote đường dẫn đó nếu truy cập trực tiếp.

## Canonical raw view

`datasets/raw_canonical/` có đúng:

- `train/`: 500 case có image + label;
- `validation/`: 70 case chỉ có image;
- `PROVENANCE.json`: mô tả archive và override;
- case `BraTS-MEN-RT-0402-1`: symlink tới standalone patch `data/extracted/BraTS-MEN-RT-0402-1`.

Tạo lại view và manifest bằng:

```bash
python transfer/scripts/build_canonical_raw_view.py
python transfer/scripts/validate_canonical_dataset.py
```

Script idempotent và từ chối thay thế một đường dẫn thật hoặc symlink sai target. Nó không sửa/xóa raw archive.

## Máy chạy SSH

- Host: `130.191.49.60`
- User: `ydang`
- Remote root: `/mnt/beegfs/ydang/BRATS`
- Python env: `/mnt/beegfs/ydang/BRATS/envs/brats-py311`
- Slurm partition: `LocalQ`; GPU pilot dùng `rtx_a6000`

`transfer/scripts/push_workspace_to_remote.sh` mặc định dry-run, dùng `rsync -L` khi materialize dữ liệu profile và không dùng `--delete`. Đọc [`transfer/remote_configs/ydang/README.md`](transfer/remote_configs/ydang/README.md) trước khi submit job.

## Quy tắc an toàn

Không di chuyển/xóa archive, model, prediction hay kết quả cũ chỉ để làm đẹp cấu trúc. Các đường dẫn tuyệt đối trong metadata cũ vẫn có giá trị audit. Mọi lần chạy lại phải ghi sang `experiments/<tên_run>/` trước; chỉ cài vào dataset chính sau khi validation pass.
