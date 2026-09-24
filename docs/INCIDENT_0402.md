# Incident: duplicate case 0402 và silent overwrite

## Tóm tắt

Pipeline cũ phát hiện `BraTS-MEN-RT-0402-1` bị trùng nhưng chỉ ghi `critical` vào report, không dừng và không loại một bản. Hai dòng sau đó cùng được preprocess vào một output đặt tên theo `case_id`, làm bản xử lý từ training v2 ghi đè output patch tốt hơn.

## Bằng chứng tái lập

- `manifests/manifest_raw.csv` có 0402 tại dòng 2 (`ad_hoc_labeled`, standalone patch) và dòng 321 (`train`, training v2).
- `reports/failed_cases.csv` đánh dấu cả hai là `duplicate_case_id,critical`.
- `logs/preprocessing_log.jsonl` cho thấy patch được xử lý trước, training-v2 xử lý sau; cả hai dùng cùng tên output.
- `src/brats_men_rt/stages/preprocessing.py` lặp tất cả manifest rows và dựng output chỉ từ `case_id`; không có guard duplicate/overwrite.
- `src/brats_men_rt/stages/splitting.py` vừa collapse row theo dictionary `case_id`, vừa dùng QC rows còn trùng. Hậu quả 0402 xuất hiện ở validation của fold 3 và fold 4.

Đây là lỗi **tách/gộp ở cấp manifest và split**, sau đó thành lỗi **ghi đè ở preprocessing**. Không phải do hai ZIP bị giải nén lẫn nhau.

## Lỗi thứ hai tìm thấy trong preprocessing cũ

Hàm `_normalize_image` lấy percentile từ foreground ban đầu nhưng gọi `np.clip` trên **toàn bộ ảnh**. Khi lower percentile dương, background `0` bị đổi thành lower percentile. Hàm sau đó tính lại `work != 0`, khiến background vừa bị đổi cũng đi vào mean/std. `_crop_bounds` chạy trên ảnh đã normalize nên thấy toàn bộ FOV khác 0 và không crop được.

Smoke test corrected 0402 xác nhận raw có 16,690,299 voxel khác 0 trên 24,780,800 voxel, nhưng legacy output có đủ 24,780,800 voxel khác 0. QC lịch sử cho thấy hiện tượng “toàn bộ output khác 0” ở **556/571 rows**, gồm **485/500 train rows**, 70/70 validation rows và standalone 0402. Đây là bằng chứng phạm vi, không chỉ suy đoán từ một case.

Do nnU-Net raw được dựng từ `manifest_preprocessed.csv` và symlink tới `data_processed`, baseline cũ đã dùng các output legacy này. nnU-Net có thể tự crop/normalize thêm nên chưa thể quy toàn bộ sai số metric cho lỗi này, nhưng artifact preprocessing cũ không đúng với ý định “nonzero clip/z-score + crop”.

## Ảnh hưởng đã thấy

- Split cũ có 500 unique train case nhưng tổng validation assignments qua 5 fold là 501.
- 0402 nằm trong `splits/fold_3_val.txt` và `splits/fold_4_val.txt`.
- Dice cũ của 0402: fold 3 khoảng `0.02661`, fold 4 bằng `0`; cross-validation chọn giá trị fold 3.
- Baseline nnU-Net cũ có mean cross-validation Dice `0.773835` trên 500 case, nhưng split không còn hợp lệ về phương pháp và metric 0402 bị ảnh hưởng.
- Bundle P0--P3 ban đầu thiếu 0402 ở cả bốn profile. 0402 hiện đã được preprocess lại từ patch đúng và bổ sung đủ.

Không có bằng chứng các case khác bị duplicate theo cùng cách. Tuy vậy, mọi kết quả phụ thuộc split cũ cần được gắn nhãn “legacy/affected”, không dùng như so sánh công bằng với pipeline mới.

## Khắc phục

1. Giữ raw archive/extracted nguyên trạng để audit.
2. Tạo canonical symlink view có đúng 500 train + 70 validation.
3. Override đúng một case 0402 bằng standalone patch.
4. Tạo lại manifest từ canonical view.
5. Validation phải trả exit code khác 0 khi có duplicate, overlap, thiếu hoặc thừa file.
6. Preprocess vào staging; chỉ cài output khi toàn bộ profile pass.

Nếu còn cần baseline legacy, phải sửa thứ tự/mask clipping rồi preprocess lại **toàn cohort** và tạo lại nnU-Net raw/preprocessed + splits. Không ghi đè artifact cũ; giữ chúng dưới nhãn `legacy_affected` để đối chiếu.

Run sửa 0402 trên remote: Slurm job `20932`, hoàn tất thành công; metadata tại `experiments/preprocessing_0402/runs/20260907T123944Z_ab972e57/run.json`.

QC thấy label từ 2 connected components xuống 1 sau nearest-neighbor resampling. Component bị mất chỉ đúng 1 voxel raw (`0.3719 mm³`); tổng label volume thay đổi khoảng `-0.13094%`. Đây là cảnh báo resampling cần lưu lại, không phải mất khối u chính.
