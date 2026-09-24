# Trạng thái reports

> Cập nhật 2026-09-07: các report trong thư mục này là artifact của pipeline legacy. Không dùng chúng để chứng minh canonical dataset hiện tại đã sạch.

Các report inventory/QC/preprocessing/split/baseline được tạo từ manifest 571 rows, trong đó case 0402 xuất hiện hai lần. Preprocessing legacy còn làm background khác 0 ở 556/571 rows, và split đưa 0402 vào validation của hai folds. Vì vậy các report này chỉ có giá trị audit:

- `dataset_summary.md`, `integrity_report.md`, `intensity_qc_report.md`;
- `orientation_spacing_qc_report.md`, `qc_overlay_report.md`;
- `preprocessing_report.md`, `preprocessing_delta_report.md`;
- `split_balance_report.md`, `leakage_check_report.md`;
- `metric_protocol.md`, `uncertainty_aware_segmentation_report.md`, `reproducibility_report.md`.

`label_semantics.md` vẫn có giá trị về phạm vi label/clinical claims. Các file chỉ ghi “Scaffold created” chưa chứa kết quả thực nghiệm.

Trạng thái có thẩm quyền hiện tại nằm ở [`../docs/README.md`](../docs/README.md), đặc biệt là incident 0402 và validation gates.
