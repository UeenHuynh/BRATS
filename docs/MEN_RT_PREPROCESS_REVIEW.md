# Tiền xử lý BraTS-MEN-RT: đối chiếu nguồn công bố và hướng thử tiếp

Kiểm tra cập nhật ngày 2026-09-23. Kết quả P0--P3 gốc có phạm vi **fold 0, seed 42, 102 ca outer-test**, cùng model và protocol cho năm nhánh; spacing bổ sung có fold 1 và fold 2, seed 42. Đây chưa phải đánh giá đủ nhiều folds/seeds. Không dùng outer-test để chỉnh hyperparameter rồi báo lại như một phép thử độc lập.

## Kết quả hiện có

| Nhánh | Dice trung bình | Surface Dice 2 mm trung bình |
| --- | ---: | ---: |
| P0: median spacing + z-score | 0,6734 | 0,6692 |
| P1: P0 + percentile clipping | 0,6694 | 0,6738 |
| P2: P0 + N4 | 0,6589 | 0,6539 |
| P3: P0 + N4 + clipping | 0,6256 | 0,6221 |
| legacy_corrected | 0,5952 | 0,5694 |

Nguồn: `experiments/ablation_5arm_5fold_3seed/<arm>/fold_0/seed_42/DONE.json`. P0 có 20/102 ca Dice <0,5 và 10/102 ca Dice <0,1. Một số mask dự đoán lớn bất thường, ví dụ 0238 và 0270. P0 là mốc tốt nhất theo Dice trung bình trong lần thử này, nhưng chất lượng mô hình còn thấp và không ổn định. `legacy_corrected` còn khác P0 về grid/crop, nên so sánh này là **end-to-end pipeline**, không cô lập riêng hiệu ứng chuẩn hóa cường độ.

## Pilot spacing và các biến thể tiếp theo — 2026-09-23

Ba pilot dùng cùng U-Net 3D tự viết, seed 42, 100 ca, 10 epoch và 20 ca validation:

| Biến thể | Dice validation | HD95 (mm) | Precision |
| --- | ---: | ---: | ---: |
| P0 baseline | 0,0220 | 190,13 | 0,0118 |
| Spacing 1 mm đẳng hướng | **0,0356** | **181,50** | **0,0192** |
| Z-score theo brain mask | 0,0252 | 190,44 | 0,0135 |
| Augmentation cường độ | 0,0251 | 190,21 | 0,0134 |

Spacing 1 mm được đưa vào huấn luyện đầy đủ trên fold 1 và fold 2, seed 42. Fold 1 cho inner Dice `0,7294` so với P0 `0,7158`; outer Dice `0,6735` so với P0 `0,6719`. Fold 2 spacing đạt outer Dice `0,6723`; P0 fold 2 đang chạy nên chưa có so sánh paired hợp lệ.

Auto gate `21366` chọn spacing vì chênh lệch inner Dice fold 1 là `+0,0136`, với ngưỡng chọn `+0,01`. Đây chỉ là quy tắc chọn ứng viên trên inner validation; outer-test không được dùng để chọn. Kết quả fold 2 cũ đã resume từ checkpoint partial và được giữ riêng dưới `seed_42_resumed_21836`, không dùng làm xác nhận chính thức.

Diễn giải hiện tại: spacing 1 mm ổn định quanh Dice `0,67` nhưng chưa chứng minh tốt hơn P0. Cần hoàn tất P0 fold 2, sau đó mới quyết định có chạy thêm fold/seed xác nhận hay bỏ spacing.

## Người khác xử lý dữ liệu tương tự thế nào?

1. **Chính bộ BraTS-MEN-RT:** MRI T1 sau tiêm được phát hành ở hướng và độ phân giải gốc; nhóm tạo dữ liệu không nội suy cường độ và không skull-strip, vì cấu trúc ngoài sọ cũng có thể liên quan tới lập kế hoạch xạ trị. Họ dùng defacing để bảo vệ danh tính. Đây là quy trình **chuẩn bị dữ liệu công bố**, không phải khẳng định mọi mô hình phải huấn luyện ở native grid. Nguồn: [bài báo dữ liệu BraTS-MEN-RT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12948943/).
2. **nnU-Net v2:** với MRI, mặc định chuẩn hóa z-score **theo từng ca**. Pipeline crop vùng khác 0, có thể dùng mask khác 0 khi chuẩn hóa, rồi resample tới spacing do planner chọn; dự đoán được đưa về hình học ảnh gốc. Không có bước tăng tương phản cố định/CLAHE mặc định cho MRI. Nguồn: [tài liệu chuẩn hóa nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/explanation/intensity-normalization.md), [mã preprocessor](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/preprocessors/default_preprocessor.py).
3. **Nhóm xếp thứ ba BraTS 2024 MEN-RT:** chuyển dữ liệu sang định dạng nnU-Net và chạy `nnUNetv2_plan_and_preprocess`; MedNeXt của họ dùng target spacing 1 × 1 × 1 mm. Họ huấn luyện/ensemble nhiều checkpoint; điều này khác đáng kể baseline U-Net đơn hiện tại. Đây là pipeline của **một nhóm dự thi**, không phải chuẩn bắt buộc của bộ dữ liệu. Nguồn: [repo và hướng dẫn tái lập của nhóm](https://github.com/andre-fs-ferreira/BraTS_2023_2024_solutions/blob/main/BraTS2024_Task3.md).
4. **BraTS meningioma tiền phẫu 2023 là bài toán khác:** dữ liệu nhiều chuỗi được đăng ký vào atlas, resample 1 mm đẳng hướng và skull-strip. Không sao chép nguyên quy trình này sang MEN-RT chỉ có T1c và GTV ở native space. Nguồn: [bài báo dữ liệu BraTS-MEN tiền phẫu](https://www.nature.com/articles/s41597-024-03350-9).

## Gợi ý đã lưu cho lần thử kế tiếp

1. Giữ P0 làm baseline hiện tại. Chưa thêm CLAHE/tăng tương phản cố định, vì clipping P1 không tăng Dice, còn N4 P2/P3 thấp hơn trong lần thử này.
2. Audit các ca Dice thấp: xem overlay ảnh/mask, đối chiếu kích thước và vị trí, đo false positives, kiểm tra ca u nhỏ, đa ổ, hậu phẫu, sát nền sọ và lỗi hình học. Ưu tiên các mask dự đoán cực lớn.
3. Thử **augmentation cường độ chỉ khi huấn luyện** (ví dụ gamma, brightness/contrast ngẫu nhiên và bias field nhẹ), giữ ảnh validation/test nguyên quy trình P0. Đây là giả thuyết cần thử, không phải cải thiện đã được chứng minh. [MONAI có các biến đổi tương ứng](https://monai.readthedocs.io/en/latest/transforms.html).
4. Đối chiếu với baseline nnU-Net v2 3D full resolution trên cùng split và raw labels, lưu output riêng. Model hiện tại chỉ dùng crop ngẫu nhiên và flip, chưa có augmentation cường độ; tăng chất lượng huấn luyện có thể quan trọng hơn thêm bước tiền xử lý cố định.
5. Chọn threshold và postprocessing, nếu thử, trên **inner validation**; dùng fold/test chưa bị dùng để chọn phương án để đánh giá thay đổi. Không tự động giữ một connected component lớn nhất vì dữ liệu có thể có nhiều tổn thương.

Các thay đổi ảnh cố định (orientation, spacing, crop, normalization) cần tạo output staging mới, kiểm tra geometry/label và provenance trước khi huấn luyện; không ghi đè P0 đã kiểm chứng.

Audit 20 ca P0 có Dice <0,5 đã được lưu tại [`experiments/low_dice_audit/p0_fold0_seed42/REPORT.md`](../experiments/low_dice_audit/p0_fold0_seed42/REPORT.md). Ba ca có false positive gần như toàn bộ ở vùng ảnh bằng 0; phần lớn ca còn lại cần xử lý lỗi định vị/nhận diện.

Thử support mask tiếp trên 40 ca inner validation và đủ 102 ca outer-test cùng fold 0: Dice trung bình lần lượt `0,6681 → 0,6864` và `0,6734 → 0,6882`; chỉ một ca validation và ba ca test cải thiện lớn. Cả 500 nhãn P0 đều không có voxel GT trên nền zero. Vì ba ca test đã được xem trước khi thử phép che, kết quả test sau mask chỉ là phân tích khám phá, không dùng như cải thiện độc lập.

Đã kiểm tra nguy cơ loại nhầm hốc zero bên trong: hốc zero có ở 97/102 ca, nhưng không có voxel GT nào trong hốc ở 500 nhãn. Quy tắc `exterior-only` (chỉ loại zero thông với mép thể tích) cho Dice 0,688182, gần như quy tắc strict 0,688233, và an toàn hơn về mặt hình học. Chi tiết trong báo cáo audit.
