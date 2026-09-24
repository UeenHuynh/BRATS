# Data provenance và canonical policy

## Nguồn chính thức

Dataset chính là BraTS-MEN-RT trên Synapse (`syn59059779`). Workspace đang giữ ba archive:

| Nội dung | Synapse | Phiên bản | MD5 local |
| --- | --- | --- | --- |
| Training, 500 case | `syn60085033` | v2 | `5bd14a6794e5874b6dd2a09a64795a4c` |
| Validation, 70 case | `syn61484746` | v1 | `885678ebe03224a3cce4b3a82f861c99` |
| Standalone case 0402 | `syn64826221` | v1 | `aadfd702c07ce72709e5249b73333ad8` |

Changelog chính thức cho biết training v2 sửa sai khác NIfTI direction/spacing giữa image và label ngày 30-05-2024: <https://www.synapse.org/Synapse:syn53708249/discussion/threadId=11148>. Trang patch 0402: <https://www.synapse.org/Synapse:syn64826221>.

## Vì sao 0402 là override

Standalone archive không phải case train thứ 501. Nó có cùng case ID và cùng label với bản nằm trong training v2, nhưng image khác hoàn toàn:

| File image 0402 | SHA-256 |
| --- | --- |
| Standalone patch, được chọn | `5969ef53c6f43549c0fd42e1582e3caa2eac89d3e4dc74c453e4e319ac981fcf` |
| Bản cũ trong training v2, bị loại | `0033422b0b8c1ce6df10524641b7f9524c6c036515b8212a48c171574ccda059` |

Hai bản có cùng shape `352 x 352 x 200`, spacing, affine/header và nonzero support; label có cùng SHA-256 `72e1bea52eb70845bb123dcf69772269f8b59c822ebdaa4e65095379a36cb457`. Tuy nhiên image training-v2 có dải giá trị gần toàn miền signed 16-bit, trong khi patch có dải cường độ MRI hợp lý hơn. Hash member trong ZIP khớp file giải nén, nên đây không phải lỗi extraction.

Canonical policy: **patch 0402 thay thế bản 0402 trong training v2**. Không gộp cả hai, không cho cả hai đi vào manifest, và không ghi đè raw source.

## Cách biểu diễn

`datasets/raw_canonical/` dùng symlink tương đối tới raw nguyên trạng. Mọi case train khác trỏ vào `BraTS-MEN-RT-Train-v2`; riêng 0402 trỏ vào standalone patch. Manifest chuẩn là `code/brats-research/manifests/local_dataset.csv`.

Kiểm tra thường ngày không hash ba ZIP lớn:

```bash
python transfer/scripts/validate_canonical_dataset.py
```

Khi vừa tải/chuyển archive hoặc nghi ngờ file hỏng:

```bash
python transfer/scripts/validate_canonical_dataset.py --verify-archives
```

Thông tin mô tả dataset và preprocessing nguyên bản được công bố tại <https://www.nature.com/articles/s41597-026-06649-x>. Bài báo mô tả dữ liệu ở native orientation/resolution, không intensity resampling hay skull stripping; bài báo không nêu riêng sự cố 0402.
