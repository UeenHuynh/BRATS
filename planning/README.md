# Trạng thái tài liệu planning

Các file đánh số trong thư mục này là kế hoạch thiết kế ban đầu của **pipeline legacy**. Chúng được giữ để audit quyết định nghiên cứu, không phải runbook hiện tại và không phản ánh tự động canonical override 0402.

Trước khi thực thi bất kỳ bước nào trong các kế hoạch cũ, đọc theo thứ tự:

1. [`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md)
2. [`../docs/INCIDENT_0402.md`](../docs/INCIDENT_0402.md)
3. [`../docs/VALIDATION_AND_RERUN.md`](../docs/VALIDATION_AND_RERUN.md)

Các kế hoạch về label semantics, metric và giới hạn clinical claim vẫn có giá trị khái niệm. Các kế hoạch inventory, preprocessing, split và baseline phải được sửa/re-run trên canonical manifest trước khi dùng cho kết quả mới.
