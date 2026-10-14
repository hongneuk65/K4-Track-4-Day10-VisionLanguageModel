# Ghi nhận phiên Colab thực tế ngày 14/10/2026

Học viên: Nguyễn Văn Hồng. Đây là ghi chú tiến độ, chưa phải báo cáo hoàn chỉnh hay evidence JSON của scorer.

Notebook: https://colab.research.google.com/drive/1bdy_FR8JcOyX287_Jw6TK4kS5d6xBq7N

GPU T4; Python 3.13; Transformers 5.19.0. Dataset_version `d10-21e4070301`. Train 1422, validation 276, test 311 ảnh. E0 đã chạy đủ 311 ảnh, inference batch 4, giới hạn 256 token, greedy decoding. Log scorer trên Colab hiển thị các giá trị làm tròn: parse_valid_rate 0.630, precision 0.066, recall 0.048, F1 0.055, CI95 0.037–0.077, meanIoU 0.627, FP/ảnh 1.28. F1 từng lớp: pothole 0.090, crack 0.003, manhole 0.142. Khi viết REPORT phải dùng số chính xác từ `report/evidence/e0.json`, không thay thế bằng các số làm tròn ở ghi chú này.

Smoke test validation quan sát được output của ảnh `vlcsnap-2025-02-18-18h30m20s293` lặp cùng box `[10, 550, 100, 600]` và khóa `bbox_2d` nhiều lần; output kết thúc giữa một mảng tọa độ nên không phải JSON hoàn chỉnh. Ảnh `vlcsnap-2025-02-18-18h30m22s081` lặp khóa `bbox_2d` với nhiều mảng trong cùng object và cũng bị cắt. Không sửa output thô trước khi chấm.

Ví dụ output hoàn chỉnh của ảnh `vlcsnap-2025-02-18-18h30m25s530` trong smoke validation:

```json
[{"label":"manhole","bbox_2d":[607,515,670,545]},{"label":"crack","bbox_2d":[0,375,1000,999]}]
```

Ví dụ này chỉ chứng minh output có thể hợp lệ cú pháp; chưa kết luận box đúng nếu chưa đối chiếu ground truth của chính ảnh đó.

E1 chạy với batch 1, accumulation 8, seed 42, 300 bước theo config. Log đã quan sát: bước 1 loss 1.2199; 10: 0.7222; 20: 0.7395; 30: 0.6325; 40: 0.6130; 50: 0.6156. Đây chưa phải kết quả E1 cuối cùng. Các cell E2, E3, chọn setup bằng validation rồi đánh giá test và E4 data size đã được xếp chạy tuần tự sau E1.

Kiểm tra local: 85 passed, 1 skipped (test dataset yêu cầu biến môi trường). Mã đề trong scripts/, tests/, configs/ giữ nguyên. REPORT.md vẫn cần kết quả E1–E4 và phân tích ảnh trước khi bài đủ điều kiện nộp.
