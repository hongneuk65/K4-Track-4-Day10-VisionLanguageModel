# Báo cáo LAB Day 10 — Bản nộp tạm

Họ tên: Nguyễn Văn Hồng  
Ngày cập nhật: 16/10/2026  
dataset_version: `d10-21e4070301`

## Trạng thái bản nộp

Đây là bản nộp tạm theo yêu cầu của học viên, chưa phải bản hoàn tất.
E1–E3 đã huấn luyện 300 bước và chấm đủ 311 ảnh test trên Colab.
E4 (10% dữ liệu train) đang chạy; log gần nhất xác nhận bước 200/300.
Chưa có kết quả E4 cuối cùng, không suy diễn số liệu từ loss.

Các số E1–E3 dưới đây được ghi lại từ log scorer trên Colab, làm tròn ba chữ số.
Evidence JSON E1–E3, ảnh dự đoán và thông tin tài nguyên đầy đủ hiện được lưu trong
`MyDrive/lab-day10/results/e3-test.zip` của tài khoản chạy
`thainguyenvan193@gmail.com`; chưa đồng bộ về commit nộp tạm này.
Repo hiện có evidence JSON chính xác của E0. Không tạo JSON thay thế từ các số làm tròn.

Notebook: [lab_day10_colab.ipynb](https://colab.research.google.com/drive/1bdy_FR8JcOyX287_Jw6TK4kS5d6xBq7N?authuser=2).

## Kết quả đã quan sát

- E0, zero-shot: test F1 = 0.05544554455445545; JSON hợp lệ = 0.6302250803858521.
  CI95 F1 = [0.037037037037037035, 0.07684319833852545].
  Nguồn: `report/evidence/e0.json`.
- E1, LoRA LLM: validation F1 = 0.195; test F1 = 0.260,
  precision = 0.281, recall = 0.242, JSON hợp lệ = 0.987.
- E2, LoRA LLM + projector đầy đủ: validation F1 = 0.270;
  test F1 = 0.332, precision = 0.353, recall = 0.312, JSON hợp lệ = 1.000.
- E3, thêm LoRA ViT: validation F1 = 0.363; test F1 = 0.432,
  precision = 0.459, recall = 0.408, JSON hợp lệ = 1.000.
  Loss bước 300 = 0.4278.
- E4: chọn setup E3 bằng validation, so sánh 10% với 100% dữ liệu train;
  giữ 300 bước. Chưa báo cáo điểm cuối.

E3 được chọn bằng validation trước khi đọc kết quả test.
Trên phiên chạy này, mở projector tăng test F1 khoảng 0.072 so với E1;
thêm LoRA ViT tăng khoảng 0.100 so với E2.
Cần dùng CI95 và tài nguyên từ JSON gốc để hoàn tất nhận xét về hiệu quả chi phí.

## Dữ liệu và phương pháp

Chia theo phiên quay: train 1.422 ảnh/13 phiên, validation 276 ảnh/2 phiên,
test 311 ảnh/3 phiên. Số box pothole/crack/manhole tương ứng:
train 1.074/1.750/691; validation 66/470/103; test 121/299/163.
Chia theo phiên giúp hạn chế rò rỉ giữa các khung hình gần nhau.

Model Qwen/Qwen3-VL-4B-Instruct; GPU T4; batch size 1, gradient accumulation 8,
seed 42, 300 bước mỗi thí nghiệm. Tọa độ đầu ra là số nguyên thang 0–1000.
Chỉ tính loss trên câu trả lời; không sửa output thô trước khi chấm.
LLM dùng NF4; ViT và projector giữ khả năng huấn luyện phù hợp;
projector huấn luyện bằng float32.
E1 cập nhật 33.030.144 tham số, E2 142.135.296, E3 148.426.752.

## Còn thiếu trước bản cuối

- Hoàn thành E4 và chấm validation/test.
- Đồng bộ evidence JSON E1–E4 cùng ảnh minh họa từ Drive về repo.
- Hoàn thiện 10 câu trả lời theo mẫu đề, CI95, thời gian và VRAM thực đo.
- Chạy lại `python scripts/check_submission.py` và xử lý các mục FAIL.
- Cập nhật commit cuối lên GitHub.

Bản tạm chưa đáp ứng đầy đủ điều kiện nộp của README/RUBRIC.
Không tính E5/GRPO là thí nghiệm đã hoàn thành.
