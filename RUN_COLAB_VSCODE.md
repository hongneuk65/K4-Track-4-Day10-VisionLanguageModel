# Chạy lab với kernel Colab trong VS Code

**Đã chuyển sang cách không dùng Drive:** notebook hiện nhúng mã nguồn và xuất kết quả qua output đã lưu. Làm theo [RUN_COLAB_NO_DRIVE.md](RUN_COLAB_NO_DRIVE.md). Nội dung bên dưới là hướng dẫn cũ khi dùng Drive.

## Trạng thái

Phần mã bắt buộc E0–E4 đã triển khai trong `lab/`. Chưa có số liệu GPU hoặc dataset thật tại máy local. Chưa thể coi bài đã hoàn thành trước khi chạy GPU, đọc ảnh và hoàn thiện REPORT.

`scripts/`, `tests/` và cấu hình đề giữ nguyên. `REPORT.md` giữ mẫu gốc cho đến khi có evidence thật. E5 chỉ có hàm reward và test; chưa có GRPO hay evidence bonus.

## 1. Đưa mã local lên server Colab

Notebook local mở trong VS Code nhưng Python chạy trên server Colab; server không tự đọc được thư mục D: của máy.

1. Chọn kernel Colab GPU trong notebook.
2. Command Palette → `Colab: Mount Google Drive to Server...`, chạy snippet mà extension chèn. Drive cần mount tại `/content/drive`.
3. Chạy ở terminal **local**, tại repo: `python -m lab.package_colab`.
4. Upload `work/lab-colab-bundle.zip` lên Drive, đặt ở `MyDrive/lab-day10/lab-colab-bundle.zip`.
5. Mở `lab_day10.ipynb` đã cập nhật và chạy từng cell. Cell đầu giải nén source vào `/content/lab-day10` và cài dependency trong tiến trình Python của server.

Notebook không cần clone phiên bản upstream thiếu phần mã mới. Nếu đổi source local, tạo lại bundle; dùng thư mục ROOT mới hoặc cập nhật source trên server trước khi chạy tiếp.

Tài liệu extension chính thức: https://github.com/googlecolab/colab-vscode#commands

## 2. Dữ liệu và secrets

Notebook thử lấy Kaggle credentials từ Colab Secrets, rồi dùng `getpass` nếu API secret không khả dụng. Không print secret. Nếu notebook frontend không hỗ trợ nhập ẩn, tải dataset zip bằng tay lên Drive và bỏ cell credentials; đổi lệnh prepare sang `--zip /content/drive/MyDrive/.../dataset.zip`.

Data audit kiểm tra session, image ID, số đếm, đường dẫn ảnh và chuyển đổi box. Box suy biến sau làm tròn gây lỗi rõ ràng để bạn điều tra, không âm thầm bỏ nhãn.

## 3. Thực hiện thí nghiệm

- E0: thử 8 ảnh val rồi suy luận đủ test.
- E1–E3: mỗi run nạp model gốc riêng; cùng seed 42, rank 16, alpha 32, dropout 0.05, LR 2e-4, 300 bước optimizer và effective batch 8.
- Nếu OOM, dùng batch 1 / accumulation 8 và chạy lại các setup so sánh với thiết lập nhất quán. Dùng thư mục output mới, giữ log lỗi.
- Train kiểm tra mask, 220 vision tokens, forward loss, module LoRA và phạm vi tham số train. LoRA/projector train dùng fp32, autocast forward fp16, GradScaler và gradient checkpointing.
- E2/E3 lưu cả adapter và `projector.pt`. Nạp trong tiến trình mới, kiểm tra key/shape projector trước suy luận.
- Chọn setup cho E4 bằng F1 validation. Chỉ sau khi chốt mới chạy test.
- E4 đổi tỷ lệ train 10%/100%, giữ 300 bước; 100% tái sử dụng prediction của setup gốc và chấm bằng score.py với tên E4. Subset được chọn cố định và lưu image ID.
- Kết quả E4 10%/100% đều phải có evidence. Một file E4 qua checker chưa đủ rubric.

Mỗi lệnh GPU chạy trong subprocess riêng để giải phóng model sau run. Không tự động giảm batch, đổi prompt hoặc tăng độ dài generation sau khi xem test. Nếu output bị cắt, điều chỉnh trên val, ghi lại lựa chọn rồi chốt.

## 4. Checkpoint và ngắt session

Mỗi 100 bước lưu adapter, projector nếu có, processor, training state và run.json; copy vào `MyDrive/lab-day10/checkpoints/<run>/step-<step>`.

Lệnh train hiện **không tự resume** từ training_state; file này phục vụ khôi phục có chủ đích. Nếu run bị ngắt trước 300 bước, chạy lại bằng thư mục mới. Không nộp checkpoint 100/200 bước như run đủ 300.

Nếu run đã xong và mất disk Colab, copy thư mục run từ Drive về `work/runs/<run>` rồi chạy cell evaluation. Trọng số nằm trên Drive, không đưa vào Git.

Prediction được flush từng batch. Nếu bị ngắt, lệnh không âm thầm append; dùng file output mới để chạy lại. Lệnh `lab score` yêu cầu đủ đúng image ID của split, chặn chấm file subset như full split.

## 5. Báo cáo và tải kết quả về

Sau E4, notebook tạo:

- `report/evidence/e0.json`–`e3.json` và hai file E4;
- `report/data_audit.json`, `report/scale_probe.json`;
- sáu ảnh GT và sáu ảnh dự đoán/GT có TP, FP, FN;
- `report/REPORT_DRAFT.md`, dùng số thật và giữ marker ở phần cần quan sát cá nhân.

Đọc draft, output E0 và ảnh minh họa. Điền họ tên, hai lỗi E0, phân tích tradeoff, nguyên nhân theo lớp, E4 và giới hạn. Chuyển bản hoàn chỉnh vào REPORT.md. Không chỉ xóa marker để vượt checker.

Notebook backup report/prediction sau evaluation và tạo `MyDrive/lab-day10/results/report-artifacts.zip`. Tải zip về, đưa nội dung vào `report/` trong repo local. Copy raw prediction về `work/` nếu cần đối chiếu; work bị gitignore.

## 6. Kiểm tra trước nộp

```bash
python -m pytest -q tests student_checks --basetemp=work/final-checks
python scripts/check_submission.py
git diff -- scripts tests
git status --short
git ls-files
```

Sau khi REPORT được điền, test đề kiểm tra marker mẫu REPORT có thể fail vì chính test yêu cầu `<điền`; giữ test nguyên trạng và ghi nhận xung đột này. Các kiểm tra mã còn lại phải pass.

Checker không kiểm tra toàn bộ rubric: tự đối chiếu E1 parse >=0.95, F1 tăng >=0.10, đủ hai giá trị E4, sáu ảnh có nhận xét, log tài nguyên và báo cáo khớp evidence. Tạo ít nhất năm commit theo tiến độ thực tế, không tạo commit giả cho thí nghiệm chưa chạy. Không commit secret, data, work hoặc trọng số.
