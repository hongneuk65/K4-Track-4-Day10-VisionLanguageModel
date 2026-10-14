# Chạy lab trên Colab web — không dùng Drive

Dataset có một nhãn chiều cao 0 ở labels-YOLO/vlcsnap-00058.txt. Chuyển đáp án train mở trục co về tối thiểu 1 đơn vị; không sửa split hoặc ground truth của scorer. Ghi giới hạn này trong REPORT: box zero-area gốc không thể đạt IoU 0.5 với dự đoán có diện tích dương.

1. Mở https://colab.research.google.com/ và đăng nhập Google.
2. Chọn File → Upload notebook, chọn `lab_day10_colab.ipynb` bản **COLAB-WEB-2** (hoặc lab_day10.ipynb cùng nội dung). Không chọn bundle ZIP. Mở notebook mới thay vì tiếp tục tab đang chứa bản cũ.
3. Chọn Runtime → Change runtime type → T4 GPU → Save.
4. Chạy từng cell từ trên xuống. Cell đầu tự giải nén mã đã nhúng và cài dependency; không cần clone repo hoặc upload mã nguồn.

Cell khởi tạo và cell cài thư viện được tách riêng. Khởi tạo tạo work/, report/ và notebook trên disk trước khi pytest chạy. Log subprocess hiện trong cell. Batch train mặc định 1, accumulation 8; inference batch 1 để giảm VRAM trên T4. Run hoàn chỉnh chỉ được tái sử dụng khi cấu hình khớp; file bị ngắt không bị âm thầm append.
5. Cell dữ liệu mặc định `DATA_MODE = 'zip'`: chọn ZIP dataset road damage trên máy. Đây không phải `lab-colab-bundle.zip`. Tải dataset từ link Kaggle trong README nếu chưa có. Cách này không yêu cầu nhập Kaggle credentials vào notebook.
6. Nếu muốn tải qua Kaggle CLI, đổi DATA_MODE thành `kaggle` và thiết lập KAGGLE_USERNAME/KAGGLE_KEY trong Colab Secrets (biểu tượng chìa khóa, bật Notebook access). Notebook chỉ hỏi nhập nếu chưa có secret/environment.
7. Sau mỗi giai đoạn, notebook tạo ZIP và gọi `google.colab.files.download`. Cho phép trình duyệt tải nhiều file. File thường nằm ở Downloads hoặc vị trí bạn chọn trong hộp thoại trình duyệt; notebook không tự ghi trực tiếp vào ổ D:.
8. Nếu download bị chặn, mở bảng Files bên trái → `/content/lab-day10/work/exports/` → ZIP tương ứng → menu ba chấm → Download.
9. Sau E4, tải `final.zip` và giải nén vào thư mục gốc repo Windows. ZIP có `report/` và `work/`; giữ report/evidence, ảnh và report/REPORT_DRAFT.md. Điền phân tích cá nhân rồi chuyển báo cáo hoàn chỉnh vào REPORT.md.

Không cần bộ đồng bộ Windows hoặc Auto Save của VS Code. Hướng dẫn VS Code trong các file cũ đã được thay thế bằng quy trình này.

Checkpoint trọng số nằm ở `/content/lab-day10/work/runs/`, không có trong ZIP kết quả nhẹ. Muốn giữ adapter/projector, tải riêng từ bảng Files trước khi ngắt runtime. Ổ `/content` là tạm và có thể mất khi phiên Colab kết thúc.

Phiên chạy trực tiếp ngày 14/10/2026 dùng Transformers 5.19.0 trên T4. Với phiên bản này, inference truyền `processor_kwargs={"padding": True}` để pad batch. Danh sách bỏ qua lượng tử hóa được tạo từ tên đầy đủ của từng lớp Linear trong visual và lm_head; chỉ ghi tên cha `visual` không đủ. Mã nhúng trong notebook local đã cập nhật hai điểm này. Inference batch 4 đã được kiểm tra trên 8 ảnh validation rồi chạy đủ 311 ảnh E0; batch train vẫn là 1, accumulation 8.

Trong lần chạy đầu, kiểm tra ảnh/JSON val trước khi tiếp tục test. Nếu OOM, chỉnh batch 1 và grad accumulation 8 nhất quán cho các setup so sánh. Không tự chạy lại run vào cùng directory khi có FileExistsError; kiểm tra checkpoint đã hoàn thành chưa.

Trước khi nộp, tại repo Windows chạy `python scripts/check_submission.py`, hoàn thiện báo cáo và ≥5 commit theo tiến độ thực tế. Không commit data, work, khóa hoặc trọng số. Chưa có evidence GPU thì bài chưa hoàn thành để nộp.
