# Chạy một notebook trong VS Code, không dùng Drive

**Đã chuyển sang Colab web theo yêu cầu mới.** Notebook hiện dùng upload/download của Colab web, không cần bộ đồng bộ. Làm theo [RUN_COLAB_WEB.md](RUN_COLAB_WEB.md); nội dung bên dưới là hướng dẫn VS Code cũ.

## Thao tác của bạn

1. Mở `lab_day10.ipynb` mới nhất trong VS Code.
2. Chọn kernel Colab có GPU. Không chọn Python local Windows.
3. Bật **File → Auto Save**. Nếu không bật, nhấn Ctrl+S sau mỗi cell xuất kết quả.
4. Chạy từng cell từ trên xuống. Không cần upload bundle hay mount Drive: cell đầu chứa mã nguồn đã nhúng và giải nén vào `/content/lab-day10`.
5. Nhập Kaggle credentials khi cell yêu cầu, hoặc upload ZIP dataset trực tiếp vào `/content` bằng extension và đổi prepare sang `--zip /content/road-damage.zip`.

Sau mỗi run, notebook đóng gói evidence, hình, raw predictions và run logs vào output. Bộ nhận local lấy gói từ notebook đã lưu, kiểm tra SHA256, rồi giải nén vào repo Windows:

- `report/evidence/`: bằng chứng score.py;
- `report/figures/`: ảnh GT/prediction;
- `report/REPORT_DRAFT.md`: báo cáo nháp sau E4;
- `work/`: raw predictions, validation và run logs;
- `work/downloads/`: ZIP riêng từng giai đoạn.

Kết quả không được coi là đã về máy chỉ vì cell train kết thúc. Kiểm tra file local và `work/local-sync/status.json`: `received_stages` phải chứa giai đoạn vừa chạy.

## Bộ đồng bộ trên Windows

Codex đã khởi động bộ nhận trong phiên làm việc này. Nó chạy ẩn, chỉ theo dõi file notebook trong repo; không mở server hoặc dùng mạng.

Sau khi restart Windows hoặc bộ nhận dừng, khởi động lại **một lần** trong terminal local:

```powershell
& .\lab\start_sync.ps1
```

Nếu PowerShell chặn script, dùng terminal local chạy:

```powershell
python -m lab.local_sync --notebook lab_day10.ipynb
```

Giữ terminal đó chạy. Nếu chỉ muốn lấy kết quả một lần từ notebook đã lưu:

```powershell
python -m lab.local_sync --notebook lab_day10.ipynb --once
```

Để dừng bộ nhận ẩn:

```powershell
New-Item -ItemType File -Force work/local-sync/STOP
```

Không clear output trước khi notebook được lưu và kết quả đã về máy. Output chứa một gói ZIP dạng base64; không chứa secret hoặc trọng số. Clear output sau khi đồng bộ để notebook nhẹ trước commit. Mỗi ZIP snapshot chứa các kết quả đã có đến thời điểm xuất.

## Trọng số và mất runtime

Tự đồng bộ áp dụng cho **kết quả, ảnh và log**, không áp dụng cho adapter/projector/optimizer hàng trăm MB–GB. Checkpoint nằm tại `/content/lab-day10/work/runs/<run>/step-100`, `step-200`, `step-300`.

Muốn giữ trọng số để suy luận lại, dùng Download của extension trong cây file Colab để tải adapter, projector.pt và run.json về `work/` local trước khi ngắt runtime. Không commit trọng số.

Ổ `/content` là tạm. Bộ nhận local không cứu được dữ liệu chưa xuất trước lúc Colab ngắt. Lệnh train hiện không tự resume từ training_state; run bị ngắt trước 300 bước cần xử lý riêng hoặc chạy lại vào directory mới.

## Hạn chế frontend

`google.colab.files.download()` có hạn chế trong VS Code, vì vậy notebook dùng output đã lưu và bộ nhận local. Mã này đã được kiểm thử với notebook JSON và ZIP tổng hợp; đường truyền output thật từ kernel VS Code Colab cần kiểm tra bằng giai đoạn `data-audit` đầu tiên. Nếu VS Code không lưu MIME tùy chỉnh, có thể tải ZIP dự phòng ở `work/exports/` bằng Download của extension.

Tài liệu chính thức: https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds#filesdownload

## Hoàn thiện bài nộp

Sau E4, mở REPORT_DRAFT.md, điền họ tên và các phân tích cá nhân dựa trên output/ảnh, rồi đưa bản hoàn chỉnh vào REPORT.md. Chạy `python scripts/check_submission.py` trong terminal Windows. Cần ≥5 commit theo tiến độ thật, đủ evidence E0–E3 và hai giá trị E4. Không đưa data, secret hoặc trọng số vào Git.
