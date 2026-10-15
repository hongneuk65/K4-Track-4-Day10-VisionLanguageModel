# Trạng thái tạm dừng — 14/10/2026

**Cập nhật: đã tiếp tục theo yêu cầu ngày 14/10/2026.** Kernel T4 cũ còn dataset và checkpoint E1. E2 đã khởi động lại từ base; E3, chấm test, E4 và báo cáo đã được xếp chạy sau E2. Nội dung dưới đây ghi trạng thái của bản sao lưu trước khi tiếp tục.

Học viên: Nguyễn Văn Hồng. Đã interrupt Colab và hủy hàng đợi E2–E4 theo yêu cầu. Không có run GPU nào tiếp tục chạy. Không xóa runtime trước khi sao lưu.

## Đã lưu an toàn trên máy

- `work/pause-backup.zip`: 127858904 byte. SHA-256: `e37d0b11c3e965125208f95fbfbbfce948add054a399efcf324bae3e3e8b4118`. Checksum khớp bản Colab; CRC cả ZIP ngoài và results.zip bên trong đã kiểm tra.
- E0 test đủ 311 ảnh: `report/evidence/e0.json`, raw `work/e0.jsonl`, nhật ký inference `work/e0.run.json`.
- E1 đủ 300 bước: `work/runs/e1/step-300/adapter/` và processor, run.json; đã giải nén vào repo local. E1 cập nhật 33030144 tham số; peak VRAM 5.503240192 GB; train 51.3519941364 phút.
- E1 validation đủ 276 ảnh: `work/validation/e1.json`, `work/e1_val.jsonl` và sidecar. Chưa chấm E1 test.
- Dataset ZIP gốc còn ở `work/road-damage.zip` trên máy; dataset_version `d10-21e4070301`.

E2 đã interrupt khi log gần nhất là bước 20/300; chưa đến checkpoint 100. Không có checkpoint E2 để resume. Lần sau chạy E2 lại từ base. E3/E4 chưa chạy. REPORT.md chưa hoàn chỉnh; chưa coi bài đã đủ điều kiện nộp.

## Khôi phục lần sau

1. Mở phiên Colab T4 mới; dùng notebook local mới nhất, chạy bootstrap và cài dependency.
2. Upload `work/pause-backup.zip` vào `/content/pause-backup.zip`. Không upload trọng số lên GitHub.
3. Tại `/content/lab-day10`, chạy:

```python
cmd('-m', 'lab.restore_pause', '--zip', '/content/pause-backup.zip',
    '--sha256', 'e37d0b11c3e965125208f95fbfbbfce948add054a399efcf324bae3e3e8b4118')
```

4. Chuẩn bị lại dataset từ ZIP trên máy hoặc link Kaggle public và kiểm tra version khớp.
5. E0 và E1 sẽ được tái sử dụng khi sidecar/config/dataset khớp. Tiếp tục train E2, E3; chọn setup bằng validation; chấm E1–E3 test; chạy E4 và hoàn thiện báo cáo.

Backup không chứa optimizer của E1 vì run E1 đã hoàn tất. Có đầy đủ adapter cần cho inference; không hứa resume giữa một run bị ngắt.

## Chuyển tài khoản ngày 15/10/2026

- Runtime cũ đã thu hồi. Log notebook xác nhận E3 val F1=0.3492333901192504; trọng số E2/E3 chưa có trong backup local nên không thể phục hồi từ log.
- Tài khoản mới hosan7182@gmail.com đã kết nối T4 và mount Drive sau xác nhận quyền đọc/ghi toàn bộ Drive của người dùng.
- Đã upload và kiểm tra SHA256 gói 322325074 byte: 96bc4015f3bfe564234ddef4ac7686d283d4c8234e2d08a5111fd42de2442d93.
- Đã lưu `/content/drive/MyDrive/lab-day10/lab-account-transfer.zip`, khôi phục dataset d10-21e4070301 và E1 300 bước.
- Đã sao lưu E1 vào `MyDrive/lab-day10/runs/e1`, splits vào `splits`, kết quả vào `results/restored-e1.zip` (2464756 byte).
- E2 đang khởi động lại. Cell tiếp theo điều phối E3, test E1-E3, E4 và báo cáo; chưa xác nhận các phần đó hoàn thành.
- Train dùng `--backup /content/drive/MyDrive/lab-day10/runs` mỗi 100 bước. Kết quả sau đánh giá lưu trong `results/`; báo cáo cuối lưu `REPORT.md` tại thư mục Drive đó.
