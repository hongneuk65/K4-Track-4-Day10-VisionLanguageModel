"""Transfer measured results through saved notebook output, without Drive."""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

MIME = "application/vnd.road-lab.archive+json"


def result_archive(root="."):
    root = Path(root)
    files = list((root/"report").rglob("*"))
    files += list((root/"work").glob("*.jsonl")) + list((root/"work").glob("*.run.json"))
    files += list((root/"work/validation").glob("*.json"))
    files += list((root/"work/runs").glob("*/run.json"))
    completed_report = root / "REPORT.md"
    if completed_report.is_file() and "<điền" not in completed_report.read_text(encoding="utf-8"):
        files.append(completed_report)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            if file.is_file():
                archive.write(file, file.relative_to(root).as_posix())
    return buffer.getvalue()


def export_results(stage, root="."):
    from IPython.display import display
    content = result_archive(root)
    payload = {"version": 1, "stage": stage, "sha256": hashlib.sha256(content).hexdigest(),
               "archive_b64": base64.b64encode(content).decode("ascii")}
    # text/plain is rendered; the custom MIME is retained in the saved .ipynb.
    display({"text/plain": f"Kết quả {stage} đã đóng gói ({len(content)/1024:.0f} KB). "
                           "Lưu notebook (Ctrl+S hoặc Auto Save) để đồng bộ về máy.",
             MIME: payload}, raw=True)
    directory = Path(root)/"work/exports"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/f"{stage}.zip"
    destination.write_bytes(content)
    return destination


def export_colab_results(stage, root="."):
    """Request a browser download on Colab web; retain a ZIP if blocked."""
    from google.colab import files
    directory = Path(root)/"work/exports"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/f"{stage}.zip"
    destination.write_bytes(result_archive(root))
    print(f"ZIP kết quả: {destination}. Cho phép trình duyệt tải file.")
    try:
        files.download(str(destination))
    except Exception as exc:
        print(f"Tải tự động chưa thành công: {exc}. Tải ZIP từ bảng Files của Colab.")
    return destination
