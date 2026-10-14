"""Build the executable notebook. Run locally after changing notebook cells."""
import json
import copy
import base64
import io
import zipfile
from pathlib import Path
from lab.package_colab import source_files


def build():
    buffer = io.BytesIO()
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in source_files(root, include_notebook=False):
            archive.write(file, file.relative_to(root).as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    cells = []
    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.strip().splitlines(keepends=True)})
    md("# Lab Day 10 — bản Colab web mới\n"
       "Phiên bản: COLAB-WEB-2. Upload bản mới này và chạy từ đầu trong phiên T4.\n"
       "Mã học viên nằm trong `lab/`; công cụ đề trong `scripts/` và `tests/` giữ nguyên.\n"
       "Chạy từng cell, đọc output rồi tiếp tục. E0–E4 cần GPU thật; không có số liệu mẫu giả.\n"
       "Mở notebook trên https://colab.research.google.com/ → File → Upload notebook.\n"
       "Runtime → Change runtime type → T4 GPU. Mã nguồn đã nhúng; không cần bundle hoặc Google Drive.\n"
       "Sau mỗi giai đoạn notebook yêu cầu trình duyệt tải ZIP kết quả. Cho phép tải nhiều file nếu trình duyệt hỏi.")
    md("## 0. Môi trường (Colab T4)\n"
       "Cell đầu giải nén mã nhúng, tạo đầy đủ thư mục và notebook trên disk trước khi chạy test.\n"
       "Checkpoint trọng số giữ ở Colab; kết quả nhẹ (evidence, ảnh, predictions, logs) tự xuất sau mỗi giai đoạn.")
    code("SOURCE_B64 = " + repr(payload) + '''
from pathlib import Path
import os, sys, subprocess, json, shutil, zipfile, io, base64
ROOT = Path('/content/lab-day10')
if sys.platform == 'win32' or not Path('/content').is_dir():
    raise RuntimeError('Mở notebook trên Colab web và chọn T4 GPU.')
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as archive:
    for member in archive.infolist():
        target = (ROOT / member.filename).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(ROOT)
# Restore the executable notebook from its finite embedded snapshot.
notebook_path = ROOT / 'lab_day10.ipynb'
notebook_copy = json.loads(notebook_path.read_text(encoding='utf-8'))
notebook_copy['cells'][2]['source'][0] = 'SOURCE_B64 = ' + repr(SOURCE_B64) + '\\n'
notebook_path.write_text(json.dumps(notebook_copy, ensure_ascii=False, indent=2), encoding='utf-8')
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.colab_session import ensure_directories, command as cmd
ensure_directories(ROOT)
print('Khởi tạo thành công:', ROOT)
print('work/:', (ROOT/'work').is_dir(), '| notebook:', notebook_path.is_file())''')
    code('''
cmd('-c', "import torch; assert torch.cuda.is_available(), 'Colab runtime chưa có GPU'; print('GPU:', torch.cuda.get_device_name(0))")
cmd('-m', 'pip', 'install', '-r', 'requirements-colab.txt')
cmd('-m', 'pip', 'install', 'pytest')
cmd('-m', 'lab', 'doctor')
cmd('-m', 'pytest', '-q', 'tests', 'student_checks', '--basetemp=work/notebook-tests', '--tb=short', '-ra')
from lab.transfer import export_colab_results
from lab.colab_session import train_once, predict_once, evaluate, completed_checkpoint
def backup_results(stage='results'):
    return export_colab_results(stage, ROOT)
print('Sẵn sàng. Không sử dụng Google Drive.')''')
    code('''# Mặc định upload ZIP dataset, không cần Kaggle credentials.
# Đổi DATA_MODE thành 'kaggle' nếu muốn tải bằng tài khoản Kaggle.
DATA_MODE = 'zip'
DATA_ZIP = Path('/content/road-damage.zip')
from getpass import getpass
from google.colab import files, userdata
if DATA_MODE == 'zip':
    if not DATA_ZIP.exists():
        print('Chọn ZIP DATASET road damage, không chọn ZIP mã nguồn.')
        files.upload_file(str(DATA_ZIP))
    if not zipfile.is_zipfile(DATA_ZIP):
        raise ValueError('Dataset được chọn không phải ZIP hợp lệ')
elif DATA_MODE == 'kaggle':
    for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        if not os.environ.get(key):
            try:
                value = userdata.get(key)
            except Exception:
                value = None
            os.environ[key] = value or getpass(f'{key}: ')
else:
    raise ValueError('DATA_MODE phải là zip hoặc kaggle')
# Không print environment hoặc secret.''')
    md("## 1. Dữ liệu: tải và chia split\n"
       "Mặc định chọn ZIP dataset ở cell trước; có thể tải ZIP từ link Kaggle trong README. Chế độ kaggle là tùy chọn.")
    code('''if not Path('data/splits/meta.json').is_file():
    if DATA_MODE == 'zip':
        cmd('scripts/prepare_data.py', '--zip', DATA_ZIP)
    else:
        cmd('scripts/prepare_data.py', '--kaggle')
cmd('-m', 'lab', 'audit')
cmd('-m', 'lab', 'illustrate', '--split', 'train', '--out', 'report/figures/ground_truth')
from IPython.display import display, Image as DisplayImage
for path in sorted(Path('report/figures/ground_truth').glob('*.png')):
    display(DisplayImage(filename=str(path)))
backup_results('data-audit')''')
    md("## 2. Nhãn: YOLO → JSON thang 0–1000")
    code('''from lab.data import yolo_to_box, objects_to_answer
assert yolo_to_box('pothole', [.5,.5,.2,.2])['bbox_2d'] == [400,400,600,600]
assert yolo_to_box('pothole', [.5,.5,.2,.2], 'pixel')['bbox_2d'] == [256,144,384,216]
cmd('-m', 'lab', 'scale-probe')''')
    md("## 3. E0: zero-shot\nThử 8 ảnh val trước. Sau khi prompt hoạt động, cell tiếp theo chấm test đúng một lần.")
    code('''INFERENCE_BATCH = 1
predict_once('work/e0_smoke.jsonl', 'val', limit=8, batch_size=INFERENCE_BATCH)
from scripts.labcommon import read_jsonl
for row in read_jsonl('work/e0_smoke.jsonl')[:5]:
    print(row)''')
    code('''predict_once('work/e0.jsonl', 'test', batch_size=INFERENCE_BATCH)
cmd('-m', 'lab', 'score', '--split', 'test', '--pred', 'work/e0.jsonl', '--name', 'E0',
    '--out', 'report/evidence/e0.json')
backup_results('e0-test')''')
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss\n"
       "`lab.data.encode_example` assert prefix, mask prompt; `collate_examples` pad labels=-100 và cat patch.\n"
       "Mỗi run train kiểm tra 220 token ảnh, decode labels và forward loss trước khi train.")
    code('''cmd('-m', 'pytest', '-q', 'student_checks', '--basetemp=work/mask-checks')''')
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3\n"
       "Các setup bắt đầu độc lập từ base model. Dùng cấu hình đề; nếu OOM chọn batch 1, accumulation 8 cho cả ba setup.")
    code('''BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_STEPS = 300
assert BATCH_SIZE * GRAD_ACCUM == 8
assert MAX_STEPS == 300
def checkpoint_for(name):
    checkpoint, _ = completed_checkpoint(name)
    return checkpoint
def train_one(name, config, fraction=1.0):
    train_once(name, config, BATCH_SIZE, GRAD_ACCUM, fraction)
    backup_results(f'{name}-trained')
def evaluate_one(name, split):
    evaluate(name, split, INFERENCE_BATCH)
    backup_results(f'{name}-{split}')''')
    md("## 6. Vòng huấn luyện\nChạy từng cell. Nếu cần gỡ lỗi, dùng output directory mới; không ghi đè evidence/checkpoint.")
    for name in ("e1", "e2", "e3"):
        code(f"train_one('{name}', 'configs/{name}.yaml')\nevaluate_one('{name}', 'val')")
    md("## 7. Suy luận và chấm E1–E3\n"
       "Xem validation và loss log trước khi chạy cell này. Nạp adapter + projector trong tiến trình mới.")
    code('''validation = {n: json.loads(Path(f'work/validation/{n}.json').read_text()) for n in ('e1','e2','e3')}
BEST = max(validation, key=lambda n: validation[n]['micro']['f1'])
print('Setup chọn bằng validation:', BEST)
for name, evidence in validation.items():
    print(name, 'val F1:', evidence['micro']['f1'], 'valid:', evidence['parse_valid_rate'])
for name in ('e1', 'e2', 'e3'):
    evaluate_one(name, 'test')''')
    md("## 8. E4 (ablation data size) và báo cáo\n"
       "Đổi đúng tỷ lệ train; 100% dùng lại setup gốc, 10% train mới từ base. Cả hai giữ 300 bước.")
    code('''train_one('e4_data_size_0.1', f'configs/{BEST}.yaml', fraction=0.1)
evaluate_one('e4_data_size_0.1', 'val')
evaluate_one('e4_data_size_0.1', 'test')
cmd('-m', 'lab', 'score', '--split', 'test', '--pred', f'work/{BEST}_test.jsonl',
    '--name', 'E4_DATA_SIZE_1.0', '--run', checkpoint_for(BEST) / 'run.json',
    '--out', 'report/evidence/e4_data_size_1.0.json')
cmd('-m', 'lab', 'illustrate', '--split', 'test', '--pred', f'work/{BEST}_test.jsonl',
    '--out', 'report/figures/predictions')
cmd('-m', 'lab', 'report')
backup_results('e4-report')
for path in sorted(Path('report/figures/predictions').glob('*.png')):
    display(DisplayImage(filename=str(path)))''')
    md("## 9. Báo cáo, backup và nộp\n"
       "Cell cuối tạo `REPORT.md` từ evidence đầy đủ. Đọc báo cáo và đối chiếu ảnh trước khi nộp.\n"
       "ZIP kết quả được tải qua trình duyệt; giải nén vào repo local, giữ nguyên scripts/tests và commit theo tiến độ thật.\n"
       "E5 tùy chọn: `lab.reward.iou_reward` đã có test; chưa chạy GRPO, không tính là evidence bonus.")
    code('''STUDENT_NAME = 'Nguyễn Văn Hồng'
cmd('-m', 'lab.final_report', '--author', STUDENT_NAME)
print(Path('REPORT.md').read_text(encoding='utf-8'))
archive_path = backup_results('final')
print('ZIP dự phòng trên Colab:', archive_path)
print('Giải nén final.zip vào thư mục gốc repo Windows; ZIP chứa report/ và work/.')
print('Nếu trình duyệt chặn tải, mở Files bên trái Colab → work/exports → final.zip → Download.')
print('Trọng số ở /content/lab-day10/work/runs. Nếu cần giữ, tải riêng trước khi ngắt runtime.')
print('Chạy ở repo local sau khi hoàn thiện REPORT và commit: python scripts/check_submission.py')''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "lab_day10.ipynb"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"lab-cell-{i:03d}"
    # Avoid recursively embedding the full base64 archive into itself. The snapshot
    # has the same cells but an empty payload; bootstrap restores the actual one.
    snapshot = copy.deepcopy(notebook)
    snapshot["cells"][2]["source"][0] = "SOURCE_B64 = ''\n"
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("lab_day10.ipynb", json.dumps(snapshot, ensure_ascii=False, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    notebook["cells"][2]["source"][0] = "SOURCE_B64 = " + repr(payload) + "\n"
    serialized = json.dumps(notebook, ensure_ascii=False, indent=2)+"\n"
    (root/'lab_day10.ipynb').write_text(serialized, encoding='utf-8')
    (root/'lab_day10_colab.ipynb').write_text(serialized, encoding='utf-8')


if __name__ == "__main__":
    build()
