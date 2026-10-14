"""Notebook orchestration with live logs and checks before reusing results."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def command(*args):
    argv = [sys.executable, *map(str, args)]
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Lệnh thất bại (exit {code}); xem log phía trên: {' '.join(argv)}")


def ensure_directories(root):
    root = Path(root)
    for name in ("work", "work/runs", "work/validation", "work/exports", "report/evidence", "report/figures"):
        (root/name).mkdir(parents=True, exist_ok=True)


def completed_checkpoint(name):
    directory = Path("work/runs")/name
    run = json.loads((directory/"run.json").read_text(encoding="utf-8"))
    if run["completed_steps"] != 300:
        raise ValueError(f"{name}: run chưa đủ 300 bước; không dùng làm evidence nộp bài")
    checkpoint = directory/"step-300"
    if not (checkpoint/"adapter/adapter_config.json").exists():
        raise FileNotFoundError(f"Thiếu adapter: {checkpoint}")
    if run["config"]["train_projector"] and not (checkpoint/"projector.pt").exists():
        raise FileNotFoundError(f"Thiếu projector: {checkpoint}")
    return checkpoint, run


def predict_once(path, split, checkpoint=None, limit=None, batch_size=1, max_new_tokens=256):
    from scripts.labcommon import read_jsonl
    path = Path(path)
    expected = read_jsonl(Path("data/splits")/f"{split}.jsonl")
    if limit is not None:
        expected = expected[:limit]
    if path.exists():
        summary_path = path.with_suffix(".run.json")
        if not summary_path.exists():
            raise ValueError(f"Prediction bị ngắt: {path}. Giữ file này và chọn tên output mới.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = read_jsonl(path)
        valid = (len(rows) == len(expected)
                 and {r['image_id'] for r in rows} == {r['image_id'] for r in expected}
                 and summary['split'] == split and summary['coord'] == 'norm1000'
                 and summary['checkpoint'] == (str(checkpoint) if checkpoint else None)
                 and summary['max_new_tokens'] == max_new_tokens
                 and summary['prediction_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest())
        if not valid:
            raise ValueError(f"Không tái sử dụng prediction không nhất quán: {path}")
        print(f"Tái sử dụng prediction hoàn chỉnh: {path}")
        return
    args = ["-m", "lab", "predict", "--split", split, "--out", path,
            "--batch-size", batch_size, "--max-new-tokens", max_new_tokens]
    if checkpoint:
        args += ["--checkpoint", checkpoint]
    if limit is not None:
        args += ["--limit", limit]
    command(*args)


def train_once(name, config, batch_size=1, grad_accum=8, fraction=1.0):
    import yaml
    directory = Path("work/runs")/name
    if (directory/"run.json").exists():
        checkpoint, run = completed_checkpoint(name)
        expected = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        expected.update(name=name.upper(), batch_size=batch_size, grad_accum=grad_accum, max_steps=300)
        if run['config'] != expected or run['train_fraction'] != fraction or run['coord'] != 'norm1000':
            raise ValueError(f"Cấu hình {name} đã thay đổi; dùng run directory mới")
        dataset = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))
        if run['dataset_version'] != dataset['dataset_version']:
            raise ValueError('Checkpoint thuộc dataset khác')
        print(f"Tái sử dụng run hoàn chỉnh: {checkpoint}")
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Run chưa hoàn chỉnh: {directory}; dùng tên run mới để tránh ghi đè")
    command("-m", "lab", "train", "--config", config, "--out", directory,
            "--name", name.upper(), "--fraction", fraction, "--batch-size", batch_size,
            "--grad-accum", grad_accum, "--max-steps", 300)


def evaluate(name, split, inference_batch=1):
    checkpoint, _ = completed_checkpoint(name)
    pred = Path(f"work/{name}_{split}.jsonl")
    predict_once(pred, split, checkpoint, batch_size=inference_batch)
    evidence = f"report/evidence/{name}.json" if split == "test" else f"work/validation/{name}.json"
    command("-m", "lab", "score", "--split", split, "--pred", pred, "--name", name.upper(),
            "--run", checkpoint/"run.json", "--out", evidence)
    return evidence
