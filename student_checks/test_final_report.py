"""Report guards tested with explicitly synthetic, temporary evidence only."""
import json
from pathlib import Path

import pytest

from lab.artifacts import illustrate
from lab.final_report import build
from scripts.labcommon import write_jsonl
from scripts.score import main as score_main
from student_checks.test_artifacts import synthetic_data


def test_final_report_requires_complete_consistent_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    synthetic_data(Path('data'))
    predictions = [{'image_id': f'test-{i}', 'raw': '[]'} for i in range(6)]
    write_jsonl(Path('work/e0.jsonl'), predictions)
    for name in ('e0', 'e1', 'e2', 'e3', 'e4_data_size_0.1', 'e4_data_size_1.0'):
        score_main(['--pred', 'work/e0.jsonl', '--name', name,
            '--out', f'report/evidence/{name}.json', '--n-boot', '20',
            '--meta', 'trainable_params=100', '--meta', 'peak_vram_gb=1.0',
            '--meta', 'train_minutes=1.0', '--meta', 'inference_minutes=2.0'])
    for name in ('e1', 'e2', 'e3', 'e4_data_size_0.1'):
        folder = Path(f'work/runs/{name}')
        folder.mkdir(parents=True)
        (folder / 'run.json').write_text(json.dumps({
            'completed_steps': 300, 'dataset_version': 'synthetic-test-only',
            'config': {'name': name, 'lr': .0002}, 'train_fraction': .1 if name.startswith('e4') else 1.,
            'trainable_params': 100, 'peak_vram_gb': 1., 'train_minutes': 1.,
            'matched_modules': ['fixture-only'], 'architecture': {'total': 200},
            'diagnostic': {'vision_tokens': 220, 'supervised_tokens': 12},
            'train_image_ids': ['train-0']}), encoding='utf-8')
        if name in ('e1', 'e2', 'e3'):
            output = Path(f'work/validation/{name}.json')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(Path(f'report/evidence/{name}.json').read_text())
    illustrate('data', 'work/e0.jsonl', 'test', out='report/figures/predictions')
    report = build('SYNTHETIC TEST AUTHOR')
    content = report.read_text(encoding='utf-8')
    assert '<điền' not in content
    assert content.count('### Câu ') == 10
    assert 'synthetic-test-only' in content
    assert 'chưa đạt' in content
    run_path = Path('work/runs/e2/run.json')
    run = json.loads(run_path.read_text())
    run['completed_steps'] = 100
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='Incomplete run'):
        build('SYNTHETIC TEST AUTHOR')
    assert report.read_text(encoding='utf-8') == content
