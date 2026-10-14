"""Integration uses synthetic fixtures only, isolated from submission evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from lab.artifacts import illustrate, report_draft
from lab.data import audit_data
from lab.__main__ import score_file
from scripts.labcommon import write_jsonl
from scripts.score import main as score_main


def synthetic_data(root):
    (root / 'images').mkdir(parents=True)
    counts = {}
    for split in ('train', 'val', 'test'):
        entries = []
        for i in range(6):
            identifier = f'{split}-{i}'
            Image.new('RGB', (640, 360), 'gray').save(root / 'images' / f'{identifier}.jpg')
            entries.append({'image_id': identifier, 'image_path': f'images/{identifier}.jpg',
                'width': 640, 'height': 360, 'session': split,
                'objects': [{'label': 'pothole', 'yolo': [.5, .5, .2, .2]}]})
        write_jsonl(root / 'splits' / f'{split}.jsonl', entries)
        counts[split] = {'images': 6, 'boxes': {'pothole': 6, 'crack': 0, 'manhole': 0}}
    (root / 'splits/meta.json').write_text(json.dumps({'dataset_version': 'synthetic-test-only', 'counts': counts}))


def test_audit_rejects_session_leakage(tmp_path):
    synthetic_data(tmp_path / 'data')
    assert audit_data(tmp_path / 'data')['counts']['test']['images'] == 6
    file = tmp_path / 'data/splits/val.jsonl'
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    rows[0]['session'] = 'train'
    write_jsonl(file, rows)
    with pytest.raises(ValueError, match='Leakage'):
        audit_data(tmp_path / 'data')


def test_figures_and_report_use_real_fixture_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    synthetic_data(Path('data'))
    predictions = [{'image_id': f'test-{i}', 'raw': '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'
                    if i % 2 == 0 else '[]'} for i in range(6)]
    write_jsonl(Path('work/pred.jsonl'), predictions)
    names = ['e0', 'e1', 'e2', 'e3', 'e4_data_size_0.1', 'e4_data_size_1.0']
    for name in names:
        score_main(['--pred', 'work/pred.jsonl', '--name', name, '--out', f'report/evidence/{name}.json',
                    '--n-boot', '20', '--meta', 'trainable_params=100', '--meta', 'peak_vram_gb=1.0',
                    '--meta', 'train_minutes=1.0'])
    for name in ('e1', 'e2', 'e3'):
        run_dir = Path('work/runs') / name
        run_dir.mkdir(parents=True)
        (run_dir/'run.json').write_text(json.dumps({'trainable_params': 100, 'architecture': {'total': 200},
                         'diagnostic': {'vision_tokens': 220, 'supervised_tokens': 12}}))
    captions = illustrate('data', 'work/pred.jsonl', 'test', out='report/figures/predictions')
    assert len(captions) == 6
    assert {c['category'] for c in captions} == {'correct', 'error'}
    for caption in captions:
        with Image.open(caption['file']) as image:
            assert image.size == (640, 412)
    draft = report_draft('data', 'work/runs')
    text = Path(draft).read_text(encoding='utf-8')
    assert '0.6667' in text
    assert 'synthetic-test-only' in text
    assert text.count('### Câu ') == 10
    assert '<điền' in text  # Never pretend analysis and personal identity are complete.


def test_score_rejects_partial_prediction_file(tmp_path):
    synthetic_data(tmp_path/'data')
    write_jsonl(tmp_path/'pred.jsonl', [{'image_id': 'test-0', 'raw': '[]'}])
    args = SimpleNamespace(data=str(tmp_path/'data'), pred=str(tmp_path/'pred.jsonl'), split='test',
                           name='E1', out=str(tmp_path/'out.json'), coord='norm1000', run=None)
    with pytest.raises(ValueError, match='every image'):
        score_file(args)
