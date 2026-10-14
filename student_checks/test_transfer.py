import base64
import hashlib
import io
import json
import zipfile

import pytest

from lab.transfer import MIME, result_archive
from lab.local_sync import extract_payload, receive_once


def payload_for(content, stage='e1-val'):
    return {'version': 1, 'stage': stage, 'sha256': hashlib.sha256(content).hexdigest(),
            'archive_b64': base64.b64encode(content).decode('ascii')}


def test_saved_notebook_transfers_evidence_figures_and_logs_without_weights(tmp_path):
    remote = tmp_path/'remote'
    (remote/'report/evidence').mkdir(parents=True)
    (remote/'report/evidence/e1.json').write_text('{"micro":{"f1":0.7}}')
    (remote/'work/runs/e1').mkdir(parents=True)
    (remote/'work/runs/e1/run.json').write_text('{"completed_steps":300}')
    (remote/'work/runs/e1/projector.pt').write_bytes(b'not transferred')
    (remote/'.env').write_text('SECRET=not transferred')
    (remote/'work/e1_test.jsonl').write_text('{"image_id":"a","raw":"[]"}\n')
    content = result_archive(remote)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert set(archive.namelist()) == {'report/evidence/e1.json', 'work/runs/e1/run.json', 'work/e1_test.jsonl'}
    local = tmp_path/'local'
    local.mkdir()
    notebook = local/'lab_day10.ipynb'
    notebook.write_text(json.dumps({'cells': [{'outputs': [{'data': {MIME: payload_for(content)}}]}]}))
    seen = set()
    assert receive_once(notebook, seen)[0]['stage'] == 'e1-val'
    assert (local/'report/evidence/e1.json').exists()
    assert (local/'work/runs/e1/run.json').exists()
    assert len(list((local/'work/downloads').glob('*.zip'))) == 1
    assert receive_once(notebook, seen) == []


@pytest.mark.parametrize('name', ['../outside.json', 'report/../../outside.json', 'lab/runtime.py', 'work/evil.exe'])
def test_receiver_rejects_unapproved_paths(tmp_path, name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(name, b'bad')
    with pytest.raises(ValueError):
        extract_payload(payload_for(buffer.getvalue()), tmp_path)
    assert not (tmp_path/'outside.json').exists()


def test_receiver_rejects_hash_mismatch(tmp_path):
    payload = payload_for(b'archive')
    payload['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        extract_payload(payload, tmp_path)


def test_colab_export_downloads_and_preserves_zip_on_browser_error(tmp_path, monkeypatch):
    import sys
    import types
    from lab.transfer import export_colab_results
    (tmp_path/'report').mkdir()
    (tmp_path/'report/result.json').write_text('{}')
    called = []
    colab = types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(download=lambda file: called.append(file))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    path = export_colab_results('final', tmp_path)
    assert called == [str(path)]
    assert path.exists()
    def fail(file):
        raise RuntimeError('download blocked')
    colab.files.download = fail
    assert export_colab_results('retry', tmp_path).exists()
