"""Execute the real first cell in an empty folder, without requiring Colab/GPU."""
import ast
import json
import subprocess
import sys
from pathlib import Path


def test_first_cell_restores_notebook_and_creates_all_directories(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root/'lab_day10.ipynb').read_text(encoding='utf-8'))
    bootstrap = ast.parse(''.join(document['cells'][2]['source']))
    for node in bootstrap.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets):
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                  args=[ast.Constant(str(tmp_path/'fresh'))], keywords=[])
        if isinstance(node, ast.If) and 'sys.platform' in ast.unparse(node.test):
            node.test = ast.Constant(False)  # Only bypass the Colab platform guard in this test.
    monkeypatch.chdir(tmp_path)
    original_path = list(sys.path)
    try:
        exec(compile(ast.fix_missing_locations(bootstrap), '<real-notebook-bootstrap>', 'exec'), {})
    finally:
        sys.path[:] = original_path
    extracted = tmp_path/'fresh'
    assert (extracted/'work').is_dir()
    assert (extracted/'report/evidence').is_dir()
    notebook = json.loads((extracted/'lab_day10.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    assert notebook['cells'][2]['source'][0] == document['cells'][2]['source'][0]
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q',
        'tests/test_configs_and_docs.py', 'tests/test_score.py::test_cli_end_to_end_writes_json_with_meta',
        '--basetemp=work/fresh-tests', '--tb=short'], cwd=extracted, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
