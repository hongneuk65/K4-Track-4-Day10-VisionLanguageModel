"""Restore completed E1 and measured outputs without replacing current source."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def restore(path, root='.', expected_sha256=None):
    root = Path(root).resolve()
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Backup SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(content)) as outer:
        if outer.testzip() is not None:
            raise ValueError('Corrupt backup ZIP')
        run = json.loads(outer.read('work/runs/e1/run.json'))
        if run['completed_steps'] != 300:
            raise ValueError('Backup E1 is incomplete')
        with zipfile.ZipFile(io.BytesIO(outer.read('results.zip'))) as results:
            if results.testzip() is not None:
                raise ValueError('Corrupt results ZIP')
            selections = [(outer, n) for n in outer.namelist() if n.startswith('work/runs/e1/')]
            selections += [(results, n) for n in results.namelist() if n.startswith(('work/', 'report/'))]
            for archive, name in selections:
                destination = (root / name).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError(f'Unsafe backup path: {name}')
                if archive.getinfo(name).is_dir():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
    return {'sha256': digest, 'dataset_version': run['dataset_version'],
            'e1_completed_steps': run['completed_steps'], 'files': len(selections)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True)
    parser.add_argument('--root', default='.')
    parser.add_argument('--sha256')
    args = parser.parse_args()
    print(json.dumps(restore(args.zip, args.root, args.sha256), indent=2))
