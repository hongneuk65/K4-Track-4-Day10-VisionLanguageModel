"""Local receiver: watches saved notebook outputs and extracts verified results."""
import argparse
import base64
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

from lab.transfer import MIME


def extract_payload(payload, root):
    root = Path(root).resolve()
    stage = payload.get("stage", "")
    if payload.get("version") != 1 or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", stage):
        raise ValueError("Invalid archive metadata")
    content = base64.b64decode(payload["archive_b64"], validate=True)
    digest = hashlib.sha256(content).hexdigest()
    if digest != payload["sha256"]:
        raise ValueError("Archive hash mismatch")
    approved = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for entry in archive.infolist():
            target = (root/entry.filename).resolve()
            name = Path(entry.filename).as_posix()
            if not target.is_relative_to(root) or not name.startswith(("report/", "work/")):
                raise ValueError("Unsafe result path")
            if entry.is_dir():
                continue
            if target.suffix.lower() not in (".json", ".jsonl", ".md", ".png", ".jpg", ".jpeg"):
                raise ValueError("Unexpected result file type")
            approved.append((entry, target))
        for entry, target in approved:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+".sync-tmp")
            temporary.write_bytes(archive.read(entry))
            temporary.replace(target)
    saved = root/"work/downloads"/f"{stage}-{digest[:10]}.zip"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(content)
    return digest, len(approved)


def receive_once(notebook, seen):
    notebook = Path(notebook)
    document = json.loads(notebook.read_text(encoding="utf-8"))
    received = []
    for cell in document.get("cells", []):
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get(MIME)
            if not isinstance(payload, dict) or payload.get("sha256") in seen:
                continue
            digest, count = extract_payload(payload, notebook.parent)
            seen.add(digest)
            received.append({"stage": payload["stage"], "files": count, "sha256": digest})
    return received


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="lab_day10.ipynb")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    notebook = Path(args.notebook).resolve()
    folder = notebook.parent/"work/local-sync"
    folder.mkdir(parents=True, exist_ok=True)
    state_file = folder/"state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {"seen": [], "received": []}
    seen = set(state["seen"])
    last_mtime = None
    while True:
        try:
            stamp = notebook.stat().st_mtime_ns
            if stamp != last_mtime:
                new = receive_once(notebook, seen)
                if new:
                    state["received"] += new
                    state["seen"] = sorted(seen)
                    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
                last_mtime = stamp
            (folder/"status.json").write_text(json.dumps({"running": not args.once,
                "updated": time.time(), "notebook": str(notebook),
                "received_stages": [r["stage"] for r in state["received"]]}), encoding="utf-8")
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            # A save can briefly produce incomplete JSON; retry without treating it as success.
            (folder/"error.log").write_text(str(exc), encoding="utf-8")
        if args.once or (folder/"STOP").exists():
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
