"""Pack source only; build_notebook also embeds these files for remote execution."""
from pathlib import Path
import zipfile


def source_files(root, include_notebook=True):
    files = [p for folder in ("lab", "scripts", "tests", "student_checks", "configs")
             for p in (root/folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".yaml", ".ps1")]
    files += [root/name for name in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md",
              "RUN_COLAB_VSCODE.md", "RUN_COLAB_NO_DRIVE.md", "RUN_COLAB_WEB.md", "requirements-colab.txt", "pytest.ini", ".gitignore")]
    if include_notebook:
        files.append(root/"lab_day10.ipynb")
    return sorted(files)


def package():
    root = Path(__file__).resolve().parents[1]
    destination = root / "work/lab-colab-bundle.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in source_files(root):
            archive.write(file, file.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    package()
