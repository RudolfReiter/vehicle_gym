"""Build the source archive and static deployment directory, deterministically."""

from pathlib import Path
import shutil
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
excluded = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "dist",
    "build",
    ".ruff_cache",
    ".idea",
}
archive = root / "docs/assets/source.zip"
try:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        )
        .decode()
        .split("\0")
    )
    files = sorted(root / name for name in set(names) if name)
except (FileNotFoundError, subprocess.CalledProcessError):
    files = sorted(root.rglob("*"))
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for file in files:
        relative = file.relative_to(root)
        if (
            not file.is_file()
            or file == archive
            or file.name.endswith("-training.pt")
            or any(p in excluded or p.endswith(".egg-info") for p in relative.parts)
        ):
            continue
        info = zipfile.ZipInfo(str(Path(root.name) / relative), (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, file.read_bytes())
shutil.copytree(root / "docs", root / "dist", dirs_exist_ok=True)
print(f"Built {archive.name} and dist/ for {root.name}.")
