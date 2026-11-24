"""Build a deterministic source archive from the public project files."""

import gzip
import hashlib
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    ".gitignore", ".gitattributes", "README.md", "pyproject.toml",
    "configs/mri_x4.yaml", "configs/mri_momentum_x4.yaml",
    "configs/mri_gsbm_mls_x4.yaml", "configs/mri_gsbm_mls_guided_x4.yaml",
    "configs/mri_momentum_mls_x4.yaml", "configs/mri_momentum_tdm_x4.yaml", "reports/mri/summary.json",
    "configs/mri_haar_direct_x4.yaml", "configs/mri_haar_bridge_x4.yaml", "configs/mri_haar_bridge_legacy_x4.yaml",
    "configs/mri_haar_sr_mls_x4.yaml",
)
TREES = {
    "src/vemgsbm": {".py"},
    "scripts": {".sh", ".sbatch"},
    "tests": {".py"},
    "tools": {".py"},
    "docs": {".md", ".txt"},
}


def source_files():
    paths = {ROOT / name for name in ROOT_FILES}
    for directory, extensions in TREES.items():
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in extensions and not path.name.startswith("."))
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Expected a regular project file: {path}")
        yield path


def add_file(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o755 if name.endswith((".sh", ".sbatch")) else 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    files = list(source_files())
    output = ROOT / "dist" / "vem_gsbm-source.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with output.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode="w") as archive:
            for path in files:
                name = path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                if path.suffix != ".png":
                    data = data.replace(b"\r\n", b"\n")
                add_file(archive, name, data)
                manifest.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
            add_file(archive, "SOURCE_SHA256.txt", "".join(manifest).encode("utf-8"))
    print(f"{output}: {len(files)} source, documentation and report files")


if __name__ == "__main__":
    main()
