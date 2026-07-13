from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    package_path = _PACKAGE_ROOT / candidate
    if package_path.exists():
        return package_path

    repo_path = _REPO_ROOT / candidate
    if repo_path.exists():
        return repo_path

    return repo_path


@contextmanager
def get_data_path(path: str | Path) -> Iterator[Path]:
    yield _resolve_data_path(path)


def is_data_file(path: str | Path) -> bool:
    return _resolve_data_path(path).is_file()
