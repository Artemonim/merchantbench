import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_TERM = "".join(("s", "k", "u"))
FORBIDDEN_PRODUCT_TERMS_PATTERN = (
    rf"(?i)(?<![a-z0-9]){LEGACY_TERM}s?(?![a-z0-9])|{LEGACY_TERM}[_-]"
)
FORBIDDEN_PRODUCT_TERMS = re.compile(FORBIDDEN_PRODUCT_TERMS_PATTERN)
SKIP_DIRS = {
    ".cursor",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".temporary",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
SKIP_PATHS = {Path("env/runs")}
SKIP_FILES = {
    Path(
        "docs/experiments/assets/v10-20260720-seven-model-90d-audit/"
        "data/trace-evidence.md"
    ),
}
SKIP_SUFFIXES = {".db", ".gz", ".png", ".jpg", ".jpeg", ".svg", ".pyc"}


def _is_skipped_relative(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if any(rel == skip or skip in rel.parents for skip in SKIP_PATHS):
        return True
    if rel in SKIP_FILES:
        return True
    return False


def _paths(root: Path):
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if _is_skipped_relative(rel):
            continue
        if path.is_file() and path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def _content_hits(root: Path):
    """Scan file contents in-process.

    A full-tree ``rg`` subprocess (especially Cursor's bundled ``rg.exe``)
    is a frequent Kaspersky System Watcher false positive: rapid
    enumeration of every file looks like ransomware/crawler behavior.
    """
    hits = []
    for path in _paths(root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_PRODUCT_TERMS.search(line):
                hits.append(f"{rel}:{line_no}:{line}")
    return hits


def test_path_scan_skips_repo_and_runtime_internals(tmp_path):
    legacy_filename = f"legacy-{LEGACY_TERM}"
    included = tmp_path / "env" / "tools" / "registry.py"
    included.parent.mkdir(parents=True)
    included.write_text("", encoding="utf-8")
    for ignored in (
        tmp_path / ".git" / "refs" / "heads" / legacy_filename,
        tmp_path / "env" / "runs" / "run-1" / f"{legacy_filename}.json",
        tmp_path / ".pytest_cache" / legacy_filename,
        tmp_path / ".temporary" / f"{legacy_filename}.txt",
        tmp_path / ".venv" / "Lib" / f"{legacy_filename}.py",
    ):
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("", encoding="utf-8")

    scanned = {path.relative_to(tmp_path) for path in _paths(tmp_path)}

    assert included.relative_to(tmp_path) in scanned
    assert all(legacy_filename not in str(path) for path in scanned)


def test_content_scan_skips_repo_and_runtime_internals(tmp_path):
    legacy_line = f"legacy {LEGACY_TERM} token"
    included = tmp_path / "env" / "tools" / "registry.py"
    included.parent.mkdir(parents=True)
    included.write_text("safe catalog name\n", encoding="utf-8")
    for ignored in (
        tmp_path / ".git" / "config",
        tmp_path / "env" / "runs" / "run-1" / "trace.json",
        tmp_path / ".temporary" / "git_diff.txt",
        tmp_path / "docs" / "experiments" / "assets"
        / "v10-20260720-seven-model-90d-audit" / "data" / "trace-evidence.md",
        tmp_path / "cache.db",
        tmp_path / "notes.gz",
    ):
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text(legacy_line + "\n", encoding="utf-8")

    hits = _content_hits(tmp_path)

    assert hits == []
    assert included.relative_to(tmp_path) in {
        path.relative_to(tmp_path) for path in _paths(tmp_path) if path.is_file()
    }


def test_repository_uses_product_terms():
    path_offenders = [
        str(path.relative_to(ROOT))
        for path in _paths(ROOT)
        if FORBIDDEN_PRODUCT_TERMS.search(str(path.relative_to(ROOT)))
    ][:25]
    content_offenders = _content_hits(ROOT)[:25]
    offenders = path_offenders + content_offenders

    assert not offenders, (
        "Use product terminology instead of the legacy catalog term:\n"
        + "\n".join(offenders)
    )
