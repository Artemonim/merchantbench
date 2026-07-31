import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_TERM = "".join(("s", "k", "u"))
FORBIDDEN_PRODUCT_TERMS_PATTERN = (
    rf"(?i)(?<![a-z0-9]){LEGACY_TERM}s?(?![a-z0-9])|{LEGACY_TERM}[_-]"
)
FORBIDDEN_PRODUCT_TERMS = re.compile(FORBIDDEN_PRODUCT_TERMS_PATTERN)
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
SKIP_PATHS = {Path("env/runs")}
SKIP_FILES = {
    Path(
        "docs/experiments/assets/v10-20260720-seven-model-90d-audit/"
        "data/trace-evidence.md"
    ),
}
SKIP_SUFFIXES = {".db", ".gz", ".png", ".jpg", ".jpeg", ".svg", ".pyc"}


def _rg_command():
    command = ["rg", "--pcre2", "-n", "-i"]
    for directory in sorted(SKIP_DIRS):
        command.append(f"--glob=!{directory}/**")
    for path in sorted(SKIP_PATHS):
        command.append(f"--glob=!{path}/**")
    for path in sorted(SKIP_FILES):
        command.append(f"--glob=!{path}")
    for suffix in sorted(SKIP_SUFFIXES):
        command.append(f"--glob=!*{suffix}")
    command.extend([FORBIDDEN_PRODUCT_TERMS_PATTERN, "."])
    return command


def _paths(root: Path):
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if any(rel == skip or skip in rel.parents for skip in SKIP_PATHS):
            continue
        if rel in SKIP_FILES:
            continue
        if path.is_file() and path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def test_path_scan_skips_repo_and_runtime_internals(tmp_path):
    legacy_filename = f"legacy-{LEGACY_TERM}"
    included = tmp_path / "env" / "tools" / "registry.py"
    included.parent.mkdir(parents=True)
    included.write_text("", encoding="utf-8")
    for ignored in (
        tmp_path / ".git" / "refs" / "heads" / legacy_filename,
        tmp_path / "env" / "runs" / "run-1" / f"{legacy_filename}.json",
        tmp_path / ".pytest_cache" / legacy_filename,
    ):
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("", encoding="utf-8")

    scanned = {path.relative_to(tmp_path) for path in _paths(tmp_path)}

    assert included.relative_to(tmp_path) in scanned
    assert all(legacy_filename not in str(path) for path in scanned)


def test_content_scan_skips_repo_and_runtime_internals():
    command = _rg_command()

    assert "--glob=!.git/**" in command
    assert "--glob=!env/runs/**" in command
    assert any("trace-evidence.md" in item for item in command)
    assert "--glob=!*.db" in command
    assert "--glob=!*.gz" in command


def test_repository_uses_product_terms():
    content = subprocess.run(
        _rg_command(),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert content.returncode in (0, 1), content.stderr

    path_offenders = [
        str(path.relative_to(ROOT))
        for path in _paths(ROOT)
        if FORBIDDEN_PRODUCT_TERMS.search(str(path.relative_to(ROOT)))
    ][:25]
    content_offenders = content.stdout.splitlines()[:25]
    offenders = path_offenders + content_offenders

    assert not offenders, "Use product terminology instead of the legacy catalog term:\n" + "\n".join(offenders)
