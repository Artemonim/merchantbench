#!/usr/bin/env python3
"""Python stage runner for MerchantBench local CI.

Called by ``build.ps1`` with ``--stage``, ``--root``, and ``--log-path``.
Human-readable output goes to the log file. Exactly one JSON object is printed
as the last stdout line.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

# * Subprocess ceilings (seconds). Coverage re-runs the suite with instrumentation.
TEST_TIMEOUT_SEC = 900
COVERAGE_TIMEOUT_SEC = 1200
DEFAULT_TIMEOUT_SEC = 300

# * compileall skips runtime/private trees so CI never writes bytecode there.
COMPILEALL_EXCLUDE_RE = r"[\\/](runs|private_data|\.venv|venv|\.ci_cache|\.enforcer)[\\/]"

COMPILE_TARGETS = ("env", "agent", "eval", "scripts", "tests", "build.py")
COVERAGE_PACKAGES = ("env", "agent", "eval", "scripts")
COVERAGE_WARN_PERCENT = 75.0
MAX_TEST_ISSUES = 50

RUFF_REFORMATTED_RE = re.compile(r"(\d+)\s+files?\s+reformatted", re.IGNORECASE)


class StageLogger:
    """Append-only UTF-8 log writer for a single CI stage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, message: str) -> None:
        """Write one logical line (or a pre-split block) to the stage log."""
        text = message.rstrip("\n")
        self._handle.write(text + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()


def stage_result(
    name: str,
    status: str,
    note: str = "",
    duration_ms: int = 0,
    details: Mapping[str, Any] | None = None,
    issues: Sequence[Mapping[str, Any]] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AE2 stage-result mapping."""
    return {
        "name": name,
        "status": status,
        "note": note,
        "duration_ms": int(duration_ms),
        "details": dict(details) if details else {},
        "issues": [dict(item) for item in issues] if issues else [],
        "metrics": dict(metrics) if metrics else {},
    }


def emit_result(result: Mapping[str, Any]) -> None:
    """Print the single JSON object expected by the orchestrator."""
    sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def make_issue(
    language: str,
    tool: str,
    rule: str,
    message: str,
    count: int = 1,
) -> dict[str, Any]:
    """Build one AE2 issue object."""
    return {
        "language": language,
        "tool": tool,
        "rule": rule,
        "count": int(count) if count >= 1 else 1,
        "message": message,
    }


def run_command(
    argv: Sequence[str],
    cwd: Path,
    logger: StageLogger,
    timeout_sec: int,
) -> tuple[int, str, str]:
    """Run an external command, capture output, and append it to the stage log.

    Returns:
        Tuple of ``(returncode, stdout, stderr)``. Timeout uses return code 124.
    """
    logger.write("command: {0}".format(" ".join(argv)))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        logger.write("timeout after {0}s (limit {1}s)".format(duration_ms / 1000.0, timeout_sec))
        if stdout:
            logger.write("--- stdout ---")
            logger.write(stdout)
        if stderr:
            logger.write("--- stderr ---")
            logger.write(stderr)
        return 124, stdout, stderr

    duration_ms = int((time.perf_counter() - started) * 1000)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    logger.write("exit_code: {0}".format(completed.returncode))
    logger.write("duration_ms: {0}".format(duration_ms))
    if stdout:
        logger.write("--- stdout ---")
        logger.write(stdout)
    if stderr:
        logger.write("--- stderr ---")
        logger.write(stderr)
    return completed.returncode, stdout, stderr


def relpath(root: Path, path: str | Path) -> str:
    """Return ``path`` relative to ``root`` when possible."""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path).replace("\\", "/")


def parse_reformatted_count(text: str) -> int | None:
    """Parse ruff format's 'N file(s) reformatted' summary when present."""
    match = RUFF_REFORMATTED_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def run_fmt(root: Path, logger: StageLogger) -> dict[str, Any]:
    """Format the tree with ruff (fix mode)."""
    started = time.perf_counter()
    code, stdout, stderr = run_command(
        [sys.executable, "-m", "ruff", "format", str(root)],
        cwd=root,
        logger=logger,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    combined = "{0}\n{1}".format(stdout, stderr)
    reformatted = parse_reformatted_count(combined)
    if code != 0:
        return stage_result(
            "fmt",
            "fail",
            note="ruff format exited {0}".format(code),
            duration_ms=duration_ms,
            details={"exit_code": code, "reformatted": reformatted},
        )
    note = "No files reformatted"
    if reformatted is not None:
        note = "Reformatted {0} file(s)".format(reformatted)
    return stage_result(
        "fmt",
        "ok",
        note=note,
        duration_ms=duration_ms,
        details={"exit_code": code, "reformatted": reformatted},
        metrics={"reformatted_files": reformatted if reformatted is not None else 0},
    )


def parse_ruff_json(payload: str, root: Path) -> list[dict[str, Any]]:
    """Convert ruff JSON diagnostics into AE2 issues."""
    text = payload.strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return [make_issue("python", "ruff", "parse_error", "Unable to parse ruff JSON output")]
    if not isinstance(raw, list):
        return [make_issue("python", "ruff", "parse_error", "Unexpected ruff JSON shape")]

    issues: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "unknown")
        message = str(item.get("message") or "")
        filename = str(item.get("filename") or "")
        location = item.get("location") if isinstance(item.get("location"), dict) else {}
        row = location.get("row") if isinstance(location, dict) else None
        rel = relpath(root, filename) if filename else filename
        where = rel if row is None else "{0}:{1}".format(rel, row)
        issues.append(make_issue("python", "ruff", code, "{0}: {1}".format(where, message)))
    return issues


def run_lint(root: Path, logger: StageLogger) -> dict[str, Any]:
    """Apply ruff safe fixes, then fail on remaining diagnostics."""
    started = time.perf_counter()
    fix_code, _, _ = run_command(
        [sys.executable, "-m", "ruff", "check", "--fix", str(root)],
        cwd=root,
        logger=logger,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    )
    check_code, check_stdout, check_stderr = run_command(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", str(root)],
        cwd=root,
        logger=logger,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    issues = parse_ruff_json(check_stdout or check_stderr, root)

    if check_code == 124 or fix_code == 124:
        return stage_result(
            "lint",
            "fail",
            note="ruff check timed out",
            duration_ms=duration_ms,
            issues=issues,
        )

    if issues:
        return stage_result(
            "lint",
            "fail",
            note="{0} remaining ruff finding(s)".format(len(issues)),
            duration_ms=duration_ms,
            details={"fix_exit_code": fix_code, "check_exit_code": check_code, "finding_count": len(issues)},
            issues=issues,
        )

    if check_code != 0:
        return stage_result(
            "lint",
            "fail",
            note="ruff check exited {0} without parsed findings".format(check_code),
            duration_ms=duration_ms,
            details={"fix_exit_code": fix_code, "check_exit_code": check_code},
        )

    return stage_result(
        "lint",
        "ok",
        note="No remaining ruff findings",
        duration_ms=duration_ms,
        details={"fix_exit_code": fix_code, "check_exit_code": check_code},
    )


def run_compile(root: Path, logger: StageLogger) -> dict[str, Any]:
    """Byte-compile project Python trees with compileall."""
    started = time.perf_counter()
    targets = [item for item in COMPILE_TARGETS if (root / item).exists()]
    if not targets:
        return stage_result("compile", "fail", note="No compile targets found", duration_ms=0)

    argv = [sys.executable, "-m", "compileall", "-q", "-x", COMPILEALL_EXCLUDE_RE, *targets]
    code, stdout, stderr = run_command(
        argv,
        cwd=root,
        logger=logger,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    combined = "{0}\n{1}".format(stdout, stderr).strip()
    issues: list[dict[str, Any]] = []
    if combined:
        for line in combined.splitlines():
            stripped = line.strip()
            if stripped:
                issues.append(make_issue("python", "compileall", "compile_error", stripped))

    if code != 0:
        return stage_result(
            "compile",
            "fail",
            note="compileall exited {0}".format(code),
            duration_ms=duration_ms,
            details={"exit_code": code, "targets": targets},
            issues=issues[:MAX_TEST_ISSUES],
        )
    return stage_result(
        "compile",
        "ok",
        note="Compiled {0} target(s)".format(len(targets)),
        duration_ms=duration_ms,
        details={"exit_code": code, "targets": targets},
    )


def parse_junit(path: Path) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """Parse pytest JUnit XML into counts and per-failure issues.

    Returns:
        ``(tests, failures, errors, skipped, issues)``.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    tests = 0
    failures = 0
    errors = 0
    skipped = 0
    issues: list[dict[str, Any]] = []

    for case in root.iter("testcase"):
        tests += 1
        skipped_node = case.find("skipped")
        failure_node = case.find("failure")
        error_node = case.find("error")
        if skipped_node is not None:
            skipped += 1
            continue
        node = failure_node if failure_node is not None else error_node
        if node is None:
            continue
        if failure_node is not None:
            failures += 1
            rule = "test_failure"
        else:
            errors += 1
            rule = "test_error"
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        raw_message = node.get("message") or (node.text or "")
        first_line = raw_message.strip().splitlines()[0] if raw_message.strip() else ""
        identity = "{0}::{1}".format(classname, name)
        message = identity if not first_line else "{0} — {1}".format(identity, first_line)
        if len(issues) < MAX_TEST_ISSUES:
            issues.append(make_issue("python", "pytest", rule, message))

    return tests, failures, errors, skipped, issues


def pytest_base_args() -> list[str]:
    """Shared xdist flags for test and coverage stages."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "auto",
        "--maxprocesses=8",
        "--dist=loadscope",
    ]


def run_test(root: Path, logger: StageLogger, log_path: Path) -> dict[str, Any]:
    """Run the pytest suite and summarize JUnit results."""
    started = time.perf_counter()
    junit_path = log_path.parent / "test.junit.xml"
    argv = pytest_base_args() + ["--junitxml={0}".format(junit_path)]
    code, _, _ = run_command(argv, cwd=root, logger=logger, timeout_sec=TEST_TIMEOUT_SEC)
    duration_ms = int((time.perf_counter() - started) * 1000)

    if code == 124:
        return stage_result(
            "test",
            "fail",
            note="pytest timed out after {0}s".format(TEST_TIMEOUT_SEC),
            duration_ms=duration_ms,
        )

    if not junit_path.is_file():
        return stage_result(
            "test",
            "fail",
            note="JUnit report missing at {0}".format(junit_path.as_posix()),
            duration_ms=duration_ms,
            details={"exit_code": code},
        )

    try:
        tests, failures, errors, skipped, issues = parse_junit(junit_path)
    except (ET.ParseError, OSError) as exc:
        return stage_result(
            "test",
            "fail",
            note="Unable to parse JUnit XML: {0}".format(exc),
            duration_ms=duration_ms,
            details={"exit_code": code},
        )

    details = {
        "exit_code": code,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "junit": junit_path.as_posix(),
    }
    metrics = {
        "test_counts": {
            "total": tests,
            "passed": max(tests - failures - errors - skipped, 0),
            "failed": failures,
            "skipped": skipped,
            "errors": errors,
        }
    }
    note = "{0} passed, {1} failed, {2} errors, {3} skipped".format(
        details["tests"] - failures - errors - skipped,
        failures,
        errors,
        skipped,
    )
    status = "ok" if code == 0 else "fail"
    return stage_result(
        "test",
        status,
        note=note,
        duration_ms=duration_ms,
        details=details,
        issues=issues,
        metrics=metrics,
    )


def parse_coverage_percent(path: Path) -> float | None:
    """Read ``totals.percent_covered`` from a coverage.py JSON report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    totals = payload.get("totals") if isinstance(payload, dict) else None
    if not isinstance(totals, dict):
        return None
    value = totals.get("percent_covered")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_coverage(root: Path, logger: StageLogger, log_path: Path) -> dict[str, Any]:
    """Collect line coverage. Test assertion failures do not fail this stage."""
    started = time.perf_counter()
    coverage_json = log_path.parent / "coverage.json"
    argv = pytest_base_args() + [
        "--cov=env",
        "--cov=agent",
        "--cov=eval",
        "--cov=scripts",
        "--cov-report=json:{0}".format(coverage_json),
        "--cov-report=term",
    ]
    code, _, _ = run_command(argv, cwd=root, logger=logger, timeout_sec=COVERAGE_TIMEOUT_SEC)
    duration_ms = int((time.perf_counter() - started) * 1000)

    if code == 124:
        return stage_result(
            "coverage",
            "fail",
            note="coverage pytest timed out after {0}s".format(COVERAGE_TIMEOUT_SEC),
            duration_ms=duration_ms,
        )

    # * Exit 2+ is a runner/usage failure; exit 1 is typically failing tests with data.
    if code not in (0, 1) or not coverage_json.is_file():
        return stage_result(
            "coverage",
            "fail",
            note="Coverage data missing or pytest failed to run (exit {0})".format(code),
            duration_ms=duration_ms,
            details={"exit_code": code, "coverage_json": coverage_json.as_posix()},
        )

    percent = parse_coverage_percent(coverage_json)
    if percent is None:
        return stage_result(
            "coverage",
            "fail",
            note="Coverage JSON missing totals.percent_covered",
            duration_ms=duration_ms,
            details={"exit_code": code},
        )

    metrics = {
        "coverage": {
            "lines_percent": percent,
            "status": "ok" if percent >= COVERAGE_WARN_PERCENT else "warn",
        }
    }
    details = {"exit_code": code, "lines_percent": percent, "warn_threshold": COVERAGE_WARN_PERCENT}
    if percent >= COVERAGE_WARN_PERCENT:
        return stage_result(
            "coverage",
            "ok",
            note="Line coverage {0:.1f}% (>= {1:.0f}%)".format(percent, COVERAGE_WARN_PERCENT),
            duration_ms=duration_ms,
            details=details,
            metrics=metrics,
        )
    return stage_result(
        "coverage",
        "warn",
        note="Line coverage {0:.1f}% < {1:.0f}% warn threshold".format(percent, COVERAGE_WARN_PERCENT),
        duration_ms=duration_ms,
        details=details,
        issues=[
            make_issue(
                "python",
                "coverage",
                "below_warn",
                "Line coverage {0:.1f}% is below {1:.0f}%".format(percent, COVERAGE_WARN_PERCENT),
            )
        ],
        metrics=metrics,
    )


def dispatch(stage: str, root: Path, logger: StageLogger, log_path: Path) -> dict[str, Any]:
    """Run the requested adapter stage."""
    if stage == "fmt":
        return run_fmt(root, logger)
    if stage == "lint":
        return run_lint(root, logger)
    if stage == "compile":
        return run_compile(root, logger)
    if stage == "test":
        return run_test(root, logger, log_path)
    if stage == "coverage":
        return run_coverage(root, logger, log_path)
    return stage_result(stage, "fail", note="Unknown stage: {0}".format(stage))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the adapter."""
    parser = argparse.ArgumentParser(description="MerchantBench Python CI stage runner")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("fmt", "lint", "compile", "test", "coverage"),
        help="Stage to execute. codebase-memory is owned by build.ps1.",
    )
    parser.add_argument("--root", required=True, help="Repository root")
    parser.add_argument("--log-path", required=True, dest="log_path", help="Stage log file path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Always emits one JSON line on stdout."""
    args = parse_args(argv)
    root = Path(args.root).resolve()
    log_path = Path(args.log_path)
    logger = StageLogger(log_path)
    result: dict[str, Any]
    try:
        if not root.is_dir():
            result = stage_result(args.stage, "fail", note="Root is not a directory: {0}".format(root))
        else:
            logger.write("stage: {0}".format(args.stage))
            logger.write("root: {0}".format(root))
            result = dispatch(args.stage, root, logger, log_path)
    except Exception as exc:  # noqa: BLE001 (adapter must always emit JSON)
        logger.write("unhandled error: {0!r}".format(exc))
        result = stage_result(args.stage, "fail", note="Unhandled adapter error: {0}".format(exc))
    finally:
        logger.close()

    emit_result(result)
    if result.get("status") in ("ok", "warn", "skip", "cached"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
