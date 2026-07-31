"""Persistent experiment-group configuration for dashboard analysis.

Groups span independent run databases, so their metadata lives in one small
registry-level JSON file under ``runs_root`` rather than inside any run.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any


CONFIG_VERSION = 1
CONFIG_FILENAME = "experiment_groups.json"
MAX_GROUPS = 100
MAX_BATCHES_PER_GROUP = 50
MAX_BINDINGS_PER_BATCH = 200

FRAMEWORK_PRESETS = [
    {
        "key": "react_160k_compact_30k",
        "label": "React",
    },
    {
        "key": "hermes",
        "label": "Hermes",
    },
]

MODEL_PRESETS = [
    {"label": "GPT", "model": "gpt-5.6-sol"},
    {"label": "Claude 4.8", "model": "claude-opus-4-8"},
    {"label": "GLM 5.2", "model": "bailian/glm-5.2"},
    {"label": "Qwen3.7 Max", "model": "qwen3.7-max"},
    {"label": "Qwen3.7 Plus", "model": "qwen3.7-plus"},
    {"label": "DeepSeek V4 Pro", "model": "bailian/deepseek-v4-pro"},
    {"label": "DeepSeek V4 Flash", "model": "bailian/deepseek-v4-flash"},
    {"label": "Kimi K2.6", "model": "bailian/kimi-k2.6"},
]

_FRAMEWORK_KEYS = {row["key"] for row in FRAMEWORK_PRESETS}
_MODEL_KEYS = {row["model"] for row in MODEL_PRESETS}


class ExperimentGroupStoreError(RuntimeError):
    """The persisted group document could not be safely read or written."""


def _clean_text(value: Any, *, field: str, maximum: int = 120) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    return text


def _clean_optional_run_id(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) > 160:
        raise ValueError("run_id must be at most 160 characters")
    return text


def _sanitize_group(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("each group must be an object")
    group_id = _clean_text(raw.get("id"), field="group.id", maximum=80)
    name = _clean_text(raw.get("name"), field="group.name")

    template = raw.get("template", {})
    if not isinstance(template, dict):
        raise ValueError(f"group {group_id}: template must be an object")
    raw_frameworks = template.get("frameworks", [])
    raw_models = template.get("models", [])
    if not isinstance(raw_frameworks, list):
        raise ValueError(f"group {group_id}: frameworks must be an array")
    if not isinstance(raw_models, list):
        raise ValueError(f"group {group_id}: models must be an array")
    frameworks = [str(value) for value in raw_frameworks]
    models = [str(value) for value in raw_models]
    unknown_frameworks = sorted(set(frameworks) - _FRAMEWORK_KEYS)
    unknown_models = sorted(set(models) - _MODEL_KEYS)
    if unknown_frameworks:
        raise ValueError(
            f"group {group_id}: unknown frameworks: {unknown_frameworks}"
        )
    if unknown_models:
        raise ValueError(f"group {group_id}: unknown models: {unknown_models}")
    if len(frameworks) != len(set(frameworks)):
        raise ValueError(f"group {group_id}: duplicate frameworks")
    if len(models) != len(set(models)):
        raise ValueError(f"group {group_id}: duplicate models")

    raw_batches = raw.get("batches", [])
    if not isinstance(raw_batches, list):
        raise ValueError(f"group {group_id}: batches must be an array")
    if len(raw_batches) > MAX_BATCHES_PER_GROUP:
        raise ValueError(
            f"group {group_id}: at most {MAX_BATCHES_PER_GROUP} batches"
        )
    batches = []
    batch_ids = set()
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, dict):
            raise ValueError(f"group {group_id}: each batch must be an object")
        batch_id = _clean_text(
            raw_batch.get("id"),
            field=f"group {group_id} batch.id",
            maximum=80,
        )
        if batch_id in batch_ids:
            raise ValueError(f"group {group_id}: duplicate batch id {batch_id}")
        batch_ids.add(batch_id)
        batch_name = _clean_text(
            raw_batch.get("name"),
            field=f"group {group_id} batch.name",
        )
        raw_bindings = raw_batch.get("bindings", {})
        if not isinstance(raw_bindings, dict):
            raise ValueError(
                f"group {group_id} batch {batch_id}: bindings must be an object"
            )
        if len(raw_bindings) > MAX_BINDINGS_PER_BATCH:
            raise ValueError(
                f"group {group_id} batch {batch_id}: too many bindings"
            )
        bindings = {}
        for slot_id, run_id in raw_bindings.items():
            clean_slot = _clean_text(
                slot_id,
                field=f"group {group_id} batch {batch_id} slot",
                maximum=220,
            )
            clean_run = _clean_optional_run_id(run_id)
            if clean_run:
                bindings[clean_slot] = clean_run
        batches.append({
            "id": batch_id,
            "name": batch_name,
            "bindings": bindings,
        })

    return {
        "id": group_id,
        "name": name,
        "template": {
            "frameworks": frameworks,
            "models": models,
            "include_human": bool(template.get("include_human", True)),
            "include_rule_based": bool(
                template.get("include_rule_based", True)
            ),
        },
        "batches": batches,
        "updated_at": str(raw.get("updated_at") or ""),
    }


def sanitize_document(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("request body must be an object")
    if "groups" not in raw:
        raise ValueError("groups is required")
    raw_groups = raw["groups"]
    if not isinstance(raw_groups, list):
        raise ValueError("groups must be an array")
    if len(raw_groups) > MAX_GROUPS:
        raise ValueError(f"at most {MAX_GROUPS} groups")
    groups = [_sanitize_group(group) for group in raw_groups]
    group_ids = [group["id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("duplicate group ids")
    now = datetime.now(timezone.utc).isoformat()
    for group in groups:
        group["updated_at"] = now
    return {
        "version": CONFIG_VERSION,
        "groups": groups,
        "updated_at": now,
    }


class ExperimentGroupStore:
    def __init__(self, runs_root: str):
        self.runs_root = runs_root
        self.path = os.path.join(runs_root, CONFIG_FILENAME)
        self._lock = threading.Lock()

    def load(self) -> dict:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return {
                "version": CONFIG_VERSION,
                "groups": [],
                "updated_at": None,
            }
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentGroupStoreError(
                f"cannot read {CONFIG_FILENAME}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ExperimentGroupStoreError(
                f"invalid {CONFIG_FILENAME}: root must be an object"
            )
        if payload.get("version") != CONFIG_VERSION:
            raise ExperimentGroupStoreError(
                f"unsupported {CONFIG_FILENAME} version: "
                f"{payload.get('version')!r}"
            )
        try:
            cleaned = sanitize_document(payload)
        except ValueError as exc:
            raise ExperimentGroupStoreError(
                f"invalid {CONFIG_FILENAME}: {exc}"
            ) from exc
        cleaned["updated_at"] = payload.get("updated_at")
        for index, group in enumerate(cleaned["groups"]):
            group["updated_at"] = (
                (payload.get("groups") or [])[index].get("updated_at")
                or payload.get("updated_at")
                or group["updated_at"]
            )
        return cleaned

    def save(self, raw: Any) -> dict:
        payload = sanitize_document(raw)
        os.makedirs(self.runs_root, exist_ok=True)
        temporary_path = (
            f"{self.path}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        with self._lock:
            try:
                # Refuse to overwrite a document that cannot be read safely.
                # Recovery should be explicit rather than turning corruption
                # into an apparently empty dashboard and losing all bindings.
                if os.path.exists(self.path):
                    self._load_unlocked()
                with open(temporary_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary_path, self.path)
            except ExperimentGroupStoreError:
                raise
            except OSError as exc:
                raise ExperimentGroupStoreError(
                    f"cannot write {CONFIG_FILENAME}: {exc}"
                ) from exc
            finally:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        return payload
