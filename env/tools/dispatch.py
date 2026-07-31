"""Generic tool dispatcher for the /act endpoint.

Routes an OpenAI-format tool_call to the appropriate handler in tools.tools
via the registry, handling argument mapping and idempotency.
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any, Optional

from tools import registry


def _invalid_arguments(path: str, message: str) -> dict:
    field = None
    if path.startswith("$."):
        field = path[2:].split("[", 1)[0].split(".", 1)[0]
    error = {
        "code": "invalid_arguments",
        "path": path,
        "message": message,
    }
    if field:
        error[field] = message
    return {
        "ok": False,
        "error": error,
    }


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _validate_schema(value: Any, schema: dict, path: str) -> Optional[dict]:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return _invalid_arguments(path, f"expected object, got {_type_name(value)}")
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                return _invalid_arguments(f"{path}.{key}", "required argument is missing")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(props))
            if unknown:
                return _invalid_arguments(
                    f"{path}.{unknown[0]}",
                    f"unknown argument: {unknown[0]}",
                )
        for key, item in value.items():
            if key in props:
                err = _validate_schema(item, props[key], f"{path}.{key}")
                if err is not None:
                    return err
        return None
    if expected == "array":
        if not isinstance(value, list):
            return _invalid_arguments(path, f"expected array, got {_type_name(value)}")
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return _invalid_arguments(path, f"expected at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return _invalid_arguments(path, f"expected at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                err = _validate_schema(item, item_schema, f"{path}[{idx}]")
                if err is not None:
                    return err
        return None
    if expected == "string":
        if not isinstance(value, str):
            return _invalid_arguments(path, f"expected string, got {_type_name(value)}")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return _invalid_arguments(path, f"expected integer, got {_type_name(value)}")
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return _invalid_arguments(path, f"expected number, got {_type_name(value)}")
        if not math.isfinite(float(value)):
            return _invalid_arguments(path, "expected a finite number")
    elif expected == "boolean":
        if not isinstance(value, bool):
            return _invalid_arguments(path, f"expected boolean, got {_type_name(value)}")
    if "enum" in schema and value not in schema["enum"]:
        return _invalid_arguments(path, f"expected one of {schema['enum']}")
    if "minimum" in schema and value < schema["minimum"]:
        return _invalid_arguments(path, f"expected >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        return _invalid_arguments(path, f"expected <= {schema['maximum']}")
    return None


def validate_arguments(spec: registry.ToolSpec, args_dict: Any) -> Optional[dict]:
    return _validate_schema(args_dict, spec.parameters, "$")


def dispatch_tool(env, agent_id: str, tool_name: str, args_dict: dict,
                  *, idempotency_key: Optional[str] = None,
                  idempotency_fingerprint: Optional[dict] = None,
                  denylist: Optional[set[str]] = None) -> dict:
    """Execute a single tool call and return the result dict.

    Args:
        env: The Environment instance.
        agent_id: The calling agent's ID.
        tool_name: Tool name matching a ToolSpec in the registry.
        args_dict: Parsed arguments from the tool_call's function.arguments.
        idempotency_key: If provided, used for mutating-tool deduplication.
        denylist: Tool names that are denied by the scenario. If provided,
                  calls to denied tools return an error.

    Returns:
        JSON-serializable result dict from the handler.
    """
    if denylist and tool_name in denylist:
        return {"ok": False, "error": f"tool '{tool_name}' is not available in this scenario"}
    spec = registry.get(tool_name)
    if spec is None:
        return {"ok": False, "error": f"unknown tool: {tool_name}"}
    if spec.handler is None:
        return {"ok": False, "error": f"tool {tool_name} has no handler"}
    invalid = validate_arguments(spec, args_dict)
    if invalid is not None:
        return invalid

    handler = spec.handler
    sig = inspect.signature(handler)
    params = list(sig.parameters.keys())

    def _invoke() -> dict:
        kwargs: dict[str, Any] = {}
        accepted = {"env", "agent_id"}
        for p_name in params:
            if p_name in ("env", "agent_id"):
                continue
            accepted.add(p_name)
            if p_name in args_dict:
                kwargs[p_name] = args_dict[p_name]

        unknown = sorted(set(args_dict) - accepted)
        if unknown:
            joined = ", ".join(unknown)
            return {"ok": False, "error": f"unknown arguments for {tool_name}: {joined}"}

        if "agent_id" in params:
            result = handler(env, agent_id, **kwargs)
        else:
            result = handler(env, **kwargs)

        if result is None:
            return {"ok": False, "error": "not found"}
        return result

    if spec.mutating and idempotency_key:
        return env.with_idempotency(
            idempotency_key,
            _invoke,
            fingerprint=idempotency_fingerprint,
        )
    return _invoke()
