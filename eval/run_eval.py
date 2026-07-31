"""Hosted-evaluation harness — submit a Docker image, get final net assets.

Submission contract (mirrors MLE-bench / SWE-bench / AppWorld):
  1. Submitter builds a Docker image whose entrypoint speaks the
     MerchantBench SDK protocol and reads four env vars at startup:
       MERCHANTBENCH_BASE_URL  MERCHANTBENCH_RUN_ID  MERCHANTBENCH_AGENT_ID
       MERCHANTBENCH_AGENT_TOKEN
  2. They publish the image (registry, tarball, whatever).
  3. We run this harness with --agent-image <image>. The harness:
       - boots the env container (image: $MERCHANTBENCH_ENV_IMAGE, default
         merchantbench-env:dev — must be built once via env_image/Dockerfile)
       - creates a fresh run on the eval scenario
       - boots the agent container, networked to the env container
       - waits for the env to reach status=finished
       - pulls the merchant section, computes result metrics, writes result.json
       - tears everything down (containers + network)

Result metric: see eval/scoring.py. Headline metric is the final
`net_assets` reported by the env.

Local prerequisites (one-time):
  .venv/bin/python -m pip install -r eval/requirements.txt
  cp .env.example .env && edit            # OPENAI_API_KEY etc.
  docker build -t merchantbench-env:dev   -f eval/env_image/Dockerfile env/
  docker build -t merchantbench-react:dev -f agent/submission_template/Dockerfile agent/

Run:
  .venv/bin/python -m eval.run_eval --agent-image merchantbench-react:dev --output result.json
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import uuid
from typing import Any, Optional

import requests

try:
    import docker
    from docker.errors import APIError, ImageNotFound, NotFound
except ImportError:  # pragma: no cover
    print("eval/run_eval.py requires the docker SDK: "
          ".venv/bin/python -m pip install -r eval/requirements.txt",
          file=sys.stderr)
    raise

try:
    import yaml
except ImportError:  # pragma: no cover
    print("eval/run_eval.py requires PyYAML: "
          ".venv/bin/python -m pip install -r eval/requirements.txt",
          file=sys.stderr)
    raise

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval import scoring


DEFAULT_ENV_IMAGE = os.environ.get("MERCHANTBENCH_ENV_IMAGE", "merchantbench-env:dev")
DEFAULT_AGENT_IMAGE = os.environ.get("MERCHANTBENCH_AGENT_IMAGE", "merchantbench-react:dev")
# Single source of truth: the official eval scenario IS scenarios/default.yaml.
# `--scenario default` (default) reads scenarios/default.yaml; any other name
# resolves to scenarios/<name>.yaml (so people can sweep alternate configs
# without forking the harness).
DEFAULT_SCENARIO = os.environ.get("MERCHANTBENCH_SCENARIO", "default")
DEFAULT_MASTER_SEED = int(os.environ.get("MERCHANTBENCH_MASTER_SEED", "42"))
DEFAULT_RUN_TIMEOUT = int(os.environ.get("MERCHANTBENCH_RUN_TIMEOUT", "10800"))  # 3h
ENV_INTERNAL_PORT = 5000


def _log(msg: str) -> None:
    print(f"[eval] {msg}", file=sys.stderr, flush=True)


def _load_scenario(name: str) -> dict:
    """Read env/scenarios/<name>.yaml from the repo root. The official
    leaderboard config is `default` (= env/scenarios/default.yaml)."""
    path = os.path.join(_REPO_ROOT, "env", "scenarios", f"{name}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"scenario file missing: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_env_file(path: str) -> dict[str, str]:
    """Parse a dotenv-style file. Empty values are dropped so the
    container inherits its image defaults instead of seeing an empty
    string."""
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                out[k] = v
    return out


def _build_agent_env(*, env_name: str, run_id: str, agent_id: str,
                     agent_token: str, creds: dict[str, str]) -> dict[str, str]:
    agent_env = {
        k: v
        for k, v in creds.items()
        if not k.startswith("MERCHANTBENCH_")
    }
    agent_env.update({
        "MERCHANTBENCH_BASE_URL": f"http://{env_name}:{ENV_INTERNAL_PORT}",
        "MERCHANTBENCH_RUN_ID": run_id,
        "MERCHANTBENCH_AGENT_ID": agent_id,
        "MERCHANTBENCH_AGENT_TOKEN": agent_token,
    })
    return agent_env


def _wait_http(url: str, timeout: float = 60.0,
               headers: Optional[dict[str, str]] = None) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2, headers=headers)
            if r.status_code < 500:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {url}: {last_err}")


def _wait_finished(env_url: str, run_id: str, timeout: float,
                   headers: Optional[dict[str, str]] = None) -> str:
    """Poll /runs/<rid> until status is finished or stopped. Returns
    the terminal status."""
    deadline = time.time() + timeout
    last_status: Optional[str] = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{env_url}/runs/{run_id}", timeout=10,
                             headers=headers)
            if r.status_code == 200:
                row = r.json() or {}
                status = row.get("status")
                if status != last_status:
                    _log(f"run {run_id}: {status}")
                    last_status = status
                if status in ("finished", "stopped", "error"):
                    return status
        except requests.RequestException as e:
            _log(f"poll error: {e}")
        time.sleep(5)
    raise SystemExit(f"run {run_id} did not finish within {timeout}s")


def _create_run(env_url: str, scenario: dict, master_seed: int,
                interval_ms: int,
                headers: Optional[dict[str, str]] = None) -> dict:
    body = {
        "scenario": scenario,
        "master_seed": master_seed,
        "auto_start": True,
        "interval_ms": interval_ms,
        "bootstrap_agent": "none",
        "name": f"eval-{int(time.time())}",
    }
    r = requests.post(f"{env_url}/runs", json=body, timeout=30,
                      headers=headers)
    r.raise_for_status()
    return r.json()


def _fetch_merchant(env_url: str, run_id: str, agent_id: str,
                    headers: Optional[dict[str, str]] = None) -> dict:
    r = requests.get(
        f"{env_url}/runs/{run_id}/agents/{agent_id}/sections/merchant",
        timeout=30,
        headers=headers)
    r.raise_for_status()
    return r.json()


def evaluate(*, agent_image: str, env_image: str = DEFAULT_ENV_IMAGE,
             scenario_name: str = DEFAULT_SCENARIO,
             master_seed: Optional[int] = None,
             agent_id: str = "agent_0",
             env_file: str = ".env",
             interval_ms: int = 100,
             run_timeout: float = DEFAULT_RUN_TIMEOUT,
             host_port: int = 0,
             keep_containers: bool = False) -> dict[str, Any]:
    """Run one full hosted evaluation. Returns the result dict that
    will be written to result.json."""

    scenario = _load_scenario(scenario_name)
    # Always pin the seed for reproducibility; default is 42, overridable for sweeps.
    pinned_seed = int(master_seed) if master_seed is not None else DEFAULT_MASTER_SEED
    scenario.setdefault("run", {})["master_seed"] = pinned_seed
    seed = pinned_seed

    client = docker.from_env()
    # Sanity-check images exist locally — pulls would be unexpected and
    # we'd rather fail fast than silently fetch upstream.
    for img in (env_image, agent_image):
        try:
            client.images.get(img)
        except ImageNotFound:
            raise SystemExit(
                f"image not found locally: {img}. Build it first "
                f"(see eval/README.md).")

    tag = uuid.uuid4().hex[:8]
    network_name = f"merchantbench-net-{tag}"
    env_name = f"merchantbench-env-{tag}"
    agent_name = f"merchantbench-agent-{tag}"

    network = client.networks.create(network_name, driver="bridge")
    env_container = None
    agent_container = None
    admin_token = secrets.token_urlsafe(32)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    try:
        # Bind a host port so the harness on the host can talk to the env.
        port_binding = {f"{ENV_INTERNAL_PORT}/tcp": host_port or None}
        _log(f"starting env container ({env_image})")
        env_container = client.containers.run(
            env_image,
            name=env_name,
            network=network_name,
            ports=port_binding,
            environment={
                "MERCHANTBENCH_REQUIRE_TOKENS": "1",
                "MERCHANTBENCH_ADMIN_TOKEN": admin_token,
            },
            detach=True,
            remove=False,
        )
        env_container.reload()
        # Resolve the host port docker assigned (if host_port==0).
        bindings = (env_container.attrs.get("NetworkSettings") or {}).get("Ports") or {}
        host_entries = bindings.get(f"{ENV_INTERNAL_PORT}/tcp") or []
        if not host_entries:
            raise SystemExit("env container did not expose port "
                              f"{ENV_INTERNAL_PORT}")
        host_addr = host_entries[0].get("HostIp") or "127.0.0.1"
        if host_addr in ("0.0.0.0", "::"):
            host_addr = "127.0.0.1"
        host_port_actual = int(host_entries[0]["HostPort"])
        env_url = f"http://{host_addr}:{host_port_actual}"

        _wait_http(f"{env_url}/runs", timeout=60, headers=admin_headers)
        _log(f"env up at {env_url}")

        run_payload = _create_run(env_url, scenario, seed, interval_ms,
                                  headers=admin_headers)
        run_id = run_payload["run_id"]
        agent_token = run_payload["agent_token"]
        _log(f"created run_id={run_id} seed={seed} scenario={scenario_name}")

        # Build agent env. Internal hostname = container name on the
        # bridge network (docker DNS).
        # OpenAI / model creds from the harness host's .env (NOT baked
        # into the image — submitters' images can rely on these env vars
        # being present at runtime).
        creds = _load_env_file(os.path.join(_REPO_ROOT, env_file))
        if not creds.get("OPENAI_API_KEY"):
            _log(f"warning: {env_file} has no OPENAI_API_KEY — agent will fail")
        agent_env = _build_agent_env(
            env_name=env_name,
            run_id=run_id,
            agent_id=agent_id,
            agent_token=agent_token,
            creds=creds,
        )

        _log(f"starting agent container ({agent_image})")
        agent_container = client.containers.run(
            agent_image,
            name=agent_name,
            network=network_name,
            environment=agent_env,
            detach=True,
            remove=False,
        )

        status = _wait_finished(env_url, run_id, timeout=run_timeout,
                                headers=admin_headers)
        _log(f"run terminated with status={status}")

        merchant = _fetch_merchant(env_url, run_id, agent_id,
                                   headers=admin_headers)
        result_metrics = scoring.compute(merchant)

        # Capture the agent container's tail logs for the result file —
        # useful when a submission silently misbehaves.
        try:
            agent_logs_tail = agent_container.logs(tail=200).decode(
                "utf-8", errors="replace")
        except APIError:
            agent_logs_tail = ""

        return {
            "agent_image": agent_image,
            "env_image": env_image,
            "scenario": scenario_name,
            "master_seed": seed,
            "run_id": run_id,
            "terminal_status": status,
            "agent_id": agent_id,
            **result_metrics,
            "agent_logs_tail": agent_logs_tail.splitlines()[-50:],
        }
    finally:
        if not keep_containers:
            for c, label in ((agent_container, "agent"), (env_container, "env")):
                if c is None:
                    continue
                try:
                    c.stop(timeout=5)
                except (APIError, NotFound):
                    pass
                try:
                    c.remove(force=True)
                except (APIError, NotFound):
                    pass
            try:
                network.remove()
            except (APIError, NotFound):
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MerchantBench hosted-evaluation harness.")
    ap.add_argument("--agent-image", default=DEFAULT_AGENT_IMAGE,
                    help="Submission image (must be built locally).")
    ap.add_argument("--env-image", default=DEFAULT_ENV_IMAGE)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help="Name (without .yaml) under eval/scenarios/")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="Override scenario master_seed (for sweeps).")
    ap.add_argument("--agent-id", default="agent_0")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT,
                    help="Hard cap (seconds) on a single run.")
    ap.add_argument("--host-port", type=int, default=0,
                    help="Host port to expose env on (0 = random).")
    ap.add_argument("--output", default="result.json")
    ap.add_argument("--keep-containers", action="store_true",
                    help="Skip teardown — useful for debugging.")
    args = ap.parse_args()

    result = evaluate(
        agent_image=args.agent_image,
        env_image=args.env_image,
        scenario_name=args.scenario,
        master_seed=args.master_seed,
        agent_id=args.agent_id,
        env_file=args.env_file,
        interval_ms=args.interval_ms,
        run_timeout=args.run_timeout,
        host_port=args.host_port,
        keep_containers=args.keep_containers,
    )

    out_path = os.path.abspath(args.output)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"net_assets={result.get('final_net_assets')} → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
