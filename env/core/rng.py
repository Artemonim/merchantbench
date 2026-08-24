"""Seeded RNG derivation. Each call returns an independent numpy Generator.

Reproducibility contract:
- sub_seed = blake2b(master_seed || channel || keys...)
- Same (master_seed, channel, *keys) => identical generator state => identical samples.
- Different keys give independent streams.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

VALID_CHANNELS = {
    "arrival",
    "anomaly_type",
    "anomaly_time",
    "price_change",
    "delist",
    "timeout",
    "ship_delay",
    "data_gen",
    "supplier_event_wait",
    "supplier_event_payload",
    "settlement_delay",
    "public_review",
    "risk_rates",
    "supplier_risk",
}


def derive_rng(master_seed: int, channel: str, *keys: Any) -> np.random.Generator:
    if channel not in VALID_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; allowed: {VALID_CHANNELS}")
    h = hashlib.blake2b(digest_size=8)
    h.update(str(master_seed).encode())
    h.update(b"|")
    h.update(channel.encode())
    for k in keys:
        h.update(b"|")
        h.update(str(k).encode())
    sub_seed = int.from_bytes(h.digest(), "little")
    return np.random.default_rng(sub_seed)


def bucket_pick(u: float, probs: list[float], labels: list[str]) -> str:
    """Given u in [0,1) and a non-normalized prob list, return the label whose
    cumulative bucket contains u. Remaining mass goes to the first label
    (assumed 'normal' / default)."""
    assert len(probs) == len(labels)
    cum = 0.0
    for p, lab in zip(probs, labels):
        cum += p
        if u < cum:
            return lab
    return labels[0]
