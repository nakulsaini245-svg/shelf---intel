"""Deterministic, order-independent seeding.

Every random stream is keyed by (global_seed, scope, day, stream). The key is
hashed with BLAKE2b (stable across processes and Python versions, unlike the
built-in hash()), then fed to numpy's SeedSequence/PCG64.

Consequences:
  * Day N's draws never depend on how many days are generated, so extending the
    history by one day leaves days 0..N-1 untouched (append-only determinism).
  * Adding a new stream name never perturbs existing streams.

`scope` names the entity domain whose ordering a vectorised draw follows
(e.g. "cell", "series", "group"); per-entity generators are avoided because
creating ~10^5 generators per run is needlessly slow.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

# Static world construction uses this pseudo-day so it can never collide with
# simulated days (warm-up days are negative but small: -warmup..-1).
WORLD_DAY: int = -10_000


def seed_words(global_seed: int, scope: str, day: int, stream: str) -> list[int]:
    """Four 32-bit words derived from the key; pure function of its inputs."""
    key = json.dumps([int(global_seed), scope, int(day), stream]).encode()
    digest = hashlib.blake2b(key, digest_size=16).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def rng_for(global_seed: int, scope: str, day: int, stream: str) -> np.random.Generator:
    """Fresh Generator for one (scope, day, stream) key."""
    ss = np.random.SeedSequence(seed_words(global_seed, scope, day, stream))
    return np.random.Generator(np.random.PCG64(ss))
