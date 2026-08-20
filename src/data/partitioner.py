"""Federated partitioning strategies.

Implements the three most-cited partitions used in the FL literature:

* ``iid``         - uniformly random shuffle then equal split.
* ``dirichlet``   - per-class Dirichlet(``alpha``) over clients.  Lower alpha
                    -> more skewed (heterogeneous) partitions.  This is the
                    standard non-IID protocol used by Hsu et al. 2019.
* ``shard``       - sort by label then hand out N sequential shards per client
                    (McMahan et al. 2017 protocol).

Returns a mapping ``client_id -> np.ndarray`` of training indices.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..utils.logger import get_logger


class FederatedPartitioner:
    def __init__(self, num_clients: int, strategy: str = "dirichlet",
                 alpha: float = 0.5, seed: int = 42, logger=None):
        self.num_clients = int(num_clients)
        self.strategy = strategy.lower()
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.logger = logger or get_logger("data.partitioner")

    def split(self, y: np.ndarray) -> Dict[int, np.ndarray]:
        if self.strategy == "iid":
            return self._iid(y)
        if self.strategy == "dirichlet":
            return self._dirichlet(y)
        if self.strategy == "shard":
            return self._shard(y)
        raise ValueError(f"Unknown partition strategy: {self.strategy}")

    def _iid(self, y: np.ndarray) -> Dict[int, np.ndarray]:
        idx = np.arange(len(y))
        self.rng.shuffle(idx)
        chunks = np.array_split(idx, self.num_clients)
        return {i: chunk for i, chunk in enumerate(chunks)}

    def _dirichlet(self, y: np.ndarray) -> Dict[int, np.ndarray]:
        n_classes = int(y.max()) + 1
        client_indices: List[List[int]] = [[] for _ in range(self.num_clients)]
        for c in range(n_classes):
            cls_idx = np.where(y == c)[0]
            self.rng.shuffle(cls_idx)
            proportions = self.rng.dirichlet(np.repeat(self.alpha, self.num_clients))
            split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, split_points)
            for client_id, part in enumerate(splits):
                client_indices[client_id].extend(part.tolist())
        partition = {i: np.array(sorted(idx), dtype=np.int64)
                     for i, idx in enumerate(client_indices)}
        self._log_distribution(partition, y)
        return partition

    def _shard(self, y: np.ndarray, shards_per_client: int = 2) -> Dict[int, np.ndarray]:
        order = np.argsort(y, kind="stable")
        n_shards = self.num_clients * shards_per_client
        shards = np.array_split(order, n_shards)
        shard_ids = list(range(n_shards))
        self.rng.shuffle(shard_ids)
        partition: Dict[int, np.ndarray] = {}
        for client_id in range(self.num_clients):
            assigned = shard_ids[client_id * shards_per_client:(client_id + 1) * shards_per_client]
            idx = np.concatenate([shards[s] for s in assigned])
            partition[client_id] = np.sort(idx)
        self._log_distribution(partition, y)
        return partition

    def _log_distribution(self, partition: Dict[int, np.ndarray], y: np.ndarray) -> None:
        n_classes = int(y.max()) + 1
        rows = []
        for cid, idx in partition.items():
            counts = np.bincount(y[idx], minlength=n_classes)
            rows.append({
                "client_id": cid,
                "n_samples": int(len(idx)),
                **{f"class_{c}": int(counts[c]) for c in range(n_classes)},
            })
        self.logger.info("Partition '%s' n_clients=%d, sizes=%s",
                         self.strategy, self.num_clients,
                         [r["n_samples"] for r in rows])
