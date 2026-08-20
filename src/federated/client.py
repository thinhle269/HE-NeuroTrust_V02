"""Federated client: local training and (optional) malicious behaviour."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .aggregation import flatten_state


@dataclass
class ClientUpdate:
    """Container holding everything the server needs from a single client."""
    client_id: int
    flat_update: np.ndarray          # w_local - w_global  (plaintext, float32)
    encrypted_update: Optional[object] = None  # phe EncryptedVector if HE is on
    loss_before: float = 0.0
    loss_after: float = 0.0
    n_samples: int = 0
    train_time_sec: float = 0.0
    is_malicious: bool = False


class FederatedClient:
    """Stateless-ish local trainer.

    The model architecture is shared across clients - we copy global weights
    into a private deep-copy at the start of each round and never persist
    optimiser state, mirroring the realistic FL deployment where clients are
    short-lived IoT devices.
    """

    def __init__(self, client_id: int, model_template: nn.Module,
                 train_loader: DataLoader, device: torch.device,
                 lr: float = 1e-3, local_epochs: int = 1,
                 is_malicious: bool = False,
                 malicious_attack: str = "label_flip",
                 noise_sigma: float = 1.0,
                 num_classes: int = 2,
                 seed: int = 0,
                 proximal_mu: float = 0.0,
                 optimizer: str = "sgd",
                 grad_clip_norm: float = 1.0,
                 max_update_norm: float = 100.0,
                 attack_schedule: Optional[list] = None):
        self.client_id = int(client_id)
        self.model_template = model_template
        self.train_loader = train_loader
        self.device = device
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)
        self.is_malicious = bool(is_malicious)
        self.malicious_attack = malicious_attack
        self.noise_sigma = float(noise_sigma)
        self.attack_schedule = list(attack_schedule) if attack_schedule else None
        self.num_classes = int(num_classes)
        self.proximal_mu = float(proximal_mu)
        self.optimizer_kind = str(optimizer).lower()
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm and grad_clip_norm > 0 else None
        self.max_update_norm = float(max_update_norm) if max_update_norm and max_update_norm > 0 else None
        self._rng = np.random.default_rng(seed + client_id)
        self.n_samples = sum(b[0].size(0) for b in train_loader) if train_loader else 0

    def _attack_at_round(self, round_idx: int) -> str:
        """Resolve which attack type to apply in this round.

        Without an ``attack_schedule`` this simply returns the static
        ``malicious_attack``.  With a schedule we use the *latest* entry
        whose ``start_round`` is <= ``round_idx``.  This implements the
        time-varying threat-hunting scenario in §6.X of the manuscript.
        """
        if not self.attack_schedule:
            return self.malicious_attack
        current = self.malicious_attack
        for start_round, atk in self.attack_schedule:
            if int(start_round) <= int(round_idx):
                current = str(atk)
        return current

    def train_round(self, global_state, round_idx: int = 0) -> ClientUpdate:
        """Run ``local_epochs`` epochs of SGD on local data and return an update.

        When :attr:`proximal_mu` > 0 the loss includes the FedProx proximal
        term ``mu/2 * sum_l ||w_local_l - w_global_l||^2``.  This anchors the
        local optimum to the global model and is what differentiates FedProx
        from FedAvg under statistical heterogeneity.
        """
        current_attack = self._attack_at_round(round_idx)
        import time
        local = copy.deepcopy(self.model_template).to(self.device)
        local.load_state_dict(global_state)
        local.train()
        if self.optimizer_kind == "adam":
            optim = torch.optim.Adam(local.parameters(), lr=self.lr)
        else:
            optim = torch.optim.SGD(local.parameters(), lr=self.lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        prox_anchors = None
        if self.proximal_mu > 0:
            prox_anchors = [p.detach().clone().to(self.device)
                            for p in local.parameters()]

        loss_before = self._compute_loss(local, loss_fn)
        t0 = time.time()
        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.is_malicious and current_attack == "label_flip":
                    y = self._flip_labels(y)
                optim.zero_grad(set_to_none=True)
                logits = local(X)
                loss = loss_fn(logits, y)
                if prox_anchors is not None:
                    prox = 0.0
                    for p, anchor in zip(local.parameters(), prox_anchors):
                        prox = prox + ((p - anchor) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * prox
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(local.parameters(),
                                                   max_norm=self.grad_clip_norm)
                optim.step()
        train_time = time.time() - t0
        loss_after = self._compute_loss(local, loss_fn)

        global_vec, _ = flatten_state(global_state)
        local_vec, _ = flatten_state(local.state_dict())
        delta = (local_vec - global_vec).astype(np.float32)

        if self.is_malicious:
            if current_attack == "sign_flip":
                delta = -delta
            elif current_attack == "gaussian_noise":
                delta = delta + self._rng.normal(0.0, self.noise_sigma,
                                                 size=delta.shape).astype(np.float32)

        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_update_norm is not None:
            norm = float(np.linalg.norm(delta))
            if np.isfinite(norm) and norm > self.max_update_norm:
                delta = delta * (self.max_update_norm / norm)

        return ClientUpdate(
            client_id=self.client_id,
            flat_update=delta,
            encrypted_update=None,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            n_samples=int(self.n_samples),
            train_time_sec=float(train_time),
            is_malicious=self.is_malicious,
        )

    def _compute_loss(self, model: nn.Module, loss_fn: nn.Module) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for X, y in self.train_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = model(X)
                total += float(loss_fn(logits, y).item()) * X.size(0)
                count += X.size(0)
        model.train()
        return total / max(count, 1)

    def _flip_labels(self, y: torch.Tensor) -> torch.Tensor:
        return (y + 1) % self.num_classes
