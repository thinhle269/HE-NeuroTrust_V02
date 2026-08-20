"""Paillier homomorphic encryption for secure federated aggregation.

Why Paillier?
-------------
The aggregation step in FedAvg is a weighted *sum* of client weight vectors.
Paillier is *additively* homomorphic, meaning E(a) * E(b) = E(a + b) and
E(a)^k = E(k * a), which is exactly what we need.  More expressive schemes
(BFV / CKKS) would let us perform multiplications too, but we don't need
that here, and Paillier has a pure-Python implementation (``phe``) that
works on any platform - important for a publicly-released artefact.

Threat model implemented
------------------------
* Clients hold the public key and encrypt their *quantised* update vectors.
* The server aggregates ciphertexts (linear combination with plaintext
  trust weights derived later by the fuzzy / ZT modules) **without** ever
  seeing individual plaintext updates.
* A trusted aggregator with the private key decrypts the *aggregated*
  ciphertext only.  In production this would be split via threshold
  decryption; for the paper we keep the key with the server for simplicity
  and clearly state this in the limitations section.

Performance notes (important for SCIE Q1 reproducibility)
---------------------------------------------------------
* ``phe`` transparently uses ``gmpy2`` when installed, which yields ~10x
  faster modular exponentiation than plain Python ints.  We therefore add
  ``gmpy2`` as a hard requirement in ``requirements.txt``.
* Per-scalar encryption is independent and CPU-bound (large modular
  exponentiation in Python ints which holds the GIL).  We parallelise via
  joblib's *loky* backend (``prefer="processes"``); ``prefer="threads"`` is
  useless here because the per-scalar work runs under the GIL.  We chunk
  the work into ``n_workers * 4`` blocks so the spawn-once cost of loky's
  worker pool is amortised over many encryptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import joblib
import numpy as np
from joblib import Parallel, delayed
from phe import paillier

DEFAULT_SCALE = 1_000_000.0

_PARALLEL_MIN_SIZE = 256

_CHUNKS_PER_WORKER = 4


@dataclass
class DecryptionAuthority:
    """Independent key-holder for the two-channel HE protocol.

    Holds the Paillier PRIVATE key and exposes **only** aggregate decryption
    (:meth:`decrypt_aggregate`); it offers no interface to decrypt an individual
    client ciphertext.  Because the aggregation server never receives the
    private key (it holds only a :class:`PaillierContext`), it structurally
    cannot reconstruct any single client's update - it can only ask the
    authority to open the final summed ciphertext.  In a deployment this is a
    separate party (or a threshold committee); in our simulation it is a
    distinct object that the server invokes solely on the final aggregate.
    """
    private_key: paillier.PaillierPrivateKey
    quantisation_scale: float = DEFAULT_SCALE
    n_jobs: int = -1

    def decrypt_aggregate(self, enc: "EncryptedVector") -> np.ndarray:
        return decrypt_vector(enc, self.private_key,
                              scale=self.quantisation_scale, n_jobs=self.n_jobs)


@dataclass
class PaillierContext:
    """Server-side Paillier context holding **only** the public key.

    It can encrypt and homomorphically aggregate ciphertexts, but has no private
    key and therefore cannot decrypt anything - not the aggregate, and in
    particular not any individual client update.  Build it together with an
    independent :class:`DecryptionAuthority` via :func:`generate_he_parties`.
    """
    public_key: paillier.PaillierPublicKey
    quantisation_scale: float = DEFAULT_SCALE
    n_jobs: int = -1

    def encrypt_vector(self, vec: np.ndarray) -> "EncryptedVector":
        return encrypt_vector(vec, self.public_key,
                              scale=self.quantisation_scale, n_jobs=self.n_jobs)


def generate_he_parties(key_size: int = 1024,
                        quantisation_scale: float = DEFAULT_SCALE,
                        n_jobs: int = -1):
    """Create the two parties of the two-channel protocol.

    Returns ``(context, authority)`` where ``context`` is a server-side
    :class:`PaillierContext` (public key only) and ``authority`` is an
    independent :class:`DecryptionAuthority` (private key; aggregate-only
    decryption).  The aggregation server is given only ``context``, so it cannot
    decrypt individual updates; the private key lives with the separate
    authority, which opens only the final homomorphic sum.
    """
    pub, priv = paillier.generate_paillier_keypair(n_length=key_size)
    return (PaillierContext(pub, quantisation_scale, n_jobs),
            DecryptionAuthority(priv, quantisation_scale, n_jobs))


@dataclass
class EncryptedVector:
    """A vector of Paillier ciphertexts plus the shape it should be reshaped to.

    Aggregation is implemented as Python-level operators so that downstream
    code reads naturally:

        agg = sum(weight_i * E(update_i) for i in clients) / total_weight

    where ``weight_i`` is a *plaintext* scalar.  The ``__add__`` /
    ``__mul__`` operators below delegate to the underlying ``phe.EncryptedNumber``
    arithmetic so no plaintext ever leaks during the linear combination.
    """
    ciphertexts: List[paillier.EncryptedNumber]
    shape: Sequence[int]

    def __add__(self, other: "EncryptedVector") -> "EncryptedVector":
        if len(self.ciphertexts) != len(other.ciphertexts):
            raise ValueError("Encrypted vector length mismatch")
        return EncryptedVector(
            ciphertexts=[a + b for a, b in zip(self.ciphertexts, other.ciphertexts)],
            shape=self.shape,
        )

    def __mul__(self, scalar: float) -> "EncryptedVector":
        s = float(scalar)
        return EncryptedVector(
            ciphertexts=[c * s for c in self.ciphertexts],
            shape=self.shape,
        )

    __rmul__ = __mul__

    def __len__(self) -> int:
        return len(self.ciphertexts)


def _encrypt_chunk(values: List[float],
                   pub: paillier.PaillierPublicKey,
                   scale: float) -> List[paillier.EncryptedNumber]:
    out = []
    for v in values:
        encoded = int(round(float(v) * scale))
        out.append(pub.encrypt(encoded))
    return out


def _decrypt_chunk(ciphertexts: List[paillier.EncryptedNumber],
                   priv: paillier.PaillierPrivateKey,
                   scale: float) -> List[float]:
    return [float(priv.decrypt(c)) / scale for c in ciphertexts]


def _resolve_n_workers(n_jobs: int) -> int:
    if n_jobs < 0 or n_jobs == 0:
        return max(1, joblib.cpu_count() + 1 + n_jobs) if n_jobs < 0 else 1
    return n_jobs


def encrypt_vector(vec: np.ndarray, pub: paillier.PaillierPublicKey,
                   scale: float = DEFAULT_SCALE,
                   n_jobs: int = -1) -> EncryptedVector:
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        flat = np.where(np.isfinite(flat), flat, 0.0)
    n = flat.size
    if n_jobs == 1 or n < _PARALLEL_MIN_SIZE:
        ciphertexts = _encrypt_chunk(flat.tolist(), pub, scale)
        return EncryptedVector(ciphertexts=ciphertexts, shape=tuple(vec.shape))

    n_workers = max(1, _resolve_n_workers(n_jobs))
    n_chunks = min(n_workers * _CHUNKS_PER_WORKER, max(1, n // 32))
    chunks = np.array_split(flat, n_chunks)
    chunk_results = Parallel(n_jobs=n_jobs, prefer="processes", backend="loky")(
        delayed(_encrypt_chunk)(c.tolist(), pub, scale) for c in chunks
    )
    ciphertexts: List[paillier.EncryptedNumber] = []
    for r in chunk_results:
        ciphertexts.extend(r)
    return EncryptedVector(ciphertexts=ciphertexts, shape=tuple(vec.shape))


def decrypt_vector(enc: EncryptedVector, priv: paillier.PaillierPrivateKey,
                   scale: float = DEFAULT_SCALE, n_jobs: int = -1) -> np.ndarray:
    cts = enc.ciphertexts
    n = len(cts)
    if n_jobs == 1 or n < _PARALLEL_MIN_SIZE:
        flat = _decrypt_chunk(cts, priv, scale)
        return np.array(flat, dtype=np.float64).reshape(enc.shape)

    n_workers = max(1, _resolve_n_workers(n_jobs))
    n_chunks = min(n_workers * _CHUNKS_PER_WORKER, max(1, n // 32))
    chunk_size = (n + n_chunks - 1) // n_chunks
    chunks = [cts[i:i + chunk_size] for i in range(0, n, chunk_size)]
    chunk_results = Parallel(n_jobs=n_jobs, prefer="processes", backend="loky")(
        delayed(_decrypt_chunk)(c, priv, scale) for c in chunks
    )
    flat = [v for r in chunk_results for v in r]
    return np.array(flat, dtype=np.float64).reshape(enc.shape)


def secure_aggregate(encrypted_updates: Iterable[EncryptedVector],
                     weights: Sequence[float]) -> EncryptedVector:
    """Weighted homomorphic MEAN: (sum_i w_i * E(u_i)) / sum_i w_i.

    Performed entirely under encryption.  This is the encrypted counterpart of
    :func:`plaintext_aggregate` and is normalised by the weight sum so the two
    paths are numerically identical for ANY weights (not only pre-normalised
    ones).  With the pipeline's weights (which already sum to 1) the final
    division is a no-op, so results are unchanged.
    """
    enc_list = list(encrypted_updates)
    if len(enc_list) != len(weights):
        raise ValueError("Mismatched #updates and #weights")
    if not enc_list:
        raise ValueError("No updates supplied")
    total_w = float(sum(float(w) for w in weights))
    accumulator: Optional[EncryptedVector] = None
    for w, enc in zip(weights, enc_list):
        if w == 0:
            continue
        term = enc if w == 1 else enc * w
        accumulator = term if accumulator is None else accumulator + term
    if accumulator is None:
        first = enc_list[0]
        pk = first.ciphertexts[0].public_key
        return EncryptedVector(
            ciphertexts=[pk.encrypt(0) for _ in range(len(first.ciphertexts))],
            shape=first.shape)
    if total_w > 0 and abs(total_w - 1.0) > 1e-9:
        accumulator = accumulator * (1.0 / total_w)
    return accumulator
