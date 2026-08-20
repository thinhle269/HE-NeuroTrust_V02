# HE-NeuroTrust

**Homomorphic Encryption and Neuro-Fuzzy Zero-Trust for Privacy-Preserving, Byzantine-Robust Federated Intrusion Detection in IoT**

 .

HE-NeuroTrust reconciles them with a **two-channel design**:

| Channel | What crosses the network | Who can read it |
|---|---|---|
| **Confidential** | `E(Δwᵢ)` — the model update encrypted under Paillier | Nobody individually. The aggregation server holds **only the public key**; an independent decryption authority holds the secret key and opens **only the aggregate**. |
| **Behavioural** | `aᵢ = (cosᵢ, Δlossᵢ, volᵢ)` — a 3-scalar attestation | The server, which uses it to score trust. It reveals no individual model parameter. |

A **differentiable neuro-fuzzy (ANFIS)** engine maps the attestation to a trust score — its Gaussian membership functions are *learned* from labelled attestations rather than hand-tuned — and a **continuous Zero-Trust policy** (EMA smoothing, capped reject-streak, participation floor) turns that score into a gated, trust-weighted homomorphic aggregation.

---

## 1. Headline results

CIC-IoT-2023, 10 clients, Dirichlet non-IID (α = 0.5), **30 % Byzantine (sign-flip)**, 30 rounds, **mean over 5 seeds**.

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Centralized (pooled-data reference) | 0.799 ± 0.004 | 0.634 ± 0.003 | 0.795 ± 0.004 |
| FedAvg (clean, no attack) | 0.773 ± 0.008 | 0.603 ± 0.005 | 0.766 ± 0.008 |
| FedAvg + attack (no defence) | 0.674 ± 0.034 | 0.486 ± 0.040 | 0.621 ± 0.050 |
| FedAvg + HE (privacy only) | 0.674 ± 0.034 | 0.487 ± 0.038 | 0.620 ± 0.049 |
| Coordinate-wise Median | 0.691 ± 0.041 | 0.509 ± 0.042 | 0.645 ± 0.066 |
| Krum | 0.693 ± 0.054 | 0.505 ± 0.073 | 0.647 ± 0.083 |
| Bulyan¹ | 0.694 ± 0.043 | 0.512 ± 0.042 | 0.648 ± 0.066 |
| FoolsGold | 0.691 ± 0.050 | 0.500 ± 0.076 | 0.641 ± 0.073 |
| **HE-NeuroTrust (Mamdani)** | **0.773 ± 0.006** | **0.596 ± 0.018** | **0.765 ± 0.007** |
| **HE-NeuroTrust (neuro-fuzzy)** | **0.767 ± 0.022** | **0.593 ± 0.024** | **0.757 ± 0.025** |

 

---

 
---

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.9. **`gmpy2` matters a lot**: `phe` auto-detects it and it gives roughly a 10× speed-up on the modular exponentiation that dominates Paillier encryption. Without it an encrypted round takes minutes instead of seconds.

---

## 3. Datasets

The datasets are third-party and are **not redistributed here**. Download them and point the code at your local copies.

| Dataset | Used for | Where the path is set |
|---|---|---|
| **CIC-IoT-2023** | primary IoT IDS benchmark (8 classes, 36 flow features after preprocessing) | `configs/default.yaml → paths.raw_csv_dir` |
| **Edge-IIoTset** | second, independent IIoT IDS benchmark | `run_all.py → NPZ_DATASETS["edgeiiot"]` |
| **CMU keystroke** | behavioural biometric (51-user identification) | `run_all.py → NPZ_DATASETS["keystroke"]` |
| **HMOG** | mobile continuous authentication (20-user identification, (128×6) sensor windows) | `src/data/hmog_preprocessor.py → HMOG_DIR_DEFAULT` and `run_hmog_verification.py → HMOG_DIR` |

> ⚠️ These currently contain **absolute Windows paths from the development machine** (e.g. `D:/Chi Van/CIC_IoT_Attack_2023`). Edit them for your environment before running.

Preprocessing (stratified 70/15/15 split, ≤ 50 000 flows per class, scaler **fit on the training split only**) is cached under `data/processed/` on first run.

---

 
 


## 4. Seeds and reproducibility

### Where the seeds live

The five seeds quoted in the paper are declared **once**, in the config — there is no seed hardcoded anywhere in the experiment code:

```yaml
# configs/default.yaml
seed:  42                          # legacy single-seed fallback
seeds: [42, 123, 2024, 7, 2025]    # 5 seeds -> mean ± std + paired significance tests
```

Every runner resolves them through one helper, with the precedence **CLI flag > config list > scalar fallback**:

```python
# run_all.py
def resolve_seeds(cfg, cli_seeds):
    if cli_seeds:            return [int(s) for s in cli_seeds]   # e.g. --seeds 42 123
    seeds = cfg.get("seeds")
    if seeds:                return [int(s) for s in seeds]       # <- the paper's five seeds
    return [int(cfg.seed)]
```

So `python run_all.py` with no arguments reproduces exactly the five-seed protocol reported in the paper. (`run_ablation.py` and `run_attack_study.py` call the same `run_all.run_experiment()`, so the resolution rule is shared.)

 