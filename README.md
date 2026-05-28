# HWOA-RL-XAI: Hybrid Whale Optimization–Reinforcement Learning with Explainable AI for Energy-Efficient Cloud Scheduling

[![Paper](https://img.shields.io/badge/paper-under_review-blue)](https://github.com/hwoa-rl-xai/cloud-scheduler)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-orange)](tests/)

> **Official implementation** of *"HWOA-RL-XAI: A Hybrid Whale Optimization–Reinforcement Learning Framework with Explainable AI for Energy-Efficient Workload Scheduling in Large-Scale Heterogeneous Cloud Infrastructures"*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Pre-trained Models](#pre-trained-models)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Citation](#citation)
- [License](#license)

---

## Overview

**HWOA-RL-XAI** is a four-layer hybrid framework for cloud workload scheduling that tightly integrates:

| Layer | Component | Role |
|-------|-----------|------|
| Prediction | LSTM-Autoencoder (LSTM-AE) | Dual-function workload forecasting + anomaly detection |
| Optimization | Whale Optimization Algorithm (WOA) | Meta-controller for RL hyperparameter tuning via adaptive Lévy flight |
| Scheduling | Proximal Policy Optimization (PPO-RL) | Fine-grained task-resource mapping |
| Explainability | SHAP (Shapley Additive Explanations) | Closed-loop explainability feeding feature rankings into reward shaping |

The framework targets three simultaneous objectives: **makespan minimization**, **SLA compliance**, and **energy efficiency**, evaluated at scales up to 500 heterogeneous VMs and on a real 32-node Kubernetes (v1.28) cluster.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HWOA-RL-XAI Pipeline                   │
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Prediction │───▶│Optimization │───▶│Explainability│ │
│  │  LSTM-AE    │    │  WOA + PPO  │    │    SHAP     │ │
│  └─────────────┘    └─────────────┘    └──────┬──────┘ │
│         ▲                  ▲                  │        │
│         │                  └──────────────────┘        │
│         │              (reward shaping feedback)        │
│  ┌──────┴──────────────────────────────────────────┐   │
│  │          Kubernetes Extender / CloudSim          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Results

At **500-VM scale** (CloudSim), HWOA-RL-XAI vs. best baseline (CNN-LSTM-PSO-GA):

| Metric | HWOA-RL-XAI | Best Baseline | Improvement |
|--------|-------------|---------------|-------------|
| Makespan (s) | **169.7 ± 3.4** | 198.3 ± 6.9 | **−14.4%** |
| SLA Violation (%) | **6.1 ± 0.6** | 19.2 ± 1.8 | **−68.2%** |
| Energy (kWh) | **75.2 ± 1.8** | 126.7 ± 5.6 | **−40.6%** |
| CPU Utilization (%) | **89.7 ± 1.2** | 63.8 ± 2.8 | **+40.6%** |

All improvements significant at p < 0.001, Cohen's d > 2.4 (very large effect size).

**Real Kubernetes cluster (32 nodes)** vs. default scheduler: 26.8% makespan reduction, 71.4% SLA violation reduction, 28.4% energy savings at 17.5 ms per-epoch critical-path overhead.

---

## Repository Structure

```
hwoa-rl-xai/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/                        # YAML experiment configurations
│   ├── default.yaml                # Default hyperparameters
│   ├── cloudsim_50vm.yaml
│   ├── cloudsim_200vm.yaml
│   ├── cloudsim_500vm.yaml
│   ├── kubernetes.yaml
│   └── ablation/                   # Per-ablation configs
│       ├── no_woa.yaml
│       ├── no_rl.yaml
│       ├── no_lstm.yaml
│       ├── no_levy.yaml
│       └── no_shap_loop.yaml
│
├── src/
│   ├── environments/
│   │   ├── cloudsim_env.py         # CloudSim Gym-compatible environment
│   │   └── kubernetes_env.py       # Kubernetes scheduler extender env
│   │
│   ├── models/
│   │   ├── lstm_autoencoder.py     # LSTM-AE for forecasting + anomaly detection
│   │   └── ppo_network.py          # PPO actor-critic network
│   │
│   ├── agents/
│   │   └── ppo_agent.py            # PPO agent with SHAP reward shaping
│   │
│   ├── optimizers/
│   │   └── woa_optimizer.py        # WOA with adaptive Lévy flight
│   │
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP computation + reward bonus calculation
│   │
│   ├── baselines/
│   │   ├── fcfs.py
│   │   ├── min_min.py
│   │   ├── dql_agent.py
│   │   ├── a3c_agent.py
│   │   ├── cnn_lstm_pso_ga.py
│   │   └── lstm_only.py
│   │
│   └── utils/
│       ├── metrics.py              # KPI computation (makespan, SLA, DMR, etc.)
│       ├── data_loader.py          # Google Cluster Traces + RUBiS data loading
│       ├── seeds.py                # Reproducibility seed management
│       └── visualization.py        # Result plotting utilities
│
├── experiments/
│   ├── train_hwoa_rl_xai.py        # Main training script
│   ├── evaluate.py                  # Evaluation against baselines
│   ├── ablation_study.py            # Systematic ablation runner
│   ├── sensitivity_analysis.py      # Hyperparameter sensitivity sweep
│   └── convergence_analysis.py      # Convergence curve generation
│
├── datasets/
│   ├── README.md                    # Dataset download and preparation guide
│   ├── google_cluster/
│   │   └── preprocess.py           # Google Cluster Traces preprocessing
│   └── rubis/
│       └── setup.sh                # RUBiS benchmark setup script
│
├── k8s/
│   ├── scheduler-extender/
│   │   ├── Dockerfile
│   │   ├── app.py                  # Flask API for Kubernetes extender
│   │   └── requirements.txt
│   ├── manifests/
│   │   ├── scheduler-extender-deployment.yaml
│   │   ├── scheduler-policy.yaml
│   │   ├── lstm-ae-sidecar.yaml
│   │   └── shap-worker.yaml
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lstm_ae_training.ipynb
│   ├── 03_woa_rl_training.ipynb
│   ├── 04_results_visualization.ipynb
│   └── 05_shap_dashboard.ipynb
│
├── results/
│   ├── cloudsim/
│   │   ├── 50vm/
│   │   ├── 200vm/
│   │   └── 500vm/
│   ├── kubernetes/
│   ├── ablation/
│   └── figures/                    # Publication-ready figures (PNG + PDF)
│
├── tests/
│   ├── test_lstm_ae.py
│   ├── test_woa_optimizer.py
│   ├── test_ppo_agent.py
│   ├── test_shap_explainer.py
│   ├── test_metrics.py
│   └── test_environments.py
│
├── scripts/
│   ├── download_datasets.sh        # Automated dataset download
│   ├── run_all_experiments.sh      # Full reproduction pipeline
│   └── setup_kubernetes.sh         # K8s cluster setup
│
└── docs/
    ├── REPRODUCIBILITY.md
    ├── HYPERPARAMETERS.md
    └── KUBERNETES_SETUP.md
```

---

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- Docker (for Kubernetes experiments)
- Java 11+ (for CloudSim)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/hwoa-rl-xai/cloud-scheduler.git
cd cloud-scheduler

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

```
torch>=2.0.0
stable-baselines3>=2.0.0
shap>=0.43.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
gymnasium>=0.29.0
flask>=3.0.0
prometheus-client>=0.19.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Datasets

### 1. Google Cluster Workload Traces (2011)

The primary dataset consists of 29 days of resource consumption logs from a Google production cluster with 12,500+ machines and ~25 million task events.

```bash
# Download and preprocess (requires ~50 GB disk space)
bash scripts/download_datasets.sh --dataset google_cluster

# Or manually:
# 1. Download from: https://github.com/google/cluster-data
# 2. Preprocess:
python datasets/google_cluster/preprocess.py \
    --raw_dir data/raw/google_cluster \
    --out_dir data/processed/google_cluster \
    --features cpu_rate,memory,scheduling_class,priority,start_time,end_time
```

The preprocessor extracts: CPU proportion, memory demand, scheduling class, task priority, and timestamps. It splits the data into train (days 1–20), validation (days 21–24), and test (days 25–29) partitions.

**Data statistics:**

| Split | Tasks | Duration | CPU (mean) | Memory (mean) |
|-------|-------|----------|------------|---------------|
| Train | 18.2M | 20 days | 0.063 | 0.032 |
| Val | 3.6M | 4 days | 0.061 | 0.031 |
| Test | 3.2M | 5 days | 0.064 | 0.033 |

### 2. RUBiS E-Commerce Benchmark

RUBiS is a multi-tier auction platform benchmark simulating interactive user sessions across 12 microservice types.

```bash
# Set up RUBiS in Docker
bash datasets/rubis/setup.sh

# Or via Docker Compose:
cd datasets/rubis
docker-compose up -d
```

RUBiS provides HTTP request traces, per-service CPU/memory utilization, and end-to-end latency measurements at configurable concurrency levels (we use 500 concurrent users).

### Data Directory Layout (after download)

```
data/
├── raw/
│   ├── google_cluster/          # Raw .csv.gz files from Google
│   └── rubis/                   # Raw benchmark logs
└── processed/
    ├── google_cluster/
    │   ├── train.parquet
    │   ├── val.parquet
    │   └── test.parquet
    └── rubis/
        ├── traces.parquet
        └── service_metrics.parquet
```

---

## Quick Start

### Training HWOA-RL-XAI (CloudSim, 200 VMs)

```bash
python experiments/train_hwoa_rl_xai.py \
    --config configs/cloudsim_200vm.yaml \
    --seed 42 \
    --output_dir results/cloudsim/200vm/run_seed42
```

### Evaluating Against All Baselines

```bash
python experiments/evaluate.py \
    --checkpoint results/cloudsim/200vm/run_seed42/best_model.pt \
    --config configs/cloudsim_200vm.yaml \
    --baselines fcfs,min_min,dql,a3c,cnn_lstm_pso_ga,lstm_only \
    --seeds 42,43,44,45,46,47,48,49,50,51 \
    --output results/cloudsim/200vm/evaluation.json
```

### Running the Ablation Study

```bash
python experiments/ablation_study.py \
    --base_config configs/cloudsim_200vm.yaml \
    --ablation_configs configs/ablation/ \
    --seeds 42,43,44,45,46,47,48,49,50,51 \
    --output results/ablation/
```

---

## Reproducing Experiments

All experiments from the paper can be reproduced with:

```bash
bash scripts/run_all_experiments.sh
```

This runs (in sequence):

1. **CloudSim 50 VM** (10 seeds × 6 baselines + HWOA-RL-XAI) — ~2 hours on a single GPU
2. **CloudSim 200 VM** — ~6 hours
3. **CloudSim 500 VM** — ~18 hours
4. **Ablation study** (200 VM scale) — ~4 hours
5. **Hyperparameter sensitivity sweep** — ~3 hours
6. **Convergence analysis** — ~2 hours

**Estimated total compute:** ~35 GPU-hours (NVIDIA A100) or ~70 CPU-hours.

### Random Seeds

All experiments use seeds 42–51 (10 independent runs). Seeds are fixed for:
- NumPy: `np.random.seed(seed)`
- PyTorch: `torch.manual_seed(seed)`
- Python: `random.seed(seed)`
- Environment: passed as `env_seed` to CloudSim

See [`src/utils/seeds.py`](src/utils/seeds.py) for the seeding utility.

### Hyperparameters

Full hyperparameter configuration is in [`configs/default.yaml`](configs/default.yaml). Key values used in the paper:

| Component | Parameter | Value |
|-----------|-----------|-------|
| LSTM-AE | Encoder layers | 128, 64 units |
| LSTM-AE | Learning rate | 1×10⁻³ |
| LSTM-AE | Batch size | 64 |
| LSTM-AE | Max epochs | 200 (early stopping, patience=20) |
| LSTM-AE | Loss weights (λ₁, λ₂, λ₃) | 0.5, 0.3, 0.2 |
| WOA | Population size N | 30 |
| WOA | Max iterations T | 100 |
| WOA | Lévy stability index β | 1.5 |
| WOA | Diversity threshold δ_div | 0.1 |
| PPO | Hidden layers | 2 × 256 (ReLU) |
| PPO | Discount factor γ | 0.98 |
| PPO | Clip parameter ε | 0.2 |
| PPO | SHAP weight w₅ | 0.05 |
| Objective | Weights (w₁, w₂, w₃, w₄) | 0.35, 0.30, 0.25, 0.10 |

See [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) for the full sensitivity analysis and the grid search protocol used to select these values.

---

## Pre-trained Models

Pre-trained model checkpoints for all three VM scales are available:

| Scale | Checkpoint | Size | Validation Loss |
|-------|-----------|------|-----------------|
| 50 VM | `pretrained/hwoa_rl_xai_50vm.pt` | 48 MB | — |
| 200 VM | `pretrained/hwoa_rl_xai_200vm.pt` | 48 MB | — |
| 500 VM | `pretrained/hwoa_rl_xai_500vm.pt` | 48 MB | — |
| LSTM-AE (Google Cluster) | `pretrained/lstm_ae_google.pt` | 12 MB | RMSE 3.1% |

Download all pre-trained models:

```bash
bash scripts/download_pretrained.sh
# Models are saved to pretrained/
```

To evaluate using a pre-trained checkpoint:

```bash
python experiments/evaluate.py \
    --checkpoint pretrained/hwoa_rl_xai_500vm.pt \
    --config configs/cloudsim_500vm.yaml \
    --seeds 42,43,44,45,46,47,48,49,50,51
```

---

## Kubernetes Deployment

Full deployment instructions are in [`k8s/README.md`](k8s/README.md).

### Quick Deploy

```bash
# 1. Build the scheduler extender image
docker build -t hwoa-rl-xai-extender:latest k8s/scheduler-extender/

# 2. Deploy to your cluster
kubectl apply -f k8s/manifests/

# 3. Verify the extender is running
kubectl get pods -n kube-system | grep hwoa-rl-xai
```

The extender intercepts Pod scheduling requests from kube-scheduler, queries the HWOA-RL-XAI policy server, and returns scored node rankings within the 100 ms Kubernetes scheduling timeout. The LSTM-AE sidecar publishes forecasts and anomaly alerts to a Redis message bus. SHAP attributions are published to a Grafana dashboard every 60 seconds.

**Resource requirements per component:**

| Component | CPU Request | Memory |
|-----------|------------|--------|
| Extender API | 200m | 512 Mi |
| LSTM-AE Sidecar | 500m | 1 Gi |
| SHAP Worker | 200m | 512 Mi |
| Redis | 100m | 256 Mi |
| **Total** | **~1 core** | **~2.3 GB** |

---

## Evaluation Metrics

Five primary KPIs are reported, matching exactly the definitions in the paper (Section 5.6):

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Makespan** | `max{i,j}(T_ij · x_ij)` across all tasks | ↓ Lower is better |
| **SLA Violation Rate** | % tasks breaching negotiated SLA thresholds | ↓ |
| **Deadline Miss Rate (DMR)** | % tasks completing after their deadline | ↓ |
| **CPU Utilization** | `(Σ C_i · x_ij) / (Σ M_j · T_epoch)` | ↑ Higher is better |
| **Energy Consumption** | `Σ E_ij · x_ij` (kWh) | ↓ |

Statistical testing: one-way ANOVA + post-hoc Tukey HSD + Cohen's d effect sizes. All implemented in [`src/utils/metrics.py`](src/utils/metrics.py).

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_woa_optimizer.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Citation

If you use this code or the paper's ideas in your research, please cite:

```bibtex
@article{hwoa_rl_xai_2025,
  title     = {{HWOA-RL-XAI}: A Hybrid Whale Optimization--Reinforcement Learning Framework
               with Explainable {AI} for Energy-Efficient Workload Scheduling in
               Large-Scale Heterogeneous Cloud Infrastructures},
  author    = {[Authors]},
  journal   = {[Journal]},
  year      = {2025},
  note      = {Under Review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Google Cluster Workload Traces: [https://github.com/google/cluster-data](https://github.com/google/cluster-data)
- RUBiS Benchmark: [http://rubis.ow2.org/](http://rubis.ow2.org/)
- CloudSim Toolkit: Calheiros et al. (2011)
- WOA: Mirjalili & Lewis (2016)
- SHAP: Lundberg & Lee (2017)
- PPO: Schulman et al. (2017)
