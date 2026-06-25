# REND: deep Reinforcement learning for cost-Effective Network Dismantling

This repository provides the reference implementation of **REND** and the interpretable **Boundary-Cut** heuristic, as described in our paper:

> **Goodhart's trap in cost-constrained network dismantling and its remedy.**
> Changjun Fan, Feng Qing, Li Zeng, Suoyi Tan, Aming Li, Xin Lu, Jincai Huang, Zhong Liu.

![REND framework](./paper/REND_framework.svg)

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Baseline Methods](#baseline-methods)
- [Results](#results)
- [Data and Code Availability](#data-and-code-availability)
- [Citation](#citation)

## Overview

Cost-constrained network dismantling seeks an optimal set of nodes to remove so that a
network is fragmented under **heterogeneous removal costs**. We show that the conventional
metric — the Accumulated Normalized Connectivity (ANC) defined on the absolute Giant
Connected Component (GCC) size — falls into a **Goodhart's law trap**: because it rewards
any reduction in GCC size regardless of how many nodes are removed, a powerful optimizer
learns to harvest large numbers of cheap peripheral nodes (*trivial shrinkage*) instead of
attacking genuinely critical nodes (*structural fragmentation*).

To close this trap we make three contributions:

1. **RGCC** — a corrected measure, the *Relative Giant Connected Component*, that replaces
   the static denominator `N` with the dynamic number of remaining nodes `N − k`. RGCC
   explicitly penalizes excessive removal and distinguishes fragmentation from shrinkage.
2. **REND** — a deep reinforcement learning agent trained purely on small synthetic graphs
   to maximize the RGCC-based reward (optimized through a stable lower bound `R_lb`). On
   real-world networks REND removes **50–70% fewer nodes** yet generates **3–6× more
   fragments** than existing methods.
3. **Boundary-Cut** — an interpretable, training-free heuristic distilled from REND's
   learned policy. It iteratively removes *2-core boundary nodes* (nodes in the 2-core with
   at least one neighbour in the 1-shell) in descending order of a cost-effectiveness ratio,
   and matches the black-box model's performance with zero training.

The framework supports three cost scenarios for every network: **uniform-cost**,
**degree-cost** (cost proportional to node degree), and **random-cost** (random non-negative
weights).

## Repository Structure

```
FINDER_ND_cost/
├── code/                       # All source code (Cython extensions + Python drivers)
│   ├── lib/                    # C++ sources for the Cython extensions
│   ├── *.pyx / *.pxd / *.so    # Cython bindings (graph, mvc_env, utils, boundary_cut, ...)
│   ├── REND.py                 # Core REND model (encoder/decoder, training, evaluation)
│   ├── config.py               # Centralized hyper-parameters
│   ├── setup.py                # Build script for the Cython/C++ extensions
│   ├── train.py                # Train REND on synthetic BA graphs
│   ├── solReal_REND.py         # Generate REND dismantling solutions on real networks
│   ├── solReal_BoundaryCut.py  # Generate Boundary-Cut solutions on real networks
│   ├── testReal.py             # Evaluate (RGCC-ANC score) all methods on real networks
│   └── testSynthetic.py        # Evaluate REND on synthetic BA graphs
├── data/
│   ├── real/                   # Real-world networks + uniform/degree/random cost weights
│   └── synthetic/              # BA graphs for degree_cost / random_cost scenarios
├── models/                     # Pre-trained REND model (REND.ckpt)
├── results/
│   ├── real/                   # Per-method results (REND, boundarycut, GND, HDA, ...)
│   └── synthetic/              # Synthetic-network score tables
├── paper/                      # Manuscript and framework figure
├── requirements.txt
└── README.md
```

> **Note.** All code and the C++ sources live under `code/`. Scripts are intended to be run
> from inside `code/`; they reference the data, model and result folders one level up
> (`../data`, `../models`, `../results`).

## System Requirements

### Software dependencies

The package was developed and tested with **Python 3.6.13**. Exact versions are pinned in
[`requirements.txt`](./requirements.txt):

```
Cython==0.29.24
networkx==2.5.1
numpy==1.19.2
pandas==1.1.5
scipy==1.5.2
tensorflow==1.14.0
tqdm==4.64.1
```

A C++11-capable compiler is required to build the Cython extensions
(`gcc`/`g++` ≥ 7.4; tested with **gcc 11.4.0**).

### Operating system

Tested on **Linux (Ubuntu 22.04 LTS)**.

### Hardware

The experiments in this repository were run on a **server with a single GPU**. The exact
configuration used was:

| Component | Specification                                    |
| --------- | ------------------------------------------------ |
| GPU       | 1 × NVIDIA GeForce RTX 4090 (24 GB)              |
| CPU       | x86-64, 8+ cores                                 |
| RAM       | 32+ GB                                           |
| CUDA      | Compatible driver for the installed TensorFlow   |

Only **one GPU** is needed. Training uses the GPU; evaluation is light-weight and runs on
the CPU. The REND model is compact, so 24 GB of GPU memory is far more than sufficient and a
smaller GPU (≥ 8 GB) will also work.

## Installation Guide

1. Create the environment and install the dependencies (≈ 5 minutes):

```bash
conda create -n rend_env python=3.6.13
conda activate rend_env
pip install -r requirements.txt
```

2. Build the Cython/C++ extensions from inside `code/` (≈ 1 minute):

```bash
cd code
python setup.py build_ext --inplace
```

This produces the `*.so` extension modules next to their `*.pyx`/`*.pxd` definitions.

## Usage

All commands are run from inside the `code/` directory.

### 1. Train the model (GPU)

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python train.py
```

REND is trained on synthetic Barabási–Albert graphs (30–50 nodes). Hyper-parameters can be
tuned in [`code/config.py`](./code/config.py). Checkpoints are written to `../models`.
A pre-trained model (`models/REND.ckpt`) is already provided, so training is optional for
reproducing the results.

### 2. Generate dismantling solutions on real networks

```bash
# REND solutions (uses the trained model)
CUDA_VISIBLE_DEVICES=-1 python solReal_REND.py

# Boundary-Cut solutions (interpretable heuristic, no model required)
python solReal_BoundaryCut.py
```

Solutions (node-removal orders) are saved under `../results/real/<method>/<cost>_cost/`.

### 3. Evaluate on real networks

```bash
CUDA_VISIBLE_DEVICES=-1 python testReal.py
```

This computes the RGCC-based ANC score for every method and cost scenario and writes
`solution_<cost>_score.csv` and `MaxCCList_Strategy_<network>.txt` into the corresponding
result folders.

### 4. Evaluate on synthetic networks

```bash
CUDA_VISIBLE_DEVICES=-1 python testSynthetic.py
```

Reports mean ± std scores over 100 random BA instances per scale
(30-50, 50-100, 100-200, 200-300, 300-400, 400-500 nodes) for both degree-cost and
random-cost, writing the summaries to `../results/synthetic`.

> Evaluation does not require a GPU; set `CUDA_VISIBLE_DEVICES=-1` to force CPU execution.

## Baseline Methods

REND and Boundary-Cut are compared against state-of-the-art cost-aware and structural
dismantling baselines:

| Method            | Type                                   |
| ----------------- | -------------------------------------- |
| HDA               | Adaptive high-degree                   |
| CoreHD            | k-core based                           |
| MinSum            | Message passing                        |
| BPD               | Belief-propagation decimation          |
| CI                | Collective Influence                   |
| GND               | Generalized Network Dismantling        |
| GDM               | Graph-neural-network dismantling       |
| FINDER            | Prior deep-RL model (GCC-optimized)    |
| **REND**          | Deep-RL model (RGCC-optimized, ours)   |
| **Boundary-Cut**  | Interpretable heuristic (ours)         |

External baseline references:

```
https://github.com/zhfkt/ComplexCi                                  (CI)
https://github.com/abraunst/decycler                                (MinSum)
http://power.itp.ac.cn/~zhouhj/codes.html                           (BPD)
https://github.com/hcmidt/corehd                                    (CoreHD)
https://github.com/renxiaolong/Generalized-Network-Dismantling     (GND)
```

## Results

Pre-computed results are provided under [`results/`](./results):

- `results/real/<method>/{uniform,degree,random}_cost/` — per-method, per-cost results on
  the nine real-world networks (criminal, biological, communication, infrastructure and
  social networks).
- `results/synthetic/` — score tables for the synthetic BA benchmarks.

## Data and Code Availability

Real-world networks (collected from public repositories such as SNAP) and the synthetic BA
graphs used in the paper are provided under [`data/`](./data). Each real network ships with
`uniform`, `degree` and `random` cost-weight files. The pre-trained model is in
[`models/`](./models).

## Citation

If you find this code or paper useful, please cite our work:

```bibtex
@article{fan2026goodhart,
  title   = {Goodhart's trap in cost-constrained network dismantling and its remedy},
  author  = {Fan, Changjun and Qing, Feng and Zeng, Li and Tan, Suoyi and Li, Aming and Lu, Xin and Huang, Jincai and Liu, Zhong},
  year    = {2026}
}
```

This work builds on FINDER:

```bibtex
@article{fan2020finding,
  title     = {Finding key players in complex networks through deep reinforcement learning},
  author    = {Fan, Changjun and Zeng, Li and Sun, Yizhou and Liu, Yang-Yu},
  journal   = {Nature Machine Intelligence},
  volume    = {2},
  pages     = {317--324},
  year      = {2020},
  publisher = {Nature Publishing Group}
}
```
