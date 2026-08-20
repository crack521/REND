# REND: deep Reinforcement learning for cost-Effective Network Dismantling

This repository provides the reference implementation of **REND** and the interpretable **Boundary-Cut** heuristic, as described in our paper:

> **Mitigating the measurement problem for cost-constrained network dismantling with deep reinforcement learning.**
> Changjun Fan, Feng Qing, Li Zeng, Aming Li, Suoyi Tan, Xin Lu, Jincai Huang, Zhong Liu.

![REND framework](./paper/REND_framework.jpg)

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data and Results (CodeOcean)](#data-and-results-codeocean)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Baseline Methods](#baseline-methods)
- [Citation](#citation)

## Overview

Cost-constrained network dismantling seeks an optimal set of nodes to remove so that a
network is fragmented under **heterogeneous removal costs**. We show that the conventional
metric — the Accumulated Normalized Connectivity (ANC) defined on the absolute Giant
Connected Component (GCC) size — falls into a measurement problem: because it rewards
any reduction in GCC size regardless of how many nodes are removed, a powerful optimizer
learns to harvest large numbers of cheap peripheral nodes (*trivial shrinkage*) instead of
attacking genuinely critical nodes (*structural fragmentation*).

To close this trap we make three contributions:

1. **RGCC** — a corrected measure, the *Relative Giant Connected Component*, that replaces
   the static denominator `N` with the dynamic number of remaining nodes `N − k`. RGCC
   explicitly penalizes excessive removal and distinguishes fragmentation from shrinkage.
2. **REND** — a deep reinforcement learning agent trained purely on small synthetic graphs
   to maximize the RGCC-based reward (optimized through a stable lower bound `R_lb`).
3. **Boundary-Cut** — an interpretable, training-free heuristic distilled from REND's
   learned policy. It iteratively removes *2-core boundary nodes* (nodes in the 2-core with
   at least one neighbour in the 1-shell) in descending order of a cost-effectiveness ratio,
   and matches the black-box model's performance with zero training.

The framework supports at least two cost scenarios for every network: **degree-cost** (cost proportional to node degree), and **random-cost** (random non-negative weights).

## Repository Structure

```
REND/
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
├── models/                     # Pre-trained REND model (REND.ckpt)
├── paper/                      # Manuscript and framework figure
├── data/                       # <-- NOT in git; available on CodeOcean (see below)
├── results/                    # <-- NOT in git; available on CodeOcean (see below)
├── requirements.txt
└── README.md
```

> **Note.** All code and the C++ sources live under `code/`. Scripts are intended to be run
> from inside `code/`; they reference the data, model and result folders one level up
> (`../data`, `../models`, `../results`).

## Data and Results (CodeOcean)

The `data/` (real-world and synthetic networks with their cost-weight files) and `results/` (per-method dismantling solutions and scores) directories are **not** stored in this git repository. They are available directly within the CodeOcean capsule:

- **CodeOcean Capsule:** [https://codeocean.com/capsule/4458609](https://codeocean.com/capsule/4458609)

The capsule contains the complete project environment, including:
- `data/` — all network datasets and cost-weight files
- `results/` — pre-computed dismantling solutions and evaluation scores for all methods
- `models/` — pre-trained REND model checkpoint

To run the code, simply execute it within the CodeOcean capsule environment, where all data and results are pre-populated and ready to use. No additional download or extraction steps are required.

If you wish to run the code locally, you will need to:
1. Download the `data/` and `results/` directories from the CodeOcean capsule (via the capsule's file browser).
2. Place them in the project root directory (next to `code/`).

Each real network in `data/real/` ships with `uniform`, `degree` and `random` cost-weight files.

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

| Component | Specification                                  |
| --------- | ---------------------------------------------- |
| GPU       | 1 × NVIDIA GeForce RTX 4090 (24 GB)            |
| CPU       | x86-64, 8+ cores                               |
| RAM       | 32+ GB                                         |
| CUDA      | Compatible driver for the installed TensorFlow |

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

If running on CodeOcean, the data and results are already pre-populated. For local execution, first ensure the `data/` and `results/` directories are placed in the project root. All commands below are run from inside the `code/` directory.

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

REND and Boundary-Cut are compared against the following state-of-the-art cost-aware and
structural dismantling baselines.

| Method           | Reference                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| HDA              | Holme, P., Kim, B. J., Yoon, C. N. & Han, S. K. Attack vulnerability of complex networks. *Phys. Rev. E* **65**, 056109 (2002).         |
| CoreHD           | Zdeborová, L., Zhang, P. & Zhou, H.-J. Fast and simple decycling and dismantling of networks. *Sci. Rep.* **6**, 37954 (2016).          |
| MinSum           | Braunstein, A., Dall'Asta, L., Semerjian, G. & Zdeborová, L. Network dismantling. *Proc. Natl Acad. Sci. USA* **113**, 12368–12373 (2016). |
| BPD              | Mugisha, S. & Zhou, H.-J. Identifying optimal targets of network attack by belief propagation. *Phys. Rev. E* **94**, 012305 (2016).    |
| CI               | Morone, F. & Makse, H. A. Influence maximization in complex networks through optimal percolation. *Nature* **524**, 65–68 (2015).       |
| GND              | Ren, X.-L., Gleinig, N., Helbing, D. & Antulov-Fantulin, N. Generalized network dismantling. *Proc. Natl Acad. Sci. USA* **116**, 6554–6559 (2019). |
| GDM              | Grassia, M., De Domenico, M. & Mangioni, G. Machine learning dismantling and early-warning signals of disintegration in complex systems. *Nat. Commun.* **12**, 5190 (2021). |
| FINDER           | Fan, C., Zeng, L., Sun, Y. & Liu, Y.-Y. Finding key players in complex networks through deep reinforcement learning. *Nat. Mach. Intell.* **2**, 317–324 (2020). |
| **REND**         | This work.                                                                                                                             |
| **Boundary-Cut** | This work.                                                                                                                             |

Reference implementations used for the external baselines:

```
https://github.com/zhfkt/ComplexCi                                 (CI)
https://github.com/abraunst/decycler                               (MinSum)
http://power.itp.ac.cn/~zhouhj/codes.html                          (BPD)
https://github.com/hcmidt/corehd                                   (CoreHD)
https://github.com/renxiaolong/Generalized-Network-Dismantling     (GND)
https://github.com/NetworkScienceLab/GDM                           (GDM)
https://github.com/FFrankyy/FINDER                                 (FINDER、HDA)
```

## Citation

If you find this code or paper useful, please cite our work:

```bibtex
@article{fan2026Mitigating,
  title   = {Mitigating the measurement problem for cost-constrained network dismantling with deep reinforcement learning},
  author  = {Fan, Changjun and Qing, Feng and Zeng, Li and Li, Aming and Tan, Suoyi and Lu, Xin and Huang, Jincai and Liu, Zhong},
  year    = {2026}
}
```
