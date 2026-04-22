# 🧠 The Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study Submission

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aaryan200106/Tredence-Assesment/blob/main/self_pruning_nn.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Core Idea & Approach](#-core-idea--approach)
3. [Architecture Overview](#-architecture-overview)
4. [Key Components](#-key-components)
5. [Prerequisites](#-prerequisites)
6. [Installation & Setup](#-installation--setup)
7. [How to Run](#-how-to-run)
   - [Option A: Google Colab (Recommended)](#option-a-google-colab-recommended)
   - [Option B: Run Locally](#option-b-run-locally)
8. [Project Structure](#-project-structure)
9. [Technical Deep Dive](#-technical-deep-dive)
   - [PrunableLinear Layer](#1-prunablelinear-layer)
   - [Sparsity Regularisation Loss](#2-sparsity-regularisation-loss)
   - [Training Loop](#3-training-loop)
10. [Results & Analysis](#-results--analysis)
11. [Output Plots Explained](#-output-plots-explained)
12. [Evaluation Criteria (from JD)](#-evaluation-criteria-from-jd)
13. [Submission Details](#-submission-details)

---

## 🎯 Problem Statement

> *"In the real world, deploying large neural networks is often constrained by memory and computational budgets. A common technique to make models smaller and faster is pruning where you remove less important weights or neurons after training. This challenge takes that idea a step further."*

The task is to build a feed-forward neural network that **learns to prune itself during training** not as a post-training step. The network has a built in mechanism to identify and dynamically remove its weakest connections by associating each weight with a learnable **gate parameter**.

**Dataset:** CIFAR-10 — 60,000 colour images (32×32), 10 classes, 50k train / 10k test.

---

## 💡 Core Idea & Approach

The central insight is this: if we can make the network *pay a cost* for every connection it keeps open, it will learn on its own which connections are worth keeping and which ones can be thrown away.

Here's how it works:

1. Every weight `w[i][j]` in the network is paired with a learnable **gate score** `s[i][j]`.
2. A sigmoid function maps each gate score to a value between 0 and 1: `gate = σ(s)`.
3. The actual weight used in the forward pass is `pruned_weight = w * gate`.
4. When `gate → 0`, the corresponding weight contributes nothing it's effectively **pruned**.
5. An **L1 sparsity penalty** on all gate values is added to the training loss, which continuously pushes gates toward zero unless the classification task forces them to stay open.

The network ends up doing a gradient descent powered tug-of-war: the cross entropy loss wants to keep useful gates open, while the sparsity loss wants to close all of them. The result is a naturally sparse network where only the most important connections survive.

---

## 🏗️ Architecture Overview

```
Input (CIFAR-10 Image)
        │
        │  32 × 32 × 3 = 3072 features (flattened)
        ▼
┌───────────────────┐
│  PrunableLinear   │  3072 → 1024
│  + ReLU + Dropout │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  PrunableLinear   │  1024 → 512
│  + ReLU + Dropout │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  PrunableLinear   │  512 → 256
│  + ReLU + Dropout │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  PrunableLinear   │  256 → 10
│  (output logits)  │
└───────────────────┘
        │
        ▼
   Class Prediction
```

Each `PrunableLinear` block carries **two** sets of learnable parameters: the standard weights and the gate scores. Both are updated by the same Adam optimiser on every training step.

**Total learnable parameters (weight + gate_scores + bias):**
- Layer 1: 3072 × 1024 × 2 + 1024 ≈ **6.29M** params
- Layer 2: 1024 × 512  × 2 + 512  ≈ **1.05M** params
- Layer 3: 512  × 256  × 2 + 256  ≈ **262K** params
- Layer 4: 256  × 10   × 2 + 10   ≈ **5.13K** params

---

## 🔩 Key Components

| Component | Description |
|---|---|
| `PrunableLinear` | Custom `nn.Module` replacing `nn.Linear`. Adds `gate_scores` as a second parameter tensor of the same shape as `weight`. Forward pass applies sigmoid gating before the linear operation. |
| `SelfPruningNet` | 4 layer MLP using `PrunableLinear` layers with ReLU activations and Dropout for standard regularisation. |
| `sparsity_loss()` | Method on `SelfPruningNet` that computes the L1 norm (sum) of all gate values across all prunable layers. Used as the regularisation term during training. |
| `run_experiment()` | Full training + evaluation loop for a given λ value. Tracks loss, accuracy, sparsity, and learning rate per epoch. |
| λ (lambda) sweep | Three separate experiments with `λ ∈ {1e-5, 1e-4, 1e-3}` to show the sparsity-accuracy trade-off. |

---

## 📋 Prerequisites

### Python Version
```
Python 3.10 or higher
```

### Core Libraries

| Library | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Neural network framework, autograd |
| `torchvision` | ≥ 0.15.0 | CIFAR-10 dataset loading, transforms |
| `numpy` | ≥ 1.23.0 | Array operations for gate analysis |
| `matplotlib` | ≥ 3.5.0 | Plotting training curves and gate distributions |

### Hardware
- **GPU strongly recommended** — training all three experiments takes ~15–20 minutes on a free Colab T4 GPU.
- CPU is supported but each experiment will take significantly longer (~45–90 min depending on machine).

---

## ⚙️ Installation & Setup

### Local Installation

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Aaryan200106/Tredence-Assesment.git
cd Tredence-Assesment
```

**Step 2 — Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Linux / macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install torch torchvision numpy matplotlib
```

Or using a requirements file if provided:
```bash
pip install -r requirements.txt
```

**Step 4 — Verify your PyTorch installation and GPU availability**
```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

> **Note:** The CIFAR-10 dataset (~170 MB) is downloaded automatically when you run the notebook for the first time. No manual download is needed. It gets saved to `./data/` in the working directory.

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)

This is the easiest and fastest way to run the full experiment with free GPU access.

**Step 1** — Click the badge at the top of this README:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aaryan200106/Tredence-Assesment/blob/main/self_pruning_nn.ipynb)

**Step 2** — Enable GPU runtime:
```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

**Step 3** — Run all cells:
```
Runtime → Run all    (or press Ctrl + F9)
```

**What happens next:**
- Cell 1 sets up imports and detects your GPU
- Cell 2 builds and gradient-checks `PrunableLinear`
- Cell 3 defines `SelfPruningNet` and prints the model summary
- Cell 4 downloads CIFAR-10 automatically
- Cells 5–6 define train/eval functions and the experiment runner
- Cell 7 runs all three λ experiments (this is the long step — ~15–20 min)
- Cells 8–11 print the results table and generate all plots
- Cell 12 prints the final detailed summary

**Plots are saved automatically** to your Colab session:
```
training_curves.png
gate_distributions.png
layer_sparsity.png
```
You can download these from the Files panel on the left sidebar in Colab.

---

### Option B: Run Locally

If you prefer to run the notebook on your own machine with Jupyter:

**Step 1** — Follow the installation steps above.

**Step 2** — Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

**Step 3** — Open `self_pruning_nn.ipynb` and run all cells top to bottom.

> If you don't have Jupyter installed:
> ```bash
> pip install notebook
> ```

---

## 📁 Project Structure

```
Tredence-Assesment/
│
├── self_pruning_nn.ipynb     # Main notebook — complete implementation
│
├── training_curves.png       # Plot 1: accuracy, sparsity, loss over epochs
├── gate_distributions.png    # Plot 2: gate value histograms for each λ
├── layer_sparsity.png         # Plot 3: per-layer sparsity breakdown
│
└── README.md                 # This file
```

> The `./data/` folder gets created automatically when CIFAR-10 is downloaded.

---

## 🔬 Technical Deep Dive

### 1. PrunableLinear Layer

The standard `nn.Linear` computes:
```
output = input @ weight.T + bias
```

`PrunableLinear` introduces a **gate mechanism** on top of this:

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))
        # gate_scores initialised to 1.0 → sigmoid(1) ≈ 0.73 (gates start mostly open)

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # constrain to [0, 1]
        pruned_weights = self.weight * gates               # element-wise masking
        return F.linear(x, pruned_weights, self.bias)
```

**Why does gradient flow work?**

The key is that `pruned_weights = self.weight * sigmoid(self.gate_scores)` is a fully differentiable operation. During backpropagation:

- Gradient w.r.t. `weight[i,j]` = `upstream_grad * gate[i,j]`
- Gradient w.r.t. `gate_scores[i,j]` = `upstream_grad * weight[i,j] * σ'(gate_scores[i,j])`

Where `σ'` is the sigmoid derivative. Both parameters receive gradients and are updated by Adam. This is verified in Cell 2 with an explicit assertion check.

**Gate score initialisation choice:**
Gates are initialised to `1.0` (not `0.0`). This means `sigmoid(1) ≈ 0.73`, so gates start mostly open. This is intentional — it gives the network a chance to learn useful representations first before the sparsity penalty starts forcing closures.

---

### 2. Sparsity Regularisation Loss

The total loss at each training step is:

```
L_total = L_CrossEntropy + λ × L_sparsity
```

Where:

```
L_sparsity = Σ (over all PrunableLinear layers) Σ (over all i, j) σ(gate_scores[i,j])
```

Since `σ(x) ∈ [0, 1]` and is always positive, the L1 norm of the gate values equals their sum. This simplifies computation and interpretation: `L_sparsity` is literally "the total amount of gate activation remaining in the network."

**Why L1 and not L2?**

This is a subtle but important distinction:

| Regularisation | Gradient near zero | Effect |
|---|---|---|
| L2 (`Σ g²`) | `2g → 0` as `g → 0` | Shrinks values but rarely reaches exactly zero |
| L1 (`Σ \|g\|`) | Constant `±1` | Provides a uniform "pull" toward zero, leading to exact sparsity |

The constant gradient of the L1 norm means even a gate value of `0.001` still gets the same push toward zero as a gate value of `0.5`. This is precisely why L1 is the standard choice for inducing sparsity in machine learning from Lasso regression all the way to modern neural network pruning.

**Backprop through the sparsity loss:**
```
d(L_sparsity) / d(gate_scores[i,j]) = λ × σ(gate_scores[i,j]) × (1 − σ(gate_scores[i,j]))
```
This gradient always pushes `gate_scores[i,j]` toward `−∞`, which drives `σ(gate_scores[i,j]) → 0`. The classification loss provides the counter force: if a weight is useful, its contribution to reducing cross-entropy will keep its gate open.

---

### 3. Training Loop

```
For each epoch:
  For each mini-batch:
    1. Forward pass through SelfPruningNet
    2. Compute cross-entropy loss on predictions vs labels
    3. Compute sparsity loss (sum of all sigmoid gate values)
    4. total_loss = ce_loss + λ × sparsity_loss
    5. total_loss.backward()   ← gradients flow to both weight and gate_scores
    6. optimizer.step()        ← Adam updates all parameters
  
  After epoch:
    Evaluate test accuracy (CE loss only, no sparsity term)
    Compute sparsity level (% of gates below threshold 0.01)
    Step cosine annealing scheduler
```

**Optimiser:** Adam with `lr=1e-3`, `weight_decay=1e-4`

**Scheduler:** Cosine Annealing (`T_max = num_epochs`) — smoothly decays learning rate to near-zero by the final epoch, which helps the network settle into a cleaner sparse solution.

**Epochs:** 25 per experiment (balances training time and convergence on Colab free tier)

**Batch size:** 128 for training, 256 for evaluation

**Data augmentation (training only):**
- Random crop with padding=4
- Random horizontal flip
- Normalisation using CIFAR-10 per-channel statistics: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)

---

## 📊 Results & Analysis

Three separate models are trained, one for each λ value:

| λ (Lambda) | Regime | Expected Test Accuracy | Expected Sparsity |
|---|---|---|---|
| `1e-5` | Low (weak pruning) | ~52–55% | ~10–25% |
| `1e-4` | Medium (balanced) | ~48–52% | ~50–70% |
| `1e-3` | High (aggressive) | ~40–47% | ~80–95% |

> *Exact numbers depend on the GPU and random seed. Run the notebook to reproduce.*

**Key observations:**

- As λ increases, sparsity increases and accuracy decreases this is the fundamental sparsity-accuracy trade-off.
- The **medium λ = 1e-4** model typically represents the best practical trade-off: meaningful compression while retaining reasonable classification performance.
- Even the high-λ model demonstrates that the self-pruning mechanism works correctly the network successfully identifies and removes a large fraction of its own weights.
- Per layer analysis (Plot 3) usually shows that **earlier layers (wider layers)** are pruned more aggressively than later, narrower layers. This makes intuitive sense wide early layers have more redundant capacity.

**Why does L1 on sigmoid gates create a bimodal distribution?**

After training with a non trivial λ, plotting gate values produces a characteristic bimodal histogram:
- A large spike near `0.0` — connections the network decided to prune
- A smaller cluster near `0.7–1.0` — connections the network decided to keep

This bimodality is exactly what a successful pruning mechanism should produce. It indicates that gates have reached a near binary decision state: either a connection is useful (gate ≈ 1) or it isn't (gate ≈ 0). A uniform or unimodal distribution would suggest the regularisation wasn't strong enough to drive any gates fully to zero.

---

## 📈 Output Plots Explained

### `training_curves.png`
Three panels showing epoch by epoch progression for all three λ values:
- **Left:** Test accuracy over epochs — you should see λ = 1e-5 consistently on top
- **Middle:** Sparsity level over epochs — watch how λ = 1e-3 aggressively prunes in early epochs
- **Right:** Total training loss — includes both CE and sparsity components, so higher-λ models show higher raw loss values even at convergence

### `gate_distributions.png`
The most diagnostic plot. A histogram of all gate values after training, one subplot per λ:
- **Successful result:** Large spike near 0, separate cluster away from 0 (bimodal)
- **What it means:** The network has made clean binary decisions about most of its connections
- Annotations show the exact count of gates classified as "pruned" (gate < 0.01)

### `layer_sparsity.png`
Bar charts showing per-layer sparsity broken down by layer dimensions:
- `3072→1024`, `1024→512`, `512→256`, `256→10`
- Useful for understanding *where* in the network pruning is most concentrated
- Typically shows heavier pruning in the first (widest) layer

---

## ✅ Evaluation Criteria (from JD)

The case study is evaluated on four criteria. Here's how this implementation addresses each:

| Criteria | How it's addressed |
|---|---|
| **Correctness of PrunableLinear** | `gate_scores` is an `nn.Parameter` of the same shape as `weight`. Forward pass: `sigmoid(gate_scores) * weight`. Explicit gradient assertions in Cell 2 confirm both `weight.grad` and `gate_scores.grad` are non-None after a backward pass. |
| **Training loop implementation** | `train_one_epoch()` computes `total_loss = ce_loss + λ * sparsity_loss` every step. `model.sparsity_loss()` iterates over all `PrunableLinear` layers and sums all sigmoid gate values. Standard Adam optimiser updates everything including gate scores. |
| **Quality of results and analysis** | Three λ values tested, results table printed, all three diagnostic plots generated. Report section in the notebook explains *why* L1 on sigmoid gates works, analyses the bimodal distribution, and discusses the trade-off. |
| **Code quality** | Every function has a docstring explaining its purpose, inputs, and outputs. Variable names are descriptive. Comments explain non-obvious decisions (e.g., why gate_scores are initialised to 1.0, why L1 vs L2). Helper methods (`prunable_layers()`, `all_gate_values()`, `sparsity_percent()`) separate concerns cleanly. |

---

## 📬 Submission Details

**Submitted by:** Aryan
**Internship:** Tredence Studio — AI Agents Engineering Team, 2025 Cohort
**Role:** AI Engineering Intern
**Location:** Bengaluru (Hybrid)
**Duration:** 6 months – 1 year

**Repository contents:**
- `self_pruning_nn.ipynb` — complete implementation and inline report
- `training_curves.png` — training dynamics visualisation
- `gate_distributions.png` — gate value distribution histogram
- `layer_sparsity.png` — per-layer sparsity breakdown
- `README.md` — this file

---

*This case study was completed as part of the Tredence AI Engineering Internship application. The problem involved implementing a self-pruning neural network with learnable gate parameters, L1 sparsity regularisation, and a comparative analysis across multiple regularisation strengths.*
