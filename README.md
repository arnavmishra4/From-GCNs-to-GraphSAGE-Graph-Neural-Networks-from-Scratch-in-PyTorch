# 🧠 Graph Neural Networks from Scratch in PyTorch

This repository is part of my **PyTorch Research Mastery Series** — implementing deep learning architectures from scratch using pure PyTorch.

Currently features a **Graph Convolutional Network (GCN)** built from tensor operations and adjacency normalization, with **GAT** and **GraphSAGE** planned as extensions.

---

## 🧱 Research Intent

Part of my **PyTorch Deep Research Grind** — mastering AI architectures from first principles using raw tensor operations, autograd, and linear algebra instead of high-level wrappers.

> **Goal:** Build a solid foundation in GNN theory and implementation before advancing to GAT, GraphSAGE, and graph transformers.

---

## 🧩 Overview

Graph Neural Networks operate on non-Euclidean data — graphs, networks, molecules, and relational systems.

This repo includes:
- **GCN implemented from scratch** (no `torch_geometric.nn`)
- Support for standard datasets (Cora, PubMed, Citeseer) and custom graphs
- Modular structure for future GAT and GraphSAGE integration
- Research-grade codebase following FAIR-style conventions

---

## ⚙️ Components

| Component | File | Description |
|------------|------|-------------|
| 🧱 `GCNFromScratch` | `models/gcn.py` | Pure PyTorch GCN with manual adjacency normalization |
| ⚙️ `data_loader.py` | | Dataset loading utilities |
| 🔧 `config.py` | | Hyperparameter configuration |
| 🧩 `gat.py` | | Graph Attention Network (upcoming) |
| 🧩 `graphsage.py` | | GraphSAGE (upcoming) |

---

## 🧱 Architecture

GCN implementation follows Kipf & Welling (2016):

```
H^(l+1) = σ(D̂^(-1/2) Â D̂^(-1/2) H^(l) W^(l))
```

where:
- `Â = A + I`: adjacency matrix with self-loops  
- `D̂`: degree matrix
- `W^(l)`: learnable weights
- `σ`: ReLU activation

Implementation explicitly constructs and normalizes the adjacency matrix using `D^(-1/2) A D^(-1/2)` to understand the underlying propagation mechanics.

---

## 🧠 Training

```bash
python main.py
```

Configure in `config.py`:

```python
class Config:
    dataset_name = "Cora"
    hidden_channels = 16
    learning_rate = 0.01
    num_epochs = 200
```

---

## 🧭 Roadmap

| Model | Status |
|-------|--------|
| **GCN (from scratch)** | ✅ Completed |
| **GAT** | ⏳ In progress |
| **GraphSAGE** | 🔜 Planned |
| **Graph Transformers** | 🚧 Future |

---

## 🧩 Folder Structure

```
gcn-from-scratch/
│
├── main.py
├── config.py
├── data_loader.py
│
├── models/
│   ├── gcn.py
│   ├── gat.py
│   └── graphsage.py
│
└── README.md
```

---

## 🧰 Requirements

```
torch
torch-geometric
numpy
```

---

## 🏁 Author

**Arnav Mishra**  
AI Researcher · PyTorch & Graph Neural Networks  
Bhopal, India

---

> *"Understanding GNNs begins where you stop importing prebuilt layers."*
