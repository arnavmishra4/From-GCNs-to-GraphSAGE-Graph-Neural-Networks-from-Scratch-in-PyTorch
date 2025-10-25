# 🧠 From GCNs to GraphSAGE — Graph Neural Networks from Scratch in PyTorch

This repository is part of my **PyTorch Research Mastery Series**, where I implement deep learning architectures **from scratch**, without relying on high-level abstractions.  

This particular project focuses on **Graph Neural Networks (GNNs)** — starting with a hand-built **Graph Convolutional Network (GCN)**, implemented purely using PyTorch tensor operations and adjacency normalization math.  
The repository is fully modular and structured to later include **Graph Attention Networks (GAT)** and **GraphSAGE**, extending this framework into attention-based and inductive GNN paradigms.

---

## 🧱 Research Intent

This repository is part of my ongoing **PyTorch Deep Research Grind** — a series of self-driven, low-level implementations aimed at mastering AI architectures from first principles.  
Rather than using convenience wrappers, each project reconstructs model internals using **raw tensor operations, autograd, and linear algebra**.

> **Objective:** Build an unshakable foundation in the theory and implementation of GNNs before transitioning to more advanced forms (GAT, GraphSAGE, and Transformers on Graphs).

---

## 🧩 Overview

Graph Neural Networks (GNNs) are a fundamental class of deep learning models designed to operate on **non-Euclidean data** — graphs, networks, molecules, and relational systems.  

This repo demonstrates:
- **A GCN implemented completely from scratch** (no `torch_geometric.nn`).
- Flexible **dataset support** — any graph dataset (Cora, PubMed, custom, etc.).
- Modular design for seamless expansion into **GAT** and **GraphSAGE**.
- Reproducible, research-ready codebase structure following FAIR-style conventions.

---

## ⚙️ Key Features

| Component | File | Description |
|------------|------|-------------|
| 🧱 `GCNFromScratch` | `models/gcn.py` | Pure PyTorch GCN using manual adjacency normalization and graph convolution |
| ⚙️ `data_loader.py` | | Loads built-in or user-defined graph datasets |
| 🔧 `config.py` | | Centralized configuration for hyperparameters |
| 🧩 `gat.py` | | Placeholder for Graph Attention Network (upcoming extension) |
| 🧩 `graphsage.py` | | Placeholder for GraphSAGE inductive learning (upcoming extension) |

---

## 🧱 Architecture

The **Graph Convolutional Network (GCN)** implemented here follows the mathematical formulation introduced by Kipf & Welling (2016):

```
H^(l+1) = σ(D̂^(-1/2) Â D̂^(-1/2) H^(l) W^(l))
```

where:
- `Â = A + I`: adjacency matrix with self-loops  
- `D̂`: degree matrix of `Â`  
- `W^(l)`: learnable weight matrix  
- `σ`: non-linearity (ReLU)  

The model stacks two such layers, with dropout and softmax normalization for classification.

---

## 🧮 Implementation Details

The implementation explicitly constructs the adjacency matrix, normalizes it using `D^(-1/2) A D^(-1/2)`, and performs propagation via matrix multiplications.  
This exercise was done intentionally to internalize the underlying linear algebra and propagation mechanisms behind GCNs — instead of relying on prebuilt PyG modules.

---

## 📦 Dataset Support

This framework supports **any dataset** compatible with `torch_geometric.data.Data`.  

### ✅ Built-in Datasets (PyG)
- Cora  
- Citeseer  
- PubMed  

### ✅ Custom Graphs
You can create your own dataset directly:
```python
from torch_geometric.data import Data
from data_loader import create_custom_graph
import torch

x = torch.randn(10, 4)  # 10 nodes, 4 features each
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # edges (source -> target)
y = torch.randint(0, 2, (10,))  # binary node labels

data = create_custom_graph(x, edge_index, y=y)
```

---

## 🧠 Training

Run the training script with any dataset you specify in `config.py`:

```bash
python main.py
```

You can adjust the dataset and hyperparameters in `config.py`:

```python
class Config:
    dataset_name = "Cora"   # or "PubMed", "Citeseer", or a custom graph
    hidden_channels = 16
    learning_rate = 0.01
    weight_decay = 5e-4
    num_epochs = 200
```

---

## 📊 Output Example

During training:

```
Epoch 010 | Loss: 0.9821 | Train: 0.8234 | Val: 0.8070 | Test: 0.7961
Epoch 020 | Loss: 0.7195 | Train: 0.8723 | Val: 0.8341 | Test: 0.8117
...
```

---

## 🧭 Roadmap

| Stage | Model | Status |
|-------|-------|--------|
| 1️⃣ | **GCN (from scratch)** | ✅ Completed |
| 2️⃣ | **GAT (Graph Attention Network)** | ⏳ In progress — to be integrated next |
| 3️⃣ | **GraphSAGE (Inductive Aggregation)** | 🔜 Planned |
| 4️⃣ | **Transformer-based Graph Networks** | 🚧 Future research goal |

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
torch-sparse
torch-scatter
numpy
```

---

## 📜 Citation

If you use or reference this work:

```
Arnav Mishra, "From GCNs to GraphSAGE — Graph Neural Networks from Scratch in PyTorch", 2025.
```

---

## 🏁 Author

**Arnav Mishra**  
AI Researcher · PyTorch Core & Graph Neural Networks Enthusiast  
Bhopal, India

---

> *"Understanding GNNs begins where you stop importing prebuilt layers."*
