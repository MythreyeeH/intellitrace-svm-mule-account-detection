# Cross-Channel Mule Account Detection with Heterogeneous GNNs

## Problem Statement

Money mules operate across channels to evade detection. Funds may be received via UPI or mobile applications, routed through linked wallets, and withdrawn via ATM cash-out within minutes. Traditional siloed fraud rules often fail to detect high-velocity, multi-hop, cross-channel transaction patterns.

**Proposed Solution:**  
A Heterogeneous Graph Neural Network (GNN) that unifies App, Web, ATM, and UPI transaction logs into a single entity graph and detects mule rings in near real-time.

---

# Methodology

## Phase 1: Unified Heterogeneous Entity Graph

We construct a multi-relational heterogeneous entity graph from aggregated multi-channel transaction logs.

### Node Types
- Account  
- Device  
- IP Address  
- Merchant  

### Edge Types
- `transfers_to` (Account → Account), fraud-labelled  
- `uses_device` (Account → Device)  
- `uses_ip` (Account → IP)  
- `pays_merchant` (Account → Merchant)  

### Feature Engineering
- Velocity scores  
- Behavioral anomaly metrics  
- Device/IP sharing flags  
- Pass-through ratios  
- Transaction amount and temporal gaps  

This unified graph captures structural and cross-channel relationships beyond traditional tabular models.

---

## Phase 2: CAV-HGNN (Camouflage- and Velocity-Aware Heterogeneous GNN)

CAV-HGNN extends a heterogeneous GraphSAGE baseline with mule-aware enhancements.

### 1. Heterogeneous Message Passing
Relation-specific SAGEConv or R-GCN layers aggregate signals across edge types.

### 2. Risk-Aware Neighbor Sampling
Top-k high-risk neighbors are selected per relation to prevent camouflage dilution.

### 3. Temporal Decay Weighting

w_e = exp(-Δt / τ)

Edge messages are weighted using time decay to prioritize recent high-velocity activity.

### 4. Edge-to-Account Risk Aggregation

risk(a) = Σ w_e * p_e

Transaction-level fraud probabilities are aggregated into account-level risk scores.  
Accounts exceeding a defined threshold are flagged as mule candidates.

---

# Novel Contributions

1. Cross-channel entity unification into a single heterogeneous graph.
2. CAV-HGNN pipeline combining heterogeneous GNN learning, camouflage resistance, and temporal awareness.
3. Production-oriented scoring framework: transaction-level fraud → account-level risk → real-time blocking.

---

# Technology Stack

- PyTorch Geometric
- PyTorch
- scikit-learn
- NetworkX
- Pandas
- HuggingFace Datasets

---

# Quick Start

```bash
# Clone repository
git clone <your-repo>
cd mule-detection-gnn

# Install dependencies
pip install torch torch-geometric pandas scikit-learn networkx matplotlib tqdm datasets

# Login to HuggingFace (required for dataset access)
huggingface-cli login

# Launch notebook
jupyter notebook notebooks/Intellitrace_SVM_Prototype.ipynb
```
---

# Folder Structure
```text
root/
│
├── README.md
│
├── data/
│   └── (optional processed files)
│
└── notebooks/
    └── Intellitrace_SVM_Prototype.ipynb
```

---

# Future Work
- Real-time streaming with incremental graph updates and online inference
- Temporal GNN models (TGN, DyRep) for dynamic transaction modeling
- Risk diffusion through graph propagation
- Multi-bank KYC stitching across institutions
- Adaptive thresholding based on operational risk appetite

---

# References
- CARE-GNN: Camouflage-Resistant Fraud Detection
- Heterogeneous Graph Neural Networks for AML
- Dynamic GNNs for Financial Anomaly Detection
- Nigerian Financial Transactions Dataset (HuggingFace)
