# TrustGraph - Trustworthy Graph Neural Networks for Financial Fraud Detection

A full-stack system combining Graph Neural Networks with advanced uncertainty quantification for financial fraud detection and compliance automation.

## Architecture

- **Backend**: FastAPI + PyTorch Geometric, deployed on HuggingFace Spaces
- **Frontend**: Next.js 16 + Tailwind CSS + Cytoscape.js, deployed on Vercel
- **Models**: GraphSAGE, GAT, GCN trained on Kaggle GPU, hosted on HuggingFace Hub
- **Auth**: Firebase Authentication + Firestore
- **LLM Agent**: Groq API for compliance report generation

## Uncertainty Quantification Methods

1. **Conformal Prediction (CF-GNN)**: Distribution-free coverage guarantees with Adaptive Prediction Sets
2. **Evidential Deep Learning (EDL)**: Dirichlet-based uncertainty via subjective logic decomposition
3. **Monte Carlo Dropout**: Bayesian approximation with epistemic/aleatoric decomposition

## Datasets

- **Elliptic Bitcoin**: 203K nodes, 234K edges - cryptocurrency transaction fraud detection
- **DGraph-Fin**: 3.7M nodes - consumer credit social network
- **Amazon-Fraud**: 12K nodes - e-commerce review network

## Graph Topology Engineering

Four alternative topologies to mitigate structural noise:
- Feature k-NN Graph (k=5, L2 distance)
- Cosine Similarity Graph (threshold > 0.92)
- Temporal SNAP Graph (cross-timestep k=3 NN)
- Augmented Graph (original + similarity edges)

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Deployment

- **Backend**: Auto-syncs to HuggingFace Space via GitHub Actions
- **Frontend**: Deploys to Vercel on push
- **Models**: Hosted on HuggingFace Hub (Arko007/trustworthy-gnn-fraud-models)
