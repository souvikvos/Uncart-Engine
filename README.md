<div align="center">

# 📦 UNCart — E-Commerce Return Intelligence Platform
### *Pre-Dispatch Return Prediction, Explainable AI (XAI) & Grounded Policy RAG*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML__29-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/XAI-Linear%20SHAP-blueviolet.svg)](https://github.com/slundberg/shap)
[![Deployment](https://img.shields.io/badge/Render-Live%20Inference-brightgreen.svg?logo=render&logoColor=white)](https://uncart-engine.onrender.com/docs)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

**Live AI Microservice:** [https://uncart-engine.onrender.com/docs](https://uncart-engine.onrender.com/docs)  
**Academic Requirement:** Project ID: `ML_29` — Customer Product Return Prediction (Mandatory Algorithm: **Logistic Regression**)

</div>

---

## 📌 Executive Summary

Product returns (Return-to-Origin / RTO) cost Indian and global e-commerce merchants over **₹250 – ₹350 per returned shipment** in reverse courier logistics, damaged packaging, and locked inventory. 

**UNCart** is a decision intelligence platform designed for warehouse managers and online merchants. It flags high-risk orders **before packages leave the warehouse**, breaks down the exact mathematical drivers using **Explainable AI (SHAP)**, and synthesizes policy-backed mitigation steps using a **Grounded RAG Engine** over store operational SOPs.

---

## 🏛️ System Architecture

UNCart is built on a modern **Decoupled Microservice Architecture**:

```
[ Srayan: React + Tailwind Frontend ]
                │
                │  HTTP / REST
                ▼
[ Soyam: Express.js API Gateway + MongoDB ]
                │
                │  Bulk CSV & Live Ingestion
                ▼
[ Souvik: Python FastAPI Intelligence Microservice (Deployed on Render) ]
    ├── 1. Preprocessing: ColumnTransformer (StandardScaler + OneHotEncoder)
    ├── 2. Primary ML Pipeline: Logistic Regression (Project ID: ML_29)
    ├── 3. Benchmark ML Pipeline: Random Forest Classifier
    ├── 4. Explainable AI: Real-time Game-Theoretic SHAP Linear Attribution
    └── 5. Grounded RAG Agent: Policy Chunk Retrieval + LLM Synthesis
```

---

## 🚀 Key Features

* **Dual-Model Inference Engine:**
  * **Primary (ML_29):** Calibrated **Logistic Regression** trained for ultra-fast, sub-10ms inference and exact mathematical interpretability.
  * **Benchmark:** **Random Forest Classifier** used as an academic performance baseline.
* **Explainable AI (SHAP):** Decomposes every return prediction into localized Shapley values, calculating exact additive feature impacts (e.g. `+40.0% Cash on Delivery`, `+20.0% Item Quantity`, `-15.0% Account Age`).
* **Grounded Policy RAG:** Retrieves specific clauses from the store's Standard Operating Procedure (`store_policy.txt`) to synthesize non-hallucinated, actionable warehouse mitigation steps (e.g., OTP verification, size checks).
* **Batch Vectorized Scoring:** Microservice endpoint optimized to score up to 1,000 orders simultaneously with sub-second response times.

---

## 📊 Model Evaluation & Benchmarks

Trained on **6,000 statistically calibrated e-commerce transactions** reflecting real-world retail return distributions (COD doorstep rejections, apparel size-trialing):

| Metric | Logistic Regression (ML_29) *(Primary)* | Random Forest *(Benchmark)* | Industry Baseline |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **80.86%** | 79.80% | ~65.0% |
| **Accuracy** | **73.17%** | 72.92% | ~58.0% |
| **Precision** | **75.15%** | 73.04% | — |
| **Recall** | **76.52%** | **80.45%** | — |
| **F1-Score** | **75.83%** | 76.57% | — |
| **Inference Latency** | **< 3 ms** | ~12 ms | — |
| **Mathematical Explainability** | **Exact (Linear SHAP)** | Approximation | None |

> **Academic Note:** While Random Forest achieved slightly higher recall, **Logistic Regression** was chosen for the primary production microservice due to its sub-3ms latency at checkout, lower computational footprint, and exact linear interpretability.

---

## 🔌 API Reference & Live Endpoints

Base URL: `https://uncart-engine.onrender.com` (Interactive Swagger: `/docs`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Heartbeat probe checking memory models & RAG policy index. |
| `GET` | `/metrics` | Academic benchmark scores (Accuracy, ROC-AUC, Confusion Matrix). |
| `POST` | `/predict` | Single-order live scoring with real-time SHAP feature attribution. |
| `POST` | `/predict/batch` | Bulk scoring endpoint for CSV uploads (attaches SHAP drivers per row). |
| `POST` | `/investigate` | Semantic RAG retrieval over `store_policy.txt` returning mitigation advice. |

---

## 📁 Repository Structure

```text
├── docs/
│   ├── UNCART_MASTER_PLAN.md      # Master design document & team matrix
│   └── architecture_diagram.html  # Standalone interactive architecture diagram
├── ml/
│   ├── engine.py                  # Production FastAPI serving microservice
│   ├── train.py                   # Model training, calibration & SHAP export
│   ├── generate_dataset.py        # 6,000-order retail dataset generator
│   ├── export_charts.py           # Publication-ready matplotlib charts exporter
│   ├── store_policy.txt           # Chunked SOP for Grounded RAG retrieval
│   ├── requirements.txt           # Python production dependencies
│   ├── uncart_pipeline.joblib     # Exported Logistic Regression pipeline (ML_29)
│   ├── uncart_rf_pipeline.joblib  # Exported Random Forest pipeline
│   ├── model_metrics.json         # Evaluation benchmarks JSON
│   ├── feature_metadata.json      # Preprocessing column schema
│   └── shap_summary.json          # Global feature attribution weights
├── .gitignore
└── README.md
```

---

## 🛠️ Local Development & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/souvikvos/Uncart-Engine.git
cd Uncart-Engine
```

### 2. Set up Python virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r ml/requirements.txt
```

### 3. Run the AI Microservice
```bash
python ml/engine.py
```
Server will start locally at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` for the interactive API explorer.

---

## 👥 Engineering Team

| Member | Domain | Key Responsibilities |
| :--- | :--- | :--- |
| **Souvik Ghosh** *(Lead)* | **AI / ML & Inference** | Logistic Regression modeling (ML_29), SHAP linear attribution, FastAPI engine, Grounded RAG pipeline & Cloud deployment. |
| **Soyam Bhalotia** | **Backend & Systems** | Express.js API Gateway, MongoDB persistence, CSV upload parsing & service orchestration. |
| **Srayan Acharjee** | **Frontend & Visual Analytics** | React (Vite) + Tailwind CSS dashboard (Orders Radar, Live Simulator, SHAP Inspector & Analytics). |
| **Sourav Patra** | **QA, Testing & Docs** | Test dataset curation (`normal`, `high_risk`, `edge_cases`), edge testing & college presentation. |

---

<div align="center">
<sub>Built with academic rigor for Project ID: <b>ML_29 — Customer Product Return Prediction</b></sub>
</div>
