# Multi-Disease Brain Screening via Connectome Graph Neural Networks (NeuroScreen)

An AI-driven Clinical Decision Support System (CDSS) for automated screening of brain conditions using Graph Neural Networks (GNN). The system converts brain scans into anatomical brain graphs and evaluates probability scores for **Healthy**, **Alzheimer's Disease (AD)**, **Parkinson's Disease (PD)**, and **Autism Spectrum Disorder (ASD)** while showing an interactive 3D visual explanation.

---

## 📌 Project Overview

Traditional CNNs process 3D brain MRI as heavy voxel grids, which consumes significant GPU memory and ignores topological brain connectivity. **NeuroScreen** solves this by:

1. Dividing brain scans into **116 anatomical regions** using the standard **AAL-116 atlas**.
2. Modeling regions as **nodes** and inter-regional connectivity as **edges**.
3. Processing the graph using a lightweight **2-layer Graph Convolutional Network (NeuroGNN)**.
4. Outputting independent probabilities using a **multi-label Sigmoid head** (allowing multiple conditions without artificial Softmax suppression).
5. Displaying an interactive **3D glass-brain visualization** where damaged or high-salience areas (such as the Putamen/Basal Ganglia in PD) appear as glowing red nodes.

---

---

## 🖥️ System Walkthrough

| Step | Feature | Description |
| :--- | :--- | :--- |
| **1** | **Patient Intake & Upload** | Enter patient info and upload a de-identified NIfTI (`.nii`/`.nii.gz`) volume or 2D MRI slice. |
| **2** | **Instant Localizer Preview** | The backend extracts and displays the central axial slice to verify orientation. |
| **3** | **GNN Screening Results** | Displays independent risk scores for Healthy, AD, PD, and ASD. |
| **4** | **3D Connectome Explainability** | Renders an interactive 3D WebGL brain showing affected regions (e.g., Node 74 in PD). |
| **5** | **Clinical Summary Report** | Auto-generates a printable report sheet for hospital records. |

---

## 🏗️ Architecture & Model Pipeline

```
[Uploaded Brain Scan (NIfTI / MRI)]
                │
                ▼
  [Standardize to MNI-152 Space]
                │
                ▼
[Extract 116 Regions via AAL Atlas]
                │
                ▼
   [Graph: 116 Nodes × 64 Features]
   [Edges: Functional / Spatial Adjacency]
                │
                ▼
  [NeuroGNN (2-Layer Graph Convolution)]
   - GCNConv(64 → 64) + BatchNorm(116) + LeakyReLU
   - Dynamic DropEdge (p=0.15) for regularization
   - GCNConv(64 → 32) + BatchNorm(116) + LeakyReLU
                │
                ▼
    [Global Pooling (Mean + Max)]
                │
                ▼
  [Multi-Label Sigmoid Output Head]
  ┌──────────┬──────────┬──────────┐
  ▼          ▼          ▼          ▼
Healthy     ASD         AD         PD
  (%)       (%)        (%)        (%)
```

---

## 📊 Key Results

Tested on a held-out clinical test cohort of 55 scans from **ABIDE**, **OASIS-1**, and **PPMI**:

* **Bitwise Classification Accuracy:** 98.64%
* **Alzheimer's Sensitivity:** 100.00%
* **Parkinson's Sensitivity:** 90.62%
* **Primary Detected Biomarkers:** Correctly highlighted **Node 74 (Putamen / Basal Ganglia circuit)** and Substantia Nigra at 92.4% salience in Parkinson's test scans.

---

## 📁 Repository Structure

```
├── app.py                           # Flask backend server & inference endpoints
├── train_multidisease_production.py # NeuroGNN architecture & training logic
├── checkpoints/
│   └── best_multidisease_model.pt   # Pretrained 116-node model checkpoint
├── docs/
│   └── screenshots/                 # UI screenshots and visual assets
│       ├── 01_upload_and_preview.png
│       ├── 02_diagnostic_output.png
│       ├── 03_3d_glass_brain_salience.png
│       └── 04_clinical_report.png
├── static/
│   ├── app.js                       # 3D Three.js brain visualizer & frontend logic
│   ├── style.css                    # UI styles (Dark theme)
│   └── models/
│       └── brain.obj                # 3D brain anatomical surface mesh
├── templates/
│   └── index.html                   # Web interface dashboard
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🚀 Quick Setup Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/NeuroScreen-CDSS.git
cd NeuroScreen-CDSS

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1
# Activate (Linux / Mac)
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add Screenshot Assets (Optional)

Create the directory and place your application screenshots if you are contributing:
```bash
mkdir -p docs/screenshots
```

### 4. Run the App

Make sure `checkpoints/best_multidisease_model.pt` exists, then:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 👥 Project Information

* **Institution:** SCMS School of Engineering and Technology (SSET), Cochin
* **Department:** Artificial Intelligence & Data Science
* **Students:**
  * Ajana Biju (SCM23CD007)
  * Alan Parappil Agi (SCM23CD011)
  * Amal N V (SCM23CD013)
  * Devika A S (SCM23CD025)
* **Project Guide:** Dr. Sindhya K. Nambiar (Associate Professor & HOD, Dept. of AI & DS)
* **Project Coordinator:** Ms. Shali Sara Abraham (Assistant Professor, Dept. of AI & DS)

---
