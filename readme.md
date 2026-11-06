# NeuroScreen CDSS: Multi-Disease Brain Screening via Connectome Graph Neural Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Atlas: AAL--116](https://img.shields.io/badge/Brain%20Atlas-AAL--116-orange.svg)](https://www.gin.cnrs.fr/en/tools/aal/)

NeuroScreen is an explainable Clinical Decision Support System (CDSS) for automated, concurrent screening of neurodegenerative and neurodevelopmental brain disorders. Bypassing rigid, computationally prohibitive 3D convolutional voxel grids, NeuroScreen models multi-modal volumetric brain scans as 116-node anatomical connectomes using the Automated Anatomical Labeling (AAL-116) atlas.

The underlying **NeuroGNN** architecture evaluates whole-brain circuit disruptions and outputs independent probabilities across **Healthy Control (HC)**, **Autism Spectrum Disorder (ASD)**, **Alzheimer's Disease (AD)**, and **Parkinson's Disease (PD)** without cross-disease suppression, while rendering an interactive 3D WebGL glass-brain salience map to explain predictions.

---

## Key Features

- **Standardized AAL-116 Connectome Modeling:** Parcellates volumetric neuroimaging inputs into 116 anatomical Regions of Interest (ROIs), shrinking input complexity from $\sim 10^7$ raw voxels down to a compact graph $(\mathcal{V}, \mathcal{E}, X, A)$.
- **Multi-Modal Compatibility:** Operates on resting-state fMRI (Pearson correlation connectomes), structural T1 MRI, and DaTscans (stereotaxic 3D Euclidean distance decay priors).
- **Comorbidity-Ready Multi-Label Architecture:** Employs an independent multi-label Sigmoid classification head instead of a zero-sum Softmax layer, enabling detection of co-occurring neuropathologies.
- **Topological DropEdge & Feature Jittering:** Prevents over-smoothing and protects against scanner-specific domain shifts across multi-site data.
- **Explainable 3D WebGL Visualizer:** Projects interactive glass-brain connectomes in the browser, highlighting salient defect regions (e.g., Putamen / Substantia Nigra Node 74 in Parkinson's cases).
- **Automated Clinical Reporting:** Compiles standardized diagnostic reports with clinician credential verification, multi-channel risk breakdowns, and primary anatomical driver summaries.

---

## System Architecture

```text
       Raw Multi-Modal Scan (fMRI / sMRI / DaTscan)
                           │
                           ▼
        Spatial Normalization & MNI-152 Registration
                           │
                           ▼
          AAL-116 Atlas Anatomical Parcellation
                           │
                           ▼
            Brain Graph Modeling: G = (V, E, X, A)
             ├─ Node Features: X ∈ ℝ^(116 × 64)
             └─ Adjacency Matrix: A ∈ ℝ^(116 × 116)
                           │
                           ▼
        NeuroGNN Engine (2-Layer Graph Convolution)
             ├─ Layer 1: GCNConv(64 → 64) + BatchNorm1d(116) + LeakyReLU
             ├─ Dynamic DropEdge (p = 0.15) & Feature Jittering (σ = 0.01)
             └─ Layer 2: GCNConv(64 → 32) + BatchNorm1d(116) + LeakyReLU
                           │
                           ▼
      Global Pooling (Concat Mean + Max Readout) ──> h_G ∈ ℝ^64
                           │
                           ▼
          Independent Multi-Label Sigmoid Head
       ┌──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
   P(Healthy)       P(ASD)         P(AD)          P(PD)
                           │
                           ▼
   3D Glass-Brain Defect Salience Projection + PDF Summary Report
```

---

## Performance Summary

Evaluated on a strictly held-out test cohort of 55 clinical scans across the **ABIDE**, **OASIS-1**, and **PPMI** repositories:

| Metric / Dimension | Value | Clinical Finding |
| :--- | :---: | :--- |
| **Bitwise Decision Accuracy** | **98.64%** | Consistent decision reliability across all 4 independent channels |
| **Alzheimer's Sensitivity** | **100.0%** | Full capture of hippocampal and temporoparietal atrophy |
| **Parkinson's Sensitivity** | **90.62%** | High sensitivity to nigrostriatal and striatal degeneration |
| **Comorbidity Capability** | Multi-Label | Uncoupled probability output allows multi-disorder detection |

---

## Project Structure

```text
Clean-Brain-Screening/
├── app.py                         # Flask backend, DICOM/NIfTI intake, inference pipeline
├── train_multidisease_production.py # NeuroGNN model architecture & training routines
├── checkpoints/
│   └── best_multidisease_model.pt # Trained NeuroGNN checkpoint weights (116-node)
├── static/
│   ├── app.js                     # Three.js 3D glass brain renderer & UI logic
│   ├── style.css                  # Modern dark-mode clinical dashboard styling
│   └── models/
│       └── brain.obj              # Anatomical 3D cortical surface mesh
├── templates/
│   └── index.html                 # CDSS intake, 3D viewer & reporting dashboard
├── figures/                       # Architectural diagrams & evaluation plots
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA-capable GPU (optional, CPU inference supported)
- Modern WebGL-enabled browser (Google Chrome, Firefox, Edge)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/NeuroScreen-CDSS.git
cd NeuroScreen-CDSS
```

### 3. Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch Geometric:** If using GPU acceleration, install PyTorch Geometric following the official wheel instructions for your CUDA version:
> ```bash
> pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
> ```

---

## Running the Application

1. Verify that your trained model checkpoint is present in the `checkpoints/` directory:
   ```bash
   checkpoints/best_multidisease_model.pt
   ```
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
4. Authenticate using clinician credentials, input patient demographic details, upload a de-identified NIfTI (`.nii` / `.nii.gz`) volume or axial slice preview, and click **Start AI Analysis**.

---

## Clinical Decision Support Workflow

1. **Intake & Localizer Verification:** Uploaded NIfTI scans undergo rapid affine parsing; the middle axial slice is extracted on the fly to confirm alignment and rule out motion artifacts.
2. **Atlas Parcellation:** Regional signal moments are extracted across the 116 AAL anatomical parcels.
3. **Graph Forward Pass:** The normalized graph $(\mathcal{V}, \mathcal{E}, X, A)$ passes through the 2-layer GNN; independent Sigmoid heads evaluate condition risks.
4. **Interactive 3D Glass Brain:** Salient defect nodes (e.g., Node 74 / Putamen at 92.4% salience in PD cases) are highlighted as glowing red spheres over baseline cyan nodes.
5. **Auditable Reporting:** Generates an archival diagnostic summary with patient demographics, predicted probabilities, identified anatomical drivers, and sign-off blocks.

---

## Datasets Utilized

- **ABIDE:** Autism Brain Imaging Data Exchange ([fcon_1000.projects.nitrc.org/indi/abide](https://fcon_1000.projects.nitrc.org/indi/abide/))
- **OASIS-1:** Open Access Series of Imaging Studies ([oasis-brains.org](https://www.oasis-brains.org/))
- **PPMI:** Parkinson's Progression Markers Initiative ([ppmi-info.org](https://www.ppmi-info.org/))

---

## Team & Acknowledgements

**Department of Artificial Intelligence and Data Science**  
**SCMS School of Engineering and Technology (SSET), Cochin, Kerala, India**  
*(Affiliated to APJ Abdul Kalam Technological University)*

- **Ajana Biju** (SCM23CD007)
- **Alan Parappil Agi** (SCM23CD011)
- **Amal N V** (SCM23CD013)
- **Devika A S** (SCM23CD025)

**Project Guide:** Dr. Sindhya K. Nambiar (Associate Professor & Head of Department, Dept. of AI & DS)  
**Project Coordinator:** Ms. Shali Sara Abraham (Assistant Professor, Dept. of AI & DS)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.