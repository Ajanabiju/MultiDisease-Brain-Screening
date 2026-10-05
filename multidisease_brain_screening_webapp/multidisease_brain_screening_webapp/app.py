import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from PIL import Image

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train_multidisease_production import ProductionNeuroGNN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

DISEASE_CLASSES = {
    0: {"name": "Healthy Control (HC)", "badge": "success", "desc": "No significant neurodegenerative or functional dysconnectivity patterns detected."},
    1: {"name": "Autism Spectrum Disorder (ASD)", "badge": "warning", "desc": "Altered resting-state functional connectome degree centrality and network topology identified."},
    2: {"name": "Alzheimer's Disease (AD)", "badge": "danger", "desc": "Cortical thinning and hippocampal/temporal atrophy signatures detected."},
    3: {"name": "Parkinson's Disease (PD)", "badge": "danger", "desc": "Subcortical, basal ganglia, and nigrostriatal volumetric degradation patterns detected."}
}

# Load Model Checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProductionNeuroGNN().to(device)

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_multidisease_model.pt")
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"[*] Production model successfully loaded from {CHECKPOINT_PATH}")
else:
    print(f"[!] Warning: Checkpoint not found at {CHECKPOINT_PATH}. Model initialized with random weights.")

# Pre-load Atlas & Masker for parcellation
print("[*] Preparing AAL-116 Atlas Parcellation Masker...")
aal = datasets.fetch_atlas_aal()
aal_maps = aal.maps
coords = plotting.find_parcellation_cut_coords(labels_img=aal_maps)
coords = np.array(coords, dtype=np.float32)
if len(coords) < 116:
    coords = np.pad(coords, ((0, 116 - len(coords)), (0, 0)), mode='edge')
else:
    coords = coords[:116]

diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sqrt(np.sum(diff ** 2, axis=-1))
sigma = np.std(dist) if np.std(dist) > 0 else 1.0
BASE_ADJ = np.exp(- (dist ** 2) / (2 * (sigma ** 2)))
np.fill_diagonal(BASE_ADJ, 1.0)
deg = np.sum(BASE_ADJ, axis=1, keepdims=True)
deg[deg == 0] = 1.0
BASE_ADJ /= deg

masker = NiftiLabelsMasker(labels_img=aal_maps, standardize=None, resampling_target='labels')
masker.fit()


def process_uploaded_scan(file_path, filename):
    ext = filename.lower()
    
    # Case 1: 3D/4D NIfTI neuroimaging volume
    if ext.endswith('.nii') or ext.endswith('.nii.gz') or ext.endswith('.hdr'):
        img = nib.load(file_path)
        data = img.get_fdata()
        
        if data.ndim == 4:
            # 4D rs-fMRI: Extract functional connectome
            ts = masker.transform(img)
            corr = np.corrcoef(ts.T)
            if corr.shape[0] < 116:
                corr = np.pad(corr, ((0, 116 - corr.shape[0]), (0, 116 - corr.shape[0])), mode='constant')
            else:
                corr = corr[:116, :116]
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            adj = (corr + 1.0) / 2.0
            deg_strength = np.sum(adj, axis=1, keepdims=True)
            conn_var = np.std(adj, axis=1, keepdims=True)
            f1 = np.repeat(deg_strength, 32, axis=1).astype(np.float32)
            f2 = np.repeat(conn_var, 32, axis=1).astype(np.float32)
        else:
            # 3D Structural MRI (e.g. OASIS, PPMI)
            if data.ndim > 3:
                data = data[..., 0]
            img_3d = nib.Nifti1Image(data, img.affine)
            signals = masker.transform(img_3d).squeeze()
            if len(signals) < 116:
                signals = np.pad(signals, (0, 116 - len(signals)))
            else:
                signals = signals[:116]
            f1 = np.repeat(signals[:, None], 32, axis=1).astype(np.float32)
            f2 = np.repeat((signals * 0.88)[:, None], 32, axis=1).astype(np.float32)
            adj = BASE_ADJ.copy().astype(np.float32)

    # Case 2: 2D MRI Slice (PNG / JPG)
    else:
        pil_img = Image.open(file_path).convert('L').resize((116, 64))
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # (116, 64)
        f1 = arr[:, :32]
        f2 = arr[:, 32:]
        adj = BASE_ADJ.copy().astype(np.float32)

    x = np.concatenate([f1, f2], axis=-1)  # shape: (116, 64)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(adj, dtype=torch.float32).unsqueeze(0)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        x, adj = process_uploaded_scan(tmp_path, file.filename)
        x = x.to(device)
        adj = adj.to(device)

        with torch.no_grad():
            logits = model(x, adj)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_info = DISEASE_CLASSES[pred_idx]
        probabilities = {DISEASE_CLASSES[i]["name"]: float(round(probs[i] * 100, 2)) for i in range(4)}

        return jsonify({
            "status": "success",
            "prediction": pred_info["name"],
            "class_index": pred_idx,
            "badge": pred_info["badge"],
            "description": pred_info["desc"],
            "confidence": float(round(probs[pred_idx] * 100, 2)),
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)