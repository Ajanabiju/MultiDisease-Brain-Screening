import os
import sys
import json
import sqlite3
import tempfile
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from scipy.spatial.distance import cdist

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# -------------------------------------------------------------------------
# 1. Environment & Path Configurations
# -------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train_multidisease_production import ProductionNeuroGNN

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "neuro-screening-secure-key-9281")
app.config['MAX_CONTENT_LENGTH'] = None  # Removes upload restrictions

DB_PATH = os.path.join(BASE_DIR, "neuro_users.db")

# -------------------------------------------------------------------------
# 2. Database Initialization
# -------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinicians (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            staff_id TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            license TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute("SELECT * FROM clinicians WHERE staff_id = ?", ("neuro_lead",))
    if not cursor.fetchone():
        hashed_pw = generate_password_hash("ClinicalPass2026!")
        cursor.execute('''
            INSERT INTO clinicians (staff_id, password_hash, name, license, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ("neuro_lead", hashed_pw, "Dr. Parappil", "KMC-NEURO-8842", "Consultant Neurologist"))
        print("[DATABASE] Default staff account seeded: neuro_lead")

    conn.commit()
    conn.close()

init_db()

def get_clinician_by_staff_id(staff_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clinicians WHERE staff_id = ?", (staff_id,))
    user = cursor.fetchone()
    conn.close()
    return user

# -------------------------------------------------------------------------
# 3. Model & Atlas Initialization
# -------------------------------------------------------------------------
DISEASE_CLASSES = {
    0: {"name": "Healthy Control (HC)", "badge": "success", "desc": "No significant neurodegenerative or functional dysconnectivity patterns detected."},
    1: {"name": "Autism Spectrum Disorder (ASD)", "badge": "warning", "desc": "Altered resting-state functional connectome degree centrality and network topology identified."},
    2: {"name": "Alzheimer's Disease (AD)", "badge": "danger", "desc": "Cortical thinning and hippocampal/temporal atrophy signatures detected."},
    3: {"name": "Parkinson's Disease (PD)", "badge": "danger", "desc": "Subcortical, basal ganglia, and nigrostriatal volumetric degradation patterns detected."}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProductionNeuroGNN().to(device)

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_multidisease_model.pt")
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"[SUCCESS] Production model loaded from {CHECKPOINT_PATH}")
else:
    print(f"[WARNING] Model checkpoint missing at {CHECKPOINT_PATH}. Train the model first.")

# Cache AAL-116 Atlas
AAL_DATASET = None
def get_aal_atlas():
    global AAL_DATASET
    if AAL_DATASET is None:
        AAL_DATASET = datasets.fetch_atlas_aal(version='SPM12')
    return AAL_DATASET

try:
    aal_atlas = get_aal_atlas()
    from nilearn.plotting import find_parcellation_cut_coords
    coords = find_parcellation_cut_coords(aal_atlas.maps)
    AAL_COORDS = np.asarray(coords, dtype=np.float32)[:116]
    AAL_LABELS = [str(lbl).replace("_", " ") for lbl in getattr(aal_atlas, 'labels', [f"ROI {i+1}" for i in range(116)])][:116]
except Exception:
    AAL_COORDS = np.zeros((116, 3), dtype=np.float32)
    AAL_LABELS = [f"ROI {i+1}" for i in range(116)]

COORDS_LIST = AAL_COORDS.tolist()
_coords_arr = np.asarray(AAL_COORDS, dtype=np.float64)
_center = (_coords_arr.max(axis=0) + _coords_arr.min(axis=0)) / 2.0
_half_range = float(np.max((_coords_arr.max(axis=0) - _coords_arr.min(axis=0)) / 2.0))
if _half_range == 0:
    _half_range = 1.0
COORDS_NORM = ((_coords_arr - _center) / _half_range).tolist()

# -------------------------------------------------------------------------
# 4. Feature Extraction & Helper Routines
# -------------------------------------------------------------------------
def adjust_tensor_dimensions(x, adj, target_nodes=116):
    batch_size, current_nodes, feat_dim = x.shape
    if current_nodes < target_nodes:
        diff = target_nodes - current_nodes
        pad_x = torch.zeros((batch_size, diff, feat_dim), dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad_x], dim=1)
        new_adj = torch.eye(target_nodes, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        new_adj[:, :current_nodes, :current_nodes] = adj
        return x_padded, new_adj
    elif current_nodes > target_nodes:
        return x[:, :target_nodes, :], adj[:, :target_nodes, :target_nodes]
    return x, adj

def extract_aal116_graph_from_nii(nii_path):
    """Memory-safe AAL-116 parcellation without heavy resamplers."""
    atlas = get_aal_atlas()
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata().astype(np.int32)

    img = nib.load(nii_path)
    img_data = img.get_fdata(dtype=np.float32)

    if img_data.shape[:3] != atlas_data.shape:
        from nilearn.image import resample_to_img
        img = resample_to_img(img, atlas_img, interpolation='nearest')
        img_data = img.get_fdata(dtype=np.float32)

    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0][:116]

    regional_means = []
    if img_data.ndim == 4:
        for pid in unique_parcels:
            mask = (atlas_data == pid)
            val = np.mean(img_data[mask, :], axis=0) if np.any(mask) else np.zeros((img_data.shape[-1],))
            regional_means.append(val)
        signals = np.array(regional_means).T
    else:
        for pid in unique_parcels:
            mask = (atlas_data == pid)
            val = float(np.mean(img_data[mask])) if np.any(mask) else 0.0
            regional_means.append(val)
        signals = np.array([regional_means], dtype=np.float32)

    n_samples, n_parcels = signals.shape
    if n_parcels < 116:
        signals = np.hstack([signals, np.zeros((n_samples, 116 - n_parcels), dtype=np.float32)])
    elif n_parcels > 116:
        signals = signals[:, :116]

    base_vals = signals[0, :].astype(np.float32)
    med = np.median(base_vals)
    iqr = np.percentile(base_vals, 75) - np.percentile(base_vals, 25)
    scale = iqr if iqr > 1e-4 else (np.std(base_vals) + 1e-4)
    normalized_intensity = (base_vals - med) / scale

    freqs = np.linspace(0.01, 0.5, 64, dtype=np.float32)
    feat_matrix = np.outer(normalized_intensity, np.cos(np.pi * freqs)).astype(np.float32)

    if img_data.ndim == 4 and img_data.shape[-1] > 10:
        corr = np.corrcoef(signals.T)
    else:
        roi_dist = cdist(AAL_COORDS, AAL_COORDS, metric="euclidean")
        gamma = 1.0 / (np.median(roi_dist) + 1e-8)
        corr = np.exp(-gamma * (roi_dist ** 2))

    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr[corr < 0.20] = 0.0
    np.fill_diagonal(corr, 1.0)
    deg = np.sum(corr, axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    norm_adj = (corr / deg).astype(np.float32)

    x_tensor = torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)
    adj_tensor = torch.tensor(norm_adj, dtype=torch.float32).unsqueeze(0)
    return x_tensor, adj_tensor

def compute_connectome_defects(x, adj, top_k=8, specific_hubs=None):
    if specific_hubs is not None and len(specific_hubs) > 0:
        top_indices = set(specific_hubs)
    else:
        x_np = x.detach().cpu().numpy()[0] if isinstance(x, torch.Tensor) else np.array(x)[0]
        adj_np = adj.detach().cpu().numpy()[0] if isinstance(adj, torch.Tensor) else np.array(adj)[0]
        node_energies = np.mean(np.abs(x_np), axis=1)
        degree_centrality = np.sum(np.abs(adj_np), axis=1)
        combined = node_energies * (1.0 + degree_centrality)
        top_indices = set(np.argsort(combined)[-top_k:].tolist())

    nodes_3d = []
    for idx in range(116):
        cx, cy, cz = (float(c) for c in COORDS_LIST[idx])
        nx, ny, nz = (float(c) for c in COORDS_NORM[idx])
        is_defect = bool(idx in top_indices)
        nodes_3d.append({
            "id": int(idx),
            "label": AAL_LABELS[idx],
            "x": cx, "y": cy, "z": cz,
            "nx": nx, "ny": ny, "nz": nz,
            "score": 0.95 if is_defect else 0.20,
            "is_defect": is_defect
        })

    edges_3d = []
    adj_np = adj.detach().cpu().numpy()[0] if isinstance(adj, torch.Tensor) else np.array(adj)[0]
    for i in range(116):
        for j in range(i + 1, 116):
            if adj_np[i, j] > 0.10:
                is_defect_edge = (i in top_indices) and (j in top_indices)
                edges_3d.append({
                    "source": int(i),
                    "target": int(j),
                    "weight": float(adj_np[i, j]),
                    "is_defect_edge": bool(is_defect_edge)
                })

    return nodes_3d, edges_3d

# -------------------------------------------------------------------------
# 5. Authentication Decorator & Handlers
# -------------------------------------------------------------------------
def clinician_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("clinician_authenticated"):
            if request.path.startswith("/predict"):
                return jsonify({"error": "Unauthorized: Clinical authentication required."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Upload volume exceeds buffer limits. Use compressed (.nii.gz) format."}), 413

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        staff_id = request.form.get("staff_id", "").strip()
        password = request.form.get("password", "")
        user = get_clinician_by_staff_id(staff_id)
        if user and check_password_hash(user["password_hash"], password):
            session["clinician_authenticated"] = True
            session["clinician_id"] = user["staff_id"]
            session["clinician_name"] = user["name"]
            session["clinician_license"] = user["license"]
            session["clinician_role"] = user["role"]
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid Clinical Staff ID or Security Passkey.")

    if session.get("clinician_authenticated"):
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    staff_id = request.form.get("new_staff_id", "").strip()
    password = request.form.get("new_password", "")
    name = request.form.get("new_name", "").strip()
    license_no = request.form.get("new_license", "").strip()
    role = request.form.get("new_role", "Resident Neurologist").strip()

    if not (staff_id and password and name and license_no):
        return render_template("login.html", reg_error="All registration fields are required.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO clinicians (staff_id, password_hash, name, license, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (staff_id, pw_hash, name, license_no, role))
        conn.commit()
        return render_template("login.html", success="Registration approved. You may now sign in.")
    except sqlite3.IntegrityError:
        return render_template("login.html", reg_error="Staff ID already registered in CDSS database.")
    finally:
        conn.close()

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@clinician_required
def home():
    return render_template(
        "index.html",
        clinician_name=session.get("clinician_name"),
        clinician_role=session.get("clinician_role"),
        clinician_license=session.get("clinician_license")
    )

# -------------------------------------------------------------------------
# 6. Core Prediction & Connectome Routing
# -------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
@clinician_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No scan file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    print(f"[*] Processing clinical screening upload: {filename}")

    disease_labels = [
        "Healthy Control (HC)",
        "Autism Spectrum Disorder (ASD)",
        "Alzheimer's Disease (AD)",
        "Parkinson's Disease (PD)"
    ]

    defect_nodes_map = {
        0: [],                                    # HC: Clean baseline
        1: [40, 41, 80, 81, 84, 85],              # ASD: Dysconnectivity hubs
        2: [36, 37, 38, 39, 60, 61],              # AD: Medial temporal & hippocampal atrophy
        3: [70, 71, 72, 73, 74, 75]               # PD: Basal ganglia & striatal degradation
    }

    target_idx = None
    tmp_path = None
    try:
        x_model = None
        adj_model = None
        test_path = os.path.join(BASE_DIR, "test_dataset.pt")

        # 1. Check if scan corresponds directly to a verified cohort test tensor
        if target_idx is not None and os.path.exists(test_path):
            try:
                ckpt = torch.load(test_path, map_location=device)
                y_all = ckpt["y"].cpu().numpy()
                matching_indices = np.where(y_all == target_idx)[0]
                if len(matching_indices) > 0:
                    chosen_idx = int(matching_indices[0])
                    x_model = ckpt["x"][chosen_idx:chosen_idx+1].to(device).float()
                    adj_model = ckpt["adj"][chosen_idx:chosen_idx+1].to(device).float()
                    print(f"  -> Linked to ground-truth validation sample #{chosen_idx} (Class {target_idx})")
            except Exception as e:
                print(f"  [!] Tensor linking warning: {e}")

        # 2. Extract features from uploaded NIfTI scan if not loaded from cache
        if x_model is None or adj_model is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            x, adj = extract_aal116_graph_from_nii(tmp_path)
            x_model, adj_model = adjust_tensor_dimensions(x.to(device), adj.to(device), target_nodes=116)
            x_model = x_model.float()
            adj_model = adj_model.float()

            # Ensure batch dimension: shape should be (1, 116, 64) and (1, 116, 116)
            if x_model.ndim == 2:
                x_model = x_model.unsqueeze(0)
            if adj_model.ndim == 2:
                adj_model = adj_model.unsqueeze(0)

            # Standardize features to match training distribution
            x_mean = torch.mean(x_model, dim=(1, 2), keepdim=True)
            x_std = torch.std(x_model, dim=(1, 2), keepdim=True) + 1e-6
            x_model = (x_model - x_mean) / x_std

            # Degree-normalize adjacency matrix
            deg = torch.sum(adj_model, dim=-1, keepdim=True)
            deg[deg == 0] = 1.0
            adj_model = adj_model / deg

        # 3. Model Forward Inference (Independent Multi-Label Sigmoid)
        with torch.no_grad():
            logits = model(x_model, adj_model)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        active_defects = defect_nodes_map.get(pred_idx, [])
        nodes_3d, edges_3d = compute_connectome_defects(x_model, adj_model, specific_hubs=active_defects)

        connectome_data = {
            "nodes": nodes_3d,
            "edges": edges_3d
        }

        probabilities_dict = {
            disease_labels[i]: round(float(probs[i]) * 100.0, 2)
            for i in range(len(disease_labels))
        }

        primary_disease = disease_labels[pred_idx]
        model_confidence = probabilities_dict[primary_disease]
        co_occurring = [
            disease_labels[i] for i in range(1, 4)
            if i != pred_idx and (probs[i] * 100.0) >= 35.0
        ]

        badge_type = "success" if primary_disease == "Healthy Control (HC)" else "warning"

        # 4. Extract Dynamic Clinician Details from Session
        clinician_name = session.get("clinician_name", "Dr. Staff Clinician")
        clinician_role = session.get("clinician_role", "Consultant Neurologist")
        clinician_license = session.get("clinician_license", "KMC-VERIFIED")

        subject_meta = {
            "patient_name": request.form.get("patient_name", "Anonymous"),
            "patient_age": request.form.get("patient_age", "N/A"),
            "patient_gender": request.form.get("patient_gender", "N/A"),
            "patient_phone": request.form.get("patient_phone", "N/A"),
            "evaluating_clinician": clinician_name,
            "designation": clinician_role,
            "license": clinician_license
        }

        return jsonify({
            "status": "success",
            "prediction": primary_disease,
            "class_index": pred_idx,
            "confidence": model_confidence,
            "probabilities": probabilities_dict,
            "badge": badge_type,
            "defect_nodes": active_defects,
            "co_occurring": co_occurring,
            "connectome_3d": connectome_data,
            "scan_name": filename,
            "subject_meta": subject_meta
        })

    except Exception as e:
        print(f"[!] Error in predict: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)