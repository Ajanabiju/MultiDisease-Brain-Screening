import os
import sys
import tempfile
from functools import wraps
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from PIL import Image
from scipy.spatial.distance import cdist

# Set root directory to current file directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train_multidisease_production import ProductionNeuroGNN

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "neuro-screening-secure-key-9281")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Support up to 500MB
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(BASE_DIR, "neuro_users.db")

def init_db():
    """Initializes the SQLite database and seeds default staff if not present."""
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
    
    # Seed default account: neuro_lead / ClinicalPass2026!
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

# Initialize SQLite database on app startup
init_db()


def get_clinician_by_staff_id(staff_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clinicians WHERE staff_id = ?", (staff_id,))
    user = cursor.fetchone()
    conn.close()
    return user


# ==========================================
# AUTHENTICATION ROUTES (DATABASE BACKED)
# ==========================================

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
    print(f"\n[SUCCESS] Production model successfully loaded from {CHECKPOINT_PATH}\n")
else:
    raise FileNotFoundError(f"[ERROR] Checkpoint missing at {CHECKPOINT_PATH}. Train the model first.")

# Pre-load Atlas & Masker for parcellation
print("[*] Preparing AAL-116 Atlas Parcellation Masker...")
aal = datasets.fetch_atlas_aal()
aal_maps = aal.maps

try:
    from nilearn.plotting import find_parcellation_cut_coords
    coords = find_parcellation_cut_coords(labels_img=aal_maps)
except (AttributeError, ImportError):
    from nilearn.maskers.parcellations import _find_parcellation_cut_coords
    coords = _find_parcellation_cut_coords(labels_img=aal_maps)

AAL_LABELS = [str(lbl).replace("_", " ") for lbl in aal.labels] if hasattr(aal, 'labels') else [f"ROI {i+1}" for i in range(116)]
COORDS_LIST = coords.tolist()

# Construct Base Spatial Adjacency
dist_matrix = cdist(coords, coords, metric='euclidean')
sigma = np.std(dist_matrix) if np.std(dist_matrix) != 0 else 1.0
BASE_ADJ = np.exp(- (dist_matrix ** 2) / (2 * (sigma ** 2))).astype(np.float32)

np.fill_diagonal(BASE_ADJ, 1.0)
deg = np.sum(BASE_ADJ, axis=1, keepdims=True)
deg[deg == 0] = 1.0
BASE_ADJ /= deg

masker = NiftiLabelsMasker(labels_img=aal_maps, standardize=None, resampling_target='labels')
masker.fit()


def adjust_tensor_dimensions(x, adj, target_nodes=116):
    """
    Ensures both x and adj tensors match the exact 116-node architecture
    expected by BatchNorm and Graph Convolution layers.
    x shape:   [1, 116, 64] -> [1, 116, 64]
    adj shape: [1, 116, 116] -> [1, 116, 116]
    """
    batch_size, current_nodes, feat_dim = x.shape

    if current_nodes < target_nodes:
        diff = target_nodes - current_nodes
        # Pad node features with zeros
        pad_x = torch.zeros((batch_size, diff, feat_dim), dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad_x], dim=1)

        # Pad adjacency matrix with an identity block (self-loops for added nodes)
        new_adj = torch.eye(target_nodes, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        new_adj[:, :current_nodes, :current_nodes] = adj
        adj_padded = new_adj
        return x_padded, adj_padded

    elif current_nodes > target_nodes:
        return x[:, :target_nodes, :], adj[:, :target_nodes, :target_nodes]

    return x, adj


def compute_connectome_defects(x, adj, top_k=8):
    """
    Computes regional salience scores across brain parcels.
    Guarantees consistent 1D shapes for broadcasting and avoids shape mismatches.
    """
    # 1. Convert to numpy and peel off batch dimension
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)

    if isinstance(adj, torch.Tensor):
        adj_np = adj.detach().cpu().numpy()
    else:
        adj_np = np.array(adj)

    while x_np.ndim > 2:
        x_np = x_np[0]
    while adj_np.ndim > 2:
        adj_np = adj_np[0]

    # 2. Determine target node count strictly from adjacency matrix
    num_nodes = adj_np.shape[0]

    # 3. Orient x_np so shape is (num_nodes, features)
    if x_np.shape[0] != num_nodes and x_np.shape[1] == num_nodes:
        x_np = x_np.T

    # If x_np still doesn't match node count, truncate or pad along axis 0
    if x_np.shape[0] > num_nodes:
        x_np = x_np[:num_nodes, :]
    elif x_np.shape[0] < num_nodes:
        pad = np.zeros((num_nodes - x_np.shape[0], x_np.shape[1]), dtype=x_np.dtype)
        x_np = np.vstack([x_np, pad])

    # 4. Calculate node energy across features (axis=1) -> shape: (num_nodes,)
    node_energies = np.mean(np.abs(x_np), axis=1)

    # 5. Calculate degree centrality across nodes (axis=1) -> shape: (num_nodes,)
    degree_centrality = np.sum(np.abs(adj_np), axis=1)

    # Ensure both arrays are strictly 1D and equal length before multiplying
    node_energies = np.ravel(node_energies)[:num_nodes]
    degree_centrality = np.ravel(degree_centrality)[:num_nodes]

    combined_scores = node_energies * (1.0 + degree_centrality)

    # 6. Normalize scores between 0.1 and 1.0
    min_s, max_s = np.min(combined_scores), np.max(combined_scores)
    if max_s > min_s:
        norm_scores = 0.1 + 0.9 * ((combined_scores - min_s) / (max_s - min_s))
    else:
        norm_scores = np.ones(num_nodes) * 0.5

    # 7. Extract top salient defect nodes
    top_indices = set(np.argsort(norm_scores)[-top_k:].tolist())

    nodes_3d = []
    for idx in range(num_nodes):
        nodes_3d.append({
            "id": int(idx),
            "score": float(norm_scores[idx]),
            "is_defect": bool(idx in top_indices)
        })

    # 8. Extract defect edges
    edges_3d = []
    for src in top_indices:
        for dst in top_indices:
            if src < dst and adj_np[src, dst] > 0.15:
                edges_3d.append({
                    "source": int(src),
                    "target": int(dst),
                    "weight": float(adj_np[src, dst])
                })

    return nodes_3d, edges_3d
import nibabel as nib
import numpy as np
import torch
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

# Global cache for AAL atlas so it doesn't re-download on every analysis
AAL_DATASET = None

def get_aal_atlas():
    global AAL_DATASET
    if AAL_DATASET is None:
        AAL_DATASET = datasets.fetch_atlas_aal(version='SPM12')
    return AAL_DATASET


def extract_aal116_graph_from_nii(nii_path):
    """
    Extracts strictly:
      x:   [1, 116, 64]   (116 anatomical ROIs, 64 node features)
      adj: [1, 116, 116]  (116 x 116 connectome adjacency matrix)
    """
    atlas = get_aal_atlas()
    atlas_filename = atlas.maps

    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize=True,
        memory='nilearn_cache',
        verbose=0
    )

    # 1. Extract regional signal from the 116 AAL parcels
    signals = masker.fit_transform(nii_path)  # 4D: (time_points, parcels), 3D: (1, parcels)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    n_samples, n_parcels = signals.shape

    # Ensure exactly 116 parcels
    if n_parcels > 116:
        signals = signals[:, :116]
    elif n_parcels < 116:
        pad = np.zeros((n_samples, 116 - n_parcels), dtype=signals.dtype)
        signals = np.hstack([signals, pad])

    # 2. Build Node Features X of shape (1, 116, 64)
    if n_samples >= 64:
        # 4D functional MRI: first 64 temporal points as features
        feat_matrix = signals[:64, :].T  # Transpose to (116, 64)
    else:
        # 3D structural MRI (T1/T2): synthesize 64 feature dimensions across parcels
        base_vals = signals[0, :]  # shape: (116,)
        phases = np.linspace(0.1, 2.0, 64)  # shape: (64,)
        feat_matrix = np.outer(base_vals, phases)  # shape: (116, 64)

    # 3. Build Adjacency Matrix A of shape (1, 116, 116)
    if n_samples > 10:
        # Functional Pearson correlation
        with np.errstate(all='ignore'):
            corr = np.corrcoef(signals.T)
            corr = np.nan_to_num(corr, nan=0.0)
    else:
        # Structural/spatial Euclidean distance kernel between region centroids
        coords = np.array(atlas.region_coords)[:116]
        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))
        sigma = 20.0
        corr = np.exp(- (dist_matrix ** 2) / (2.0 * (sigma ** 2)))

    # Sparsify connectivity and set diagonal
    corr[corr < 0.20] = 0.0
    np.fill_diagonal(corr, 1.0)
    adj_matrix = corr[:116, :116]

    # Convert to PyTorch tensors with batch dimension
    x_tensor = torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)   # [1, 116, 64]
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0) # [1, 116, 116]

    return x_tensor, adj_tensor
def process_uploaded_scan(file_path, filename):
    filename_lower = filename.lower()
    if not (filename_lower.endswith(".nii") or filename_lower.endswith(".nii.gz")):
        raise ValueError("Unsupported format. Clinical CDSS requires 3D/4D volumetric NIfTI (.nii, .nii.gz).")
    return extract_aal116_graph_from_nii(file_path)
# ==========================================
# AUTHENTICATION & SECURE CLINICAL ROUTING
# ==========================================

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
    return jsonify({
        "error": "The uploaded MRI volume exceeds the maximum file size limit (500MB). Please compress (.nii.gz) or upload a smaller volume."
    }), 413


@app.route('/favicon.ico')
def favicon():
    return '', 204




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


@app.route("/predict", methods=["POST"])
@clinician_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No scan file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    patient_name = request.form.get("patient_name", request.form.get("patient_id", "ANON-SUBJ-001"))
    patient_age = request.form.get("patient_age", "N/A")
    patient_gender = request.form.get("patient_gender", "N/A")
    patient_phone = request.form.get("patient_phone", "N/A")
    scan_modality = request.form.get("scan_modality", "3D MRI")

    suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    file.save(tmp_path)

    try:
        # 1. Feature extraction from scan (116 regions)
        x, adj = process_uploaded_scan(tmp_path, file.filename)
        x = x.to(device)
        adj = adj.to(device)

        # 2. Extract 3D defective regions on true anatomical AAL-116 nodes
        nodes_3d, edges_3d = compute_connectome_defects(x, adj)

        # 3. Adapt dimensions to 116 nodes for the GNN forward pass
        x_model, adj_model = adjust_tensor_dimensions(x, adj, target_nodes=116)

        # Ensure float32 precision
        x_model = x_model.float()
        adj_model = adj_model.float()

        with torch.no_grad():
            logits = model(x_model, adj_model)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_info = DISEASE_CLASSES[pred_idx]
        probabilities = {DISEASE_CLASSES[i]["name"]: float(round(probs[i] * 100, 2)) for i in range(4)}

        return jsonify({
            "status": "success",
            "subject_meta": {
                "patient_name": patient_name,
                "patient_age": patient_age,
                "patient_gender": patient_gender,
                "patient_phone": patient_phone,
                "modality": scan_modality,
                "evaluating_clinician": session.get("clinician_name"),
                "license": session.get("clinician_license")
            },
            "prediction": pred_info["name"],
            "badge": pred_info["badge"],
            "description": pred_info["desc"],
            "confidence": float(round(probs[pred_idx] * 100, 2)),
            "probabilities": probabilities,
            "connectome_3d": {
                "nodes": nodes_3d,
                "edges": edges_3d
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)