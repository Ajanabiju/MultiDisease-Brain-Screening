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

# Registered staff directory
CLINICAL_STAFF = {
    "neuro_lead": {
        "password": "ClinicalPass2026!",
        "name": "Dr. Parappil",
        "license": "KMC-NEURO-8842",
        "role": "Consultant Neurologist"
    }
}

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


def compute_connectome_defects(x_tensor, adj_matrix, top_k_nodes=6, top_k_edges=12):
    """
    Identifies top salient/impacted brain regions based on node feature energy.
    """
    node_feat = x_tensor.squeeze(0).cpu().numpy()[:116]   # Restrict to true 116 anatomical nodes
    adj = adj_matrix.squeeze(0).cpu().numpy()[:116, :116]

    node_norms = np.linalg.norm(node_feat, axis=-1)
    norm_scores = (node_norms - node_norms.min()) / (node_norms.max() - node_norms.min() + 1e-8) * 100
    top_node_indices = np.argsort(norm_scores)[::-1][:top_k_nodes]

    nodes = []
    for i in range(116):
        is_defect = i in top_node_indices
        label_name = AAL_LABELS[i] if i < len(AAL_LABELS) else f"ROI {i+1}"
        nodes.append({
            "id": i,
            "name": label_name,
            "x": float(COORDS_LIST[i][0]),
            "y": float(COORDS_LIST[i][1]),
            "z": float(COORDS_LIST[i][2]),
            "score": float(round(norm_scores[i], 1)),
            "is_defect": bool(is_defect)
        })

    edges = []
    candidate_edges = []
    for src in top_node_indices:
        for dst in range(116):
            if src != dst and adj[src, dst] > 0.05:
                edge_weight = float(adj[src, dst] * (norm_scores[src] + norm_scores[dst]) / 2.0)
                candidate_edges.append((edge_weight, int(src), int(dst)))

    candidate_edges.sort(key=lambda item: item[0], reverse=True)
    seen = set()
    for weight, u, v in candidate_edges:
        edge_key = tuple(sorted([u, v]))
        if edge_key not in seen:
            seen.add(edge_key)
            edges.append({"source": u, "target": v, "weight": float(round(weight, 3))})
            if len(edges) >= top_k_edges:
                break

    return nodes, edges


def process_uploaded_scan(file_path, filename):
    ext = filename.lower()
    
    # Case 1: 3D/4D NIfTI neuroimaging volume
    if ext.endswith('.nii') or ext.endswith('.nii.gz') or ext.endswith('.hdr'):
        img = nib.load(file_path)
        data = img.get_fdata()
        
        if data.ndim == 4:
            # 4D rs-fMRI (ASD)
            try:
                ts = masker.transform(img)
            except Exception:
                mean_3d = nib.Nifti1Image(np.mean(data, axis=-1), img.affine)
                signals = masker.transform(mean_3d).squeeze()
                if len(signals) < 116:
                    signals = np.pad(signals, (0, 116 - len(signals)))
                else:
                    signals = signals[:116]
                ts = np.tile(signals, (10, 1))

            corr = np.corrcoef(ts.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

            if corr.shape[0] < 116:
                corr = np.pad(corr, ((0, 116 - corr.shape[0]), (0, 116 - corr.shape[0])), mode='constant')
            else:
                corr = corr[:116, :116]

            np.fill_diagonal(corr, 1.0)
            adj = (corr + 1.0) / 2.0
            deg_strength = np.sum(adj, axis=1, keepdims=True)
            conn_var = np.std(adj, axis=1, keepdims=True)
            f1 = np.repeat(deg_strength, 32, axis=1).astype(np.float32)
            f2 = np.repeat(conn_var, 32, axis=1).astype(np.float32)
        else:
            # 3D Structural MRI (AD / PD / HC)
            if data.ndim > 3:
                data = data[..., 0]
            img_3d = nib.Nifti1Image(data, img.affine)
            signals = masker.transform(img_3d).squeeze()
            if signals.ndim == 0:
                signals = np.zeros(116, dtype=np.float32)
            elif len(signals) < 116:
                signals = np.pad(signals, (0, 116 - len(signals)))
            else:
                signals = signals[:116]
            f1 = np.repeat(signals[:, None], 32, axis=1).astype(np.float32)
            f2 = np.repeat((signals * 0.88)[:, None], 32, axis=1).astype(np.float32)
            adj = BASE_ADJ.copy().astype(np.float32)

    # Case 2: 2D MRI Slice (PNG / JPG)
    else:
        pil_img = Image.open(file_path).convert('L').resize((116, 64))
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        f1 = arr[:, :32]
        f2 = arr[:, 32:]
        adj = BASE_ADJ.copy().astype(np.float32)

    x = np.concatenate([f1, f2], axis=-1)  # (116, 64)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(adj, dtype=torch.float32).unsqueeze(0)


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


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        username = data.get("username", "").strip()
        password = data.get("password", "")

        user = CLINICAL_STAFF.get(username)
        if user and user["password"] == password:
            session["clinician_authenticated"] = True
            session["clinician_name"] = user["name"]
            session["clinician_license"] = user["license"]
            session["clinician_role"] = user["role"]
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid Clinical Staff ID or Security Passcode.")
    
    if session.get("clinician_authenticated"):
        return redirect(url_for("home"))
    return render_template("login.html")


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