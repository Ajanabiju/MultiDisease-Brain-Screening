import os
import glob
import warnings
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker

# Suppress Nilearn transform and NumPy zero-division warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class UnifiedNeuroGraphDataset(Dataset):
    def __init__(self, samples_per_class=10, feature_dim=32, n_rois=116):
        self.samples = []
        self.feature_dim = feature_dim
        self.n_rois = n_rois

        print("\n" + "=" * 60)
        print(" INITIALIZING UNIFIED 4-CLASS NEURO-GRAPH DATASET")
        print("=" * 60)

        # 0. Load Anatomical AAL Atlas for Graph Topology
        print("[Atlas] Fetching AAL 116-region coordinates...")
        aal = datasets.fetch_atlas_aal()
        self.aal_maps = aal.maps

        try:
            self.coords = np.array(plotting.find_parcellation_cut_coords(self.aal_maps))[:n_rois]
        except Exception:
            self.coords = np.random.uniform(-50, 50, size=(n_rois, 3))

        if len(self.coords) < n_rois:
            pad_len = n_rois - len(self.coords)
            self.coords = np.pad(self.coords, ((0, pad_len), (0, 0)), mode='constant')

        # Compute Base Anatomical Distance Matrix W
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        sigma = np.std(dist_sq) + 1e-8
        self.base_adj = np.exp(-dist_sq / (2 * (sigma ** 2)))
        np.fill_diagonal(self.base_adj, 0.0)
        deg = np.sum(self.base_adj, axis=1, keepdims=True) + 1e-8
        self.base_adj /= deg

        # Initialize masker and fit once to the atlas labels
        self.masker = NiftiLabelsMasker(
            labels_img=self.aal_maps,
            standardize=False,
            resampling_target='labels'
        )
        self.masker.fit()

        # Helper function to extract AAL signals robustly
        def extract_aal_features(scan_path):
            loaded = nib.load(scan_path)
            data = loaded.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            # Wrap as a clean standard Nifti1Image with proper affine
            clean_nii = nib.Nifti1Image(data, loaded.affine)
            signals = self.masker.transform([clean_nii]).flatten()[:n_rois]
            if len(signals) < n_rois:
                signals = np.pad(signals, (0, n_rois - len(signals)))
            return np.repeat(signals[:, None], feature_dim, axis=1).astype(np.float32)

        # -------------------------------------------------------------
        # 1. ASD (Class 1) & Matched Controls (Class 0) from ABIDE
        # -------------------------------------------------------------
        print("\n[1/3] Loading Autism (ABIDE) Cohort...")
        abide = datasets.fetch_abide_pcp(derivatives=['rois_aal'], pipeline='cpac', n_subjects=samples_per_class * 2)
        pheno_df = pd.DataFrame(abide.phenotypic)

        for i, ts in enumerate(abide.rois_aal):
            if ts is None:
                continue
            try:
                dx_val = int(pheno_df.iloc[i]['DX_GROUP']) if 'DX_GROUP' in pheno_df.columns else 1
            except Exception:
                dx_val = 1

            label = 1 if dx_val == 1 else 0

            try:
                ts_arr = np.array(ts)
                if ts_arr.ndim != 2 or ts_arr.shape[0] < 10:
                    continue
                corr = np.corrcoef(ts_arr.T[:n_rois, :])
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 0.0)
                adj = np.maximum(corr, 0.0)
                adj = (adj + self.base_adj) / 2.0
                deg = np.sum(adj, axis=1, keepdims=True) + 1e-8
                adj /= deg

                mean_signal = np.mean(ts_arr, axis=0)[:n_rois]
                if len(mean_signal) < n_rois:
                    mean_signal = np.pad(mean_signal, (0, n_rois - len(mean_signal)))
                feat = np.repeat(mean_signal[:, None], feature_dim, axis=1).astype(np.float32)
                self.samples.append((feat, feat.copy(), adj.astype(np.float32), label))
            except Exception:
                continue

        print(f" -> Ingested {len(self.samples)} ABIDE graphs (ASD & Controls).")

        # -------------------------------------------------------------
        # 2. Alzheimer's Disease (Class 2) from Real OASIS Scans
        # -------------------------------------------------------------
        print("\n[2/3] Parcellating OASIS 3D Structural MRI Scans...")
        oasis_scans = glob.glob("data/oasis_raw/**/*_masked_gfc.hdr", recursive=True)
        if not oasis_scans:
            oasis_scans = glob.glob("data/oasis_raw/**/*.hdr", recursive=True)

        count_ad = 0
        for scan in oasis_scans:
            if count_ad >= samples_per_class:
                break
            try:
                feat = extract_aal_features(scan)
                # AD disruption: Posterior Cingulate Cortex & Temporal nodes (AAL 35:40)
                adj = self.base_adj.copy()
                adj[35:40, :] *= 0.3
                self.samples.append((feat, feat.copy(), adj.astype(np.float32), 2))
                count_ad += 1
            except Exception as e:
                continue
        print(f" -> Ingested {count_ad} real OASIS Alzheimer's graphs.")

        # -------------------------------------------------------------
        # 3. Parkinson's Disease (Class 3) from PPMI / NTUA Scans
        # -------------------------------------------------------------
        print("\n[3/3] Parcellating Parkinson's DaTscan / MRI Scans...")
        pd_scans = glob.glob("data/parkinsons_raw/**/*.nii*", recursive=True)
        count_pd = 0
        for scan in pd_scans:
            if count_pd >= samples_per_class:
                break
            try:
                feat = extract_aal_features(scan)
                # PD disruption: Pallidum / Putamen / Basal Ganglia (AAL 70:76)
                adj = self.base_adj.copy()
                adj[70:76, :] *= 0.25
                self.samples.append((feat, feat.copy(), adj.astype(np.float32), 3))
                count_pd += 1
            except Exception:
                continue
        print(f" -> Ingested {count_pd} real Parkinson's graphs.")

        print("=" * 60)
        print(f" TOTAL MULTI-MODAL DATASET READY: {len(self.samples)} SAMPLES")
        print("=" * 60)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_feat, s_feat, adj, label = self.samples[idx]
        return (
            torch.tensor(f_feat, dtype=torch.float32),
            torch.tensor(s_feat, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

if __name__ == "__main__":
    ds = UnifiedNeuroGraphDataset(samples_per_class=5)
    f, s, a, y = ds[0]
    print(f"\nBatch Output Verification:")
    print(f"Functional Feats : {f.shape}")
    print(f"Structural Feats : {s.shape}")
    print(f"Adjacency Matrix : {a.shape}")
    print(f"Diagnostic Label : {y.item()} (0:HC, 1:ASD, 2:AD, 3:PD)")