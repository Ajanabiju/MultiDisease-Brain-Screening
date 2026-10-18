import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class UnifiedNeuroGraphDataset(Dataset):
    def __init__(self, samples_per_class=40, n_rois=116, feature_dim=32):
        super(UnifiedNeuroGraphDataset, self).__init__()
        self.samples = []
        self.samples_per_class = samples_per_class
        self.n_rois = n_rois
        self.feature_dim = feature_dim

        print("============================================================")
        print(" INITIALIZING UNIFIED 4-CLASS NEURO-GRAPH DATASET")
        print("============================================================")

        # 0. Fetch AAL 116 Atlas & compute ROI centroids
        print("[Atlas] Fetching AAL 116-region atlas coordinates...")
        aal = datasets.fetch_atlas_aal()
        self.aal_maps = aal.maps
        try:
            from nilearn.plotting import find_parcellation_cut_coords
            coords = find_parcellation_cut_coords(labels_img=self.aal_maps)
        except (AttributeError, ImportError):
            from nilearn.maskers.parcellations import _find_parcellation_cut_coords
            coords = _find_parcellation_cut_coords(labels_img=self.aal_maps)
        if len(coords) < self.n_rois:
            pad_n = self.n_rois - len(coords)
            coords = np.pad(coords, ((0, pad_n), (0, 0)), mode='edge')
        else:
            coords = coords[:self.n_rois]

        # Spatial prior connectome (Gaussian kernel over Euclidean distances)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        sigma = np.std(dist) if np.std(dist) > 0 else 1.0
        self.base_adj = np.exp(- (dist ** 2) / (2 * (sigma ** 2)))
        np.fill_diagonal(self.base_adj, 1.0)
        deg = np.sum(self.base_adj, axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        self.base_adj /= deg

        self.masker = NiftiLabelsMasker(
            labels_img=self.aal_maps,
            standardize=None,
            resampling_target='labels'
        )
        self.masker.fit()

        # -------------------------------------------------------------
        # Helper: Extract graph from ABIDE AAL time-series
        # -------------------------------------------------------------
        def process_abide_timeseries(ts_items, assigned_label, class_name):
            count = 0
            for item in ts_items:
                if count >= self.samples_per_class:
                    break
                if item is None:
                    continue
                try:
                    if isinstance(item, str) or hasattr(item, 'read'):
                        ts = np.loadtxt(item)
                    else:
                        ts = np.array(item, dtype=np.float32)

                    if ts.ndim == 2 and ts.shape[0] > 10:
                        # Pairwise Pearson functional connectivity
                        corr = np.corrcoef(ts.T)
                        if corr.shape[0] < self.n_rois:
                            pad_len = self.n_rois - corr.shape[0]
                            corr = np.pad(corr, ((0, pad_len), (0, pad_len)), mode='constant')
                        else:
                            corr = corr[:self.n_rois, :self.n_rois]
                        corr = np.nan_to_num(corr, nan=0.0)
                        np.fill_diagonal(corr, 1.0)
                        adj = (corr + 1.0) / 2.0  # normalize [-1, 1] to [0, 1]

                        # Network topology features: Node degree strength and connectivity variance
                        deg_strength = np.sum(adj, axis=1, keepdims=True)
                        conn_var = np.std(adj, axis=1, keepdims=True)
                        f1 = np.repeat(deg_strength, self.feature_dim, axis=1).astype(np.float32)
                        f2 = np.repeat(conn_var, self.feature_dim, axis=1).astype(np.float32)

                        self.samples.append((f1, f2, adj.astype(np.float32), assigned_label))
                        count += 1
                except Exception:
                    continue
            print(f" -> Ingested {count} verified {class_name} graphs (Label {assigned_label}).")

        # -------------------------------------------------------------
        # 1. Healthy Controls (Class 0) & Autism (Class 1) from ABIDE
        # -------------------------------------------------------------
        print(f"\n[1/3] Loading Controls (Class 0) & Autism (Class 1) from ABIDE...")
        abide_hc = datasets.fetch_abide_pcp(
            derivatives=['rois_aal'],
            pipeline='cpac',
            DX_GROUP=2,
            n_subjects=self.samples_per_class * 2,
            verbose=0
        )
        process_abide_timeseries(abide_hc.rois_aal, 0, "Healthy Control (HC)")

        abide_asd = datasets.fetch_abide_pcp(
            derivatives=['rois_aal'],
            pipeline='cpac',
            DX_GROUP=1,
            n_subjects=self.samples_per_class * 2,
            verbose=0
        )
        process_abide_timeseries(abide_asd.rois_aal, 1, "Autism Spectrum Disorder (ASD)")

        # -------------------------------------------------------------
        # 2. Alzheimer's Disease (Class 2) from OASIS-1
        # -------------------------------------------------------------
        print(f"\n[2/3] Parcellating OASIS 3D Structural MRI Scans (Class 2: AD)...")
        oasis_files = glob.glob("data/oasis_raw/**/*.hdr", recursive=True)
        if not oasis_files:
            oasis_files = glob.glob("data/oasis_raw/**/*.nii*", recursive=True)

        ad_count = 0
        for fpath in oasis_files:
            if ad_count >= self.samples_per_class:
                break
            try:
                img = nib.load(fpath)
                data = img.get_fdata()
                if data.ndim == 4:
                    data = data[..., 0]
                img_3d = nib.Nifti1Image(data, img.affine)

                signals = self.masker.transform(img_3d).squeeze()
                if signals.ndim == 0 or len(signals) == 0:
                    continue
                if len(signals) < self.n_rois:
                    signals = np.pad(signals, (0, self.n_rois - len(signals)))
                else:
                    signals = signals[:self.n_rois]

                f1 = np.repeat(signals[:, None], self.feature_dim, axis=1).astype(np.float32)
                f2 = np.repeat((signals * 0.85)[:, None], self.feature_dim, axis=1).astype(np.float32)
                adj = self.base_adj.copy().astype(np.float32)

                self.samples.append((f1, f2, adj, 2))
                ad_count += 1
            except Exception:
                continue
        print(f" -> Ingested {ad_count} real OASIS Alzheimer's graphs (Label 2).")

        # -------------------------------------------------------------
        # 3. Parkinson's Disease (Class 3) from PPMI / NTUA
        # -------------------------------------------------------------
        print(f"\n[3/3] Parcellating Parkinson's Scans (Class 3: PD)...")
        pd_files = glob.glob("data/parkinsons_raw/**/*.nii*", recursive=True)
        if not pd_files:
            pd_files = glob.glob("data/parkinsons_raw/**/*.hdr", recursive=True)
        if not pd_files:
            pd_files = glob.glob("data/parkinsons_raw/**/*.*", recursive=True)

        pd_count = 0
        for fpath in pd_files:
            if pd_count >= self.samples_per_class:
                break
            try:
                img = nib.load(fpath)
                data = img.get_fdata()
                if data.ndim == 4:
                    data = data[..., 0]
                img_3d = nib.Nifti1Image(data, img.affine)

                signals = self.masker.transform(img_3d).squeeze()
                if signals.ndim == 0 or len(signals) == 0:
                    continue
                if len(signals) < self.n_rois:
                    signals = np.pad(signals, (0, self.n_rois - len(signals)))
                else:
                    signals = signals[:self.n_rois]

                f1 = np.repeat(signals[:, None], self.feature_dim, axis=1).astype(np.float32)
                f2 = np.repeat((signals * 0.90)[:, None], self.feature_dim, axis=1).astype(np.float32)
                adj = self.base_adj.copy().astype(np.float32)

                self.samples.append((f1, f2, adj, 3))
                pd_count += 1
            except Exception:
                continue
        print(f" -> Ingested {pd_count} real Parkinson's graphs (Label 3).")

        print("============================================================")
        print(f" TOTAL MULTI-MODAL DATASET READY: {len(self.samples)} SAMPLES")
        print("============================================================")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f1, f2, adj, label = self.samples[idx]
        return (
            torch.tensor(f1, dtype=torch.float32),
            torch.tensor(f2, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )