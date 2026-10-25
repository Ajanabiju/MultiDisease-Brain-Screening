import os
import sys
import numpy as np
import nibabel as nib
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATASET_PATH = os.path.join(BASE_DIR, "test_dataset.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "exact_training_scans")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from app import get_aal_atlas

DISEASE_NAMES = {
    0: "Healthy_Control",
    1: "Autism_Spectrum_Disorder",
    2: "Alzheimers_Disease",
    3: "Parkinsons_Disease"
}

def export_test_split_to_nii():
    if not os.path.exists(TEST_DATASET_PATH):
        print(f"[!] '{TEST_DATASET_PATH}' not found. Please run train_multidisease_production.py first.")
        return

    print(f"[*] Loading exact test split from: {TEST_DATASET_PATH}")
    checkpoint = torch.load(TEST_DATASET_PATH, map_location="cpu")
    x_test = checkpoint["x"].numpy()      # Shape: (N, 116, 64)
    y_test = checkpoint["y"].numpy()      # Shape: (N,)

    # Load AAL Atlas to get spatial coordinate grid
    atlas = get_aal_atlas()
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    header = atlas_img.header

    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0][:116]

    print("[*] Reconstructing 3D NIfTI volumes from exact test tensors...")
    
    saved_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Export the first 2 verified subjects of each disease from the test split
    for i in range(len(y_test)):
        cls_idx = int(y_test[i])
        if saved_counts[cls_idx] >= 2:
            continue

        sample_features = x_test[i]  # (116, 64)
        
        # Take the base regional amplitude across the 116 parcels
        regional_intensities = sample_features[:, 0]
        # Shift to positive MRI voxel intensities (~100 baseline)
        voxel_values = (regional_intensities * 25.0) + 100.0

        # Paint values into corresponding AAL parcels in 3D brain space
        vol_3d = np.zeros_like(atlas_data, dtype=np.float32)
        for p_idx, parcel_id in enumerate(unique_parcels):
            mask = (atlas_data == parcel_id)
            vol_3d[mask] = voxel_values[p_idx]

        disease_label = DISEASE_NAMES[cls_idx]
        filename = f"exact_test_sample_{disease_label}_idx{i}.nii.gz"
        save_path = os.path.join(OUTPUT_DIR, filename)

        nii_obj = nib.Nifti1Image(vol_3d, affine, header)
        nib.save(nii_obj, save_path)

        saved_counts[cls_idx] += 1
        print(f"  ✓ Saved [{disease_label}]: {filename}")

        if all(count >= 2 for count in saved_counts.values()):
            break

    print(f"\n[SUCCESS] Exact evaluation samples are ready in:\n'{OUTPUT_DIR}'")

if __name__ == "__main__":
    export_test_split_to_nii()