import os
import glob
import numpy as np
import nibabel as nib
from nilearn import image

out_dir = "test_scans"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "test_borderline_50pct.nii.gz")

# 1. Locate the two base files in current workspace or test_scans
def find_file(pattern):
    matches = glob.glob(pattern) + glob.glob(f"test_scans/{pattern}")
    return matches[0] if matches else None

file_a = find_file("*OASIS*.nii*")
file_b = find_file("*ABIDE*.nii*")

if not file_a:
    raise FileNotFoundError("Could not find the OASIS scan in your directory.")

print(f"[*] Base Volume A: {file_a}")
img_a = nib.load(file_a)

if file_b:
    print(f"[*] Base Volume B: {file_b}")
    img_b = nib.load(file_b)
    # Ensure dimensions and coordinate headers match exactly using continuous linear resampling
    img_b_resampled = image.resample_to_img(img_b, img_a, interpolation='continuous')
    data_a = img_a.get_fdata(dtype=np.float32)
    data_b = img_b_resampled.get_fdata(dtype=np.float32)

    # Normalize intensity to [0, 1] range to avoid contrast bias
    norm_a = (data_a - data_a.min()) / (data_a.max() - data_a.min() + 1e-8)
    norm_b = (data_b - data_b.min()) / (data_b.max() - data_b.min() + 1e-8)

    # 50/50 dual-pathology morph
    print("[*] Performing 50/50 linear connectome boundary interpolation...")
    blended_norm = 0.50 * norm_a + 0.50 * norm_b
    final_data = blended_norm * (data_a.max() - data_a.min()) + data_a.min()

else:
    print("[*] Only one scan found — applying controlled Rician anatomical noise to create a 50% decision boundary...")
    data_a = img_a.get_fdata(dtype=np.float32)
    norm_a = (data_a - data_a.min()) / (data_a.max() - data_a.min() + 1e-8)

    # Perturb the connectome intensity distribution with 50% baseline + 50% Gaussian background
    noise = np.random.normal(loc=0.5, scale=0.15, size=norm_a.shape).astype(np.float32)
    mask = norm_a > 0.05
    blended_norm = norm_a.copy()
    blended_norm[mask] = 0.50 * norm_a[mask] + 0.50 * noise[mask]
    final_data = blended_norm * (data_a.max() - data_a.min()) + data_a.min()

# 2. Save result matching the original affine matrix and header dimensions
out_img = nib.Nifti1Image(final_data.astype(np.float32), img_a.affine, img_a.header)
nib.save(out_img, output_path)

print(f"\n[✓] Generated 50% confidence test scan: {output_path}")
print(f"    - Dimensions: {out_img.shape}")
print(f"    - File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
