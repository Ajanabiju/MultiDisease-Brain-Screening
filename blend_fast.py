import os
import glob
import numpy as np
import nibabel as nib
from nilearn import image

out_dir = "test_scans"
os.makedirs(out_dir, exist_ok=True)
output_path = os.path.join(out_dir, "test_borderline_50pct.nii.gz")

def find_file(pattern):
    matches = glob.glob(pattern) + glob.glob(f"test_scans/{pattern}")
    return matches[0] if matches else None

file_a = find_file("*OASIS*.nii*")
file_b = find_file("*ABIDE*.nii*")

if not file_a:
    raise FileNotFoundError("Could not locate the baseline OASIS file.")

print(f"[*] Loading Target Geometry (Volume A): {file_a}")
img_a = nib.load(file_a)

if file_b:
    print(f"[*] Loading Secondary Scan (Volume B): {file_b}")
    img_b = nib.load(file_b)

    # 1. Collapse 4D to 3D first to avoid exhausting RAM
    if len(img_b.shape) == 4:
        print(f"    - Detected 4D input {img_b.shape}. Reducing to 3D mean frame...")
        img_b_3d = image.mean_img(img_b)
    else:
        img_b_3d = img_b

    # 2. Resample the lightweight 3D frame to match Volume A's exact affine and grid
    print("    - Resampling Volume B to Volume A grid...")
    img_b_resampled = image.resample_to_img(img_b_3d, img_a, interpolation='continuous')

    data_a = img_a.get_fdata(dtype=np.float32)
    data_b = img_b_resampled.get_fdata(dtype=np.float32)

    # 3. Min-Max normalization for fair 50/50 blending
    a_min, a_max = data_a.min(), data_a.max()
    b_min, b_max = data_b.min(), data_b.max()

    norm_a = (data_a - a_min) / (a_max - a_min + 1e-8)
    norm_b = (data_b - b_min) / (b_max - b_min + 1e-8)

    print("[*] Performing 50/50 connectome interpolation...")
    blended = 0.50 * norm_a + 0.50 * norm_b
    final_data = (blended * (a_max - a_min) + a_min).astype(np.float32)

else:
    print("[*] Creating 50% boundary via controlled feature attenuation on Volume A...")
    data_a = img_a.get_fdata(dtype=np.float32)
    a_min, a_max = data_a.min(), data_a.max()
    norm_a = (data_a - a_min) / (a_max - a_min + 1e-8)

    # Perturb intensities to create an ambiguous decision boundary
    noise = np.random.normal(loc=0.5, scale=0.12, size=norm_a.shape).astype(np.float32)
    mask = norm_a > 0.05
    norm_a[mask] = 0.50 * norm_a[mask] + 0.50 * noise[mask]
    final_data = (norm_a * (a_max - a_min) + a_min).astype(np.float32)

# Save output preserving original affine header
out_img = nib.Nifti1Image(final_data, img_a.affine, img_a.header)
nib.save(out_img, output_path)

file_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\n[✓] Successfully created: {output_path}")
print(f"    - Dimensions: {out_img.shape}")
print(f"    - Output Size: {file_mb:.2f} MB")
