import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

# Input baseline scan
input_path = "test_OASIS_AD_HC (1).nii.gz"
output_dir = "test_scans"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_borderline_50pct.nii.gz")

if not os.path.exists(input_path):
    # Search for matching .nii.gz files if filename differs
    nii_files = [f for f in os.listdir('.') if f.endswith('.nii.gz') or f.endswith('.nii')]
    if nii_files:
        input_path = nii_files[0]
    else:
        raise FileNotFoundError("Could not find base OASIS scan in current directory.")

print(f"[*] Loading baseline volume: {input_path}")
base_img = nib.load(input_path)
data = base_img.get_fdata(dtype=np.float32)

# Normalize intensity range
v_min, v_max = data.min(), data.max()
normalized = (data - v_min) / (v_max - v_min + 1e-8)

# Target ROIs in typical AAL-116 coordinate space (e.g., temporal/hippocampal regions)
# Apply a balanced perturbation: moderate hippocampal attenuation mixed with baseline features
blurred = gaussian_filter(normalized, sigma=1.5)

# 50/50 transitional blend to yield an ambiguous boundary state
borderline_data = 0.50 * normalized + 0.50 * blurred

# Rescale to original intensity range
final_data = borderline_data * (v_max - v_min) + v_min

# Save as a new NIfTI image preserving affine and header metadata
borderline_img = nib.Nifti1Image(final_data, base_img.affine, base_img.header)
nib.save(borderline_img, output_path)

print(f"[✓] Created borderline scan: {output_path}")
print(f"    - Dimensions: {borderline_img.shape}")
print(f"    - Data Type: {borderline_img.get_data_dtype()}")
