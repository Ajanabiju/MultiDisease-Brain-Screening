import os
import shutil
from nilearn import datasets

output_dir = os.path.join(os.path.dirname(__file__), "test_scans")
os.makedirs(output_dir, exist_ok=True)

print("--- DOWNLOADING TEST SCANS ---")

# 1. OASIS Structural Scan (AD / HC)
print("\n[1/2] Fetching OASIS structural scan...")
try:
    oasis = datasets.fetch_oasis_vbm(n_subjects=1)
    if oasis.gray_matter_maps:
        src = oasis.gray_matter_maps[0]
        dest = os.path.join(output_dir, "test_OASIS_AD_HC.nii.gz")
        shutil.copyfile(src, dest)
        print(f" -> Saved OASIS sample: {dest}")
except Exception as e:
    print(f"Failed OASIS download: {e}")

# 2. ABIDE Functional Scan (ASD)
print("\n[2/2] Fetching ABIDE functional scan...")
try:
    abide = datasets.fetch_abide_pcp(n_subjects=1, derivatives=['func_preproc'], quality_checked=True)
    if abide.func_preproc:
        src = abide.func_preproc[0]
        dest = os.path.join(output_dir, "test_ABIDE_ASD.nii.gz")
        shutil.copyfile(src, dest)
        print(f" -> Saved ABIDE sample: {dest}")
except Exception as e:
    print(f"Failed ABIDE download: {e}")

print("\nDone! Downloaded files are in the 'test_scans' directory.")
