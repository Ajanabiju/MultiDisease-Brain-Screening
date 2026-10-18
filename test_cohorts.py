import os
import glob
from nilearn import datasets
import nibabel as nib

print("=" * 60)
print(" VERIFYING MULTI-DISORDER DATASETS")
print("=" * 60)

# 1. Autism (ABIDE)
print("\n[1/3] Checking Autism (ABIDE)...")
try:
    abide = datasets.fetch_abide_pcp(derivatives=['rois_aal'], n_subjects=5)
    print(f" -> ABIDE OK: Found {len(abide.rois_aal)} cached subjects.")
except Exception as e:
    print(f" -> ABIDE Error: {e}")

# 2. Alzheimer's (OASIS-1)
print("\n[2/3] Checking Alzheimer's (OASIS)...")
oasis_scans = glob.glob("data/oasis_raw/**/*_masked_gfc.hdr", recursive=True)
if not oasis_scans:
    oasis_scans = glob.glob("data/oasis_raw/**/*.img", recursive=True)
print(f" -> OASIS Scans found: {len(oasis_scans)}")
if oasis_scans:
    img = nib.load(oasis_scans[0])
    print(f" -> Sample Scan: {os.path.basename(oasis_scans[0])} | Shape: {img.shape}")

# 3. Parkinson's (NTUA / Kaggle)
print("\n[3/3] Checking Parkinson's Data...")
pd_files = glob.glob("data/parkinsons_raw/**/*.*", recursive=True)
print(f" -> Parkinson's files found: {len(pd_files)}")
if pd_files:
    # Print sample file paths to inspect folder naming
    samples = [f for f in pd_files if f.endswith(('.png', '.dcm', '.jpg', '.nii', '.nii.gz'))]
    if samples:
        print(f" -> Sample Image File: {samples[0]}")
print("=" * 60)python -c "
import pandas as pd
df = pd.read_excel('data/oasis_cross-sectional.csv', engine='openpyxl')
df.to_csv('data/oasis_cross-sectional.csv', index=False, encoding='utf-8')
print('Successfully converted to clean CSV!')
print(df[['ID', 'M/F', 'Age', 'CDR']].dropna().head(10))
"