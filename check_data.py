import os
import glob
import nibabel as nib

print("=== Checking OASIS Structural Scans ===")
oasis_scans = glob.glob("data/oasis_raw/**/PROCESSED/MPRAGE/T88_111/*_masked_gfc.hdr", recursive=True)
if not oasis_scans:
    # Fallback search in case folder structure is flatter
    oasis_scans = glob.glob("data/oasis_raw/**/*_masked_gfc.hdr", recursive=True)

print(f"Total OASIS scans found: {len(oasis_scans)}")
if oasis_scans:
    sample_img = nib.load(oasis_scans[0])
    print(f"Sample scan shape: {sample_img.shape} (Ready for 116 AAL Parcellation)")

print("\n=== Checking Parkinson's Data ===")
pd_files = glob.glob("data/parkinsons_raw/**/*.*", recursive=True)
print(f"Total Parkinson's files found: {len(pd_files)}")
if pd_files:
    print(f"Sample file: {os.path.basename(pd_files[0])}")
    