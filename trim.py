import os
import nibabel as nib

target_dir = os.path.join(os.getcwd(), "clinical_test_data")

for root, _, files in os.walk(target_dir):
    for f in files:
        if (f.endswith(".nii") or f.endswith(".nii.gz")) and not f.startswith("trimmed_"):
            full_path = os.path.join(root, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            
            # If the file is larger than 60 MB, it's an uncompressed or full 4D series
            if size_mb > 60:
                print(f"[*] Trimming large 4D scan: {f} ({size_mb:.1f} MB)...")
                try:
                    img = nib.load(full_path)
                    data = img.get_fdata()
                    
                    if data.ndim == 4:
                        # Keep only the first 64 frames required by the model
                        trimmed_data = data[..., :64]
                        trimmed_img = nib.Nifti1Image(trimmed_data, img.affine, img.header)
                        
                        out_path = os.path.join(root, f"trimmed_{os.path.splitext(f)[0]}.nii.gz")
                        if out_path.endswith(".nii.nii.gz"):
                            out_path = out_path.replace(".nii.nii.gz", ".nii.gz")
                            
                        nib.save(trimmed_img, out_path)
                        print(f"    ✓ Created lightweight scan ({os.path.getsize(out_path)/(1024*1024):.1f} MB): {os.path.basename(out_path)}")
                except Exception as e:
                    print(f"    [!] Skip {f}: {e}")

print("\n[COMPLETE] You can now upload any of the 'trimmed_' files without hitting size limits.")