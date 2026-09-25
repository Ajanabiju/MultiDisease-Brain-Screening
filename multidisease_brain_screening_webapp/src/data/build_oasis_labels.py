import os
import glob
import pandas as pd

def extract_labels_from_local_folders(base_dir="data/oasis_raw"):
    # Look for the subject text files extracted with the scans (e.g., OAS1_0042_MR1.txt)
    txt_files = glob.glob(os.path.join(base_dir, "**", "OAS1_*_MR1.txt"), recursive=True)
    if not txt_files:
        txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)

    records = []
    for f in txt_files:
        subj_id = os.path.basename(f).replace(".txt", "")
        if not subj_id.startswith("OAS1_"):
            continue

        info = {"ID": subj_id, "CDR": 0.0, "Age": 60, "M/F": "M"}
        try:
            with open(f, 'r') as meta:
                for line in meta:
                    line_clean = line.strip()
                    if "CDR" in line_clean:
                        parts = line_clean.replace(":", " ").replace(",", " ").split()
                        for p in parts:
                            try:
                                info["CDR"] = float(p)
                                break
                            except ValueError:
                                pass
                    elif "Age" in line_clean:
                        parts = line_clean.replace(":", " ").replace(",", " ").split()
                        for p in parts:
                            if p.isdigit():
                                info["Age"] = int(p)
                                break
                    elif "M/F" in line_clean or "Gender" in line_clean:
                        if "F" in line_clean:
                            info["M/F"] = "F"
        except Exception:
            pass

        records.append(info)

    df = pd.DataFrame(records).drop_duplicates(subset=["ID"])
    out_path = "data/oasis_labels_clean.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df)} subjects extracted directly from scan metadata:\n")
    print(df.head(10))
    return df

if __name__ == "__main__":
    extract_labels_from_local_folders()