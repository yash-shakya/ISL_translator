from pathlib import Path
import pandas as pd

ROOT = Path(r"ISL_CSLRT_Corpus")
ANNOTATIONS = ROOT / "corpus_csv_files"

input_file = ANNOTATIONS / "ISL_CSLRT_Corpus_frame_details.xlsx"
output_file = ANNOTATIONS / "frame_annotations_cleaned.csv"

df = pd.read_excel(input_file)

def resolve_frame_path(relative_path):
    if pd.isna(relative_path):
        return None

    # Normalize Windows separators and remove accidental whitespace
    relative_path = str(relative_path).strip().replace("/", "\\")
    parts = [part.strip() for part in relative_path.split("\\") if part.strip()]

    path = ROOT.joinpath(*parts[1:]) if parts[0] == ROOT.name else ROOT.joinpath(*parts)

    return str(path.resolve())

df["Resolved path"] = df["Frames path"].apply(resolve_frame_path)

# Verify the paths
df["Exists"] = df["Resolved path"].apply(
    lambda p: Path(p).is_file() if p else False
)

print("Total annotations:", len(df))
print("Valid frame paths:", df["Exists"].sum())
print("Still missing:", (~df["Exists"]).sum())

# Save cleaned annotations for future training
df.to_csv(output_file, index=False)

print("Saved cleaned annotations to:", output_file)

if (~df["Exists"]).any():
    print("\nRemaining missing paths:")
    print(df.loc[~df["Exists"], ["Sentence", "Frames path"]].head(20))