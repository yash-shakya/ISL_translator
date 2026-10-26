from pathlib import Path
import pandas as pd
import re

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------
# Dataset root
ROOT = Path(r"ISL_CSLRT_Corpus")

# Annotation files are in a subfolder
ANNOTATIONS = ROOT / "corpus_csv_files"

VIDEO_EXCEL = ANNOTATIONS / "ISL_CSLRT_Corpus_details.xlsx"
GLOSS_CSV = ANNOTATIONS / "ISL_Corpus_sign_glosses.csv"

OUTPUT_DIR = ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load annotations
# --------------------------------------------------
videos = pd.read_excel(VIDEO_EXCEL)
glosses = pd.read_csv(GLOSS_CSV)

print("Video annotations:", len(videos))
print("Gloss annotations:", len(glosses))

print("\nVideo columns:", videos.columns.tolist())
print("Gloss columns:", glosses.columns.tolist())

# --------------------------------------------------
# 3. Normalize sentence labels for matching
# --------------------------------------------------
def normalize_sentence(text):
    if pd.isna(text):
        return ""

    text = re.sub(r"\s+", " ", str(text).strip().lower())

    # Correct the known spacing inconsistency
    text = text.replace(
        "which collegeschool are you from",
        "which college school are you from"
    )

    return text

videos["sentence_key"] = videos["Sentences"].apply(
    normalize_sentence
)

glosses["sentence_key"] = glosses["Sentence"].apply(
    normalize_sentence
)

# --------------------------------------------------
# 4. Resolve video paths
# --------------------------------------------------
def resolve_video_path(value):
    if pd.isna(value):
        return None

    raw = str(value).strip()

    if not raw or raw == ">>>":
        return None

    # Normalize Windows and Unix path separators
    raw = raw.replace("/", "\\")

    path = Path(raw)

    # Handle paths that already contain the dataset root name
    parts = list(path.parts)
    root_name = ROOT.name.lower()

    root_index = next(
        (i for i, part in enumerate(parts)
         if part.lower() == root_name),
        None
    )

    if root_index is not None:
        path = ROOT.joinpath(*parts[root_index + 1:])
    elif not path.is_absolute():
        path = ROOT / path

    return path.resolve()

videos["resolved_path"] = videos["File location"].apply(
    resolve_video_path
)

videos["video_exists"] = videos["resolved_path"].apply(
    lambda p: p.is_file() if p is not None else False
)

# --------------------------------------------------
# 5. Match videos with gloss sequences
# --------------------------------------------------
gloss_map = (
    glosses.drop_duplicates("sentence_key")
    .set_index("sentence_key")["SIGN GLOSSES"]
    .to_dict()
)

videos["gloss_sequence"] = videos["sentence_key"].map(gloss_map)

videos["has_gloss"] = videos["gloss_sequence"].apply(
    lambda x: isinstance(x, str) and bool(x.strip())
)

# --------------------------------------------------
# 6. Create manifest
# --------------------------------------------------
manifest = videos[
    ["Sentences", "File location", "resolved_path",
     "video_exists", "gloss_sequence", "has_gloss"]
].copy()

manifest = manifest.rename(columns={
    "Sentences": "sentence",
    "File location": "original_path",
    "resolved_path": "video_path"
})

manifest["gloss_tokens"] = manifest["gloss_sequence"].apply(
    lambda x: " ".join(str(x).split()) if pd.notna(x) else ""
)

# --------------------------------------------------
# 7. Report validation results
# --------------------------------------------------
print("\n========== VALIDATION REPORT ==========")
print("Total video annotation rows:", len(manifest))
print("Existing video files:", manifest["video_exists"].sum())
print("Missing/invalid video paths:", (~manifest["video_exists"]).sum())
print("Rows with gloss sequences:", manifest["has_gloss"].sum())
print("Rows without gloss sequences:", (~manifest["has_gloss"]).sum())

print("\nMissing or invalid video paths:")
print(
    manifest.loc[
        ~manifest["video_exists"],
        ["sentence", "original_path"]
    ].to_string(index=False)
)

print("\nVideos without gloss mappings:")
print(
    manifest.loc[
        ~manifest["has_gloss"],
        ["sentence", "original_path"]
    ].to_string(index=False)
)

print("\nSaved manifest to:")
print(OUTPUT_DIR / "sentence_video_manifest.csv")

# --------------------------------------------------
# 8. Exclude invalid rows and rebuild the manifest
# --------------------------------------------------

# Keep videos only when the video exists and gloss is available
valid_manifest = manifest[
    manifest["video_exists"] &
    manifest["has_gloss"]
].copy()

# Save the final usable manifest
valid_manifest.to_csv(
    OUTPUT_DIR / "sentence_video_manifest.csv",
    index=False
)

# Save excluded records separately for inspection
excluded_manifest = manifest[
    ~(manifest["video_exists"] & manifest["has_gloss"])
].copy()

excluded_manifest.to_csv(
    OUTPUT_DIR / "excluded_video_annotations.csv",
    index=False
)

print("\n========== FINAL MANIFEST ==========")
print("Valid video records:", len(valid_manifest))
print("Excluded records:", len(excluded_manifest))
print(
    "Saved final manifest:",
    OUTPUT_DIR / "sentence_video_manifest.csv"
)
print(
    "Saved excluded records:",
    OUTPUT_DIR / "excluded_video_annotations.csv"
)