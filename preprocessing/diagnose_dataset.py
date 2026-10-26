
from pathlib import Path
import re
import pandas as pd

ROOT = Path(r"ISL_CSLRT_Corpus")

ANNOTATIONS = Path(r"ISL_CSLRT_Corpus/corpus_csv_files")

OUT = ROOT / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_label(value):
    """Normalize whitespace and capitalization for comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def resolve_path(value):
    """Resolve paths from the annotation spreadsheets."""
    value = str(value).strip().replace("\\", "/")
    parts = value.split("/")

    if parts and parts[0].lower() == "isl_cslrt_corpus":
        parts = parts[1:]

    return ROOT.joinpath(*parts)


frames = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus_frame_details.xlsx"
)
words = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus_word_details.xlsx"
)
videos = pd.read_excel(
    ANNOTATIONS / "ISL_CSLRT_Corpus details.xlsx"
)
glosses = pd.read_csv(
    ANNOTATIONS / "ISL Corpus sign glosses.csv"
)

# -------------------------------------------------
# 1. Inspect all missing frame paths
# -------------------------------------------------
frames["resolved_path"] = frames["Frames path"].map(resolve_path)
frames["exists"] = frames["resolved_path"].map(
    lambda p: p.is_file()
)

missing = frames.loc[
    ~frames["exists"],
    ["Sentence", "Frames path", "resolved_path"]
]

missing.to_csv(OUT / "missing_frames.csv", index=False)

print("\n--- MISSING FRAME PATHS ---")
print("Count:", len(missing))
print(missing.to_string(index=False))

# -------------------------------------------------
# 2. Compare normalized sentence labels
# -------------------------------------------------
frames["label_norm"] = frames["Sentence"].map(normalize_label)
videos["label_norm"] = videos["Sentences"].map(normalize_label)
glosses["label_norm"] = glosses["Sentence"].map(normalize_label)

frame_labels = set(frames["label_norm"]) - {""}
video_labels = set(videos["label_norm"]) - {""}
gloss_labels = set(glosses["label_norm"]) - {""}

comparisons = {
    "Frames missing from gloss CSV": frame_labels - gloss_labels,
    "Gloss labels missing from frames": gloss_labels - frame_labels,
    "Videos missing from gloss CSV": video_labels - gloss_labels,
    "Gloss labels missing from videos": gloss_labels - video_labels,
    "Frames missing from video annotations": frame_labels - video_labels,
    "Videos missing from frame annotations": video_labels - frame_labels,
}

print("\n--- SENTENCE LABEL DIFFERENCES ---")
for description, labels in comparisons.items():
    print(f"\n{description}: {len(labels)}")
    for label in sorted(labels):
        print("  ", repr(label))

# -------------------------------------------------
# 3. Count frames and videos per sentence
# -------------------------------------------------
frame_counts = (
    frames.groupby("label_norm")
    .size()
    .rename("frame_records")
)

video_counts = (
    videos[videos["File location"].notna()]
    .groupby("label_norm")
    .size()
    .rename("video_records")
)

summary = pd.concat(
    [frame_counts, video_counts], axis=1
).fillna(0).astype(int)

summary.to_csv(OUT / "sentence_counts.csv")

print("\n--- SENTENCE COUNTS ---")
print(summary.sort_values("frame_records").to_string())

# -------------------------------------------------
# 4. Inspect invalid video entries
# -------------------------------------------------
videos["resolved_path"] = videos["File location"].map(
    lambda x: resolve_path(x) if pd.notna(x) else None
)

invalid_videos = videos[
    videos["resolved_path"].isna()
    | ~videos["resolved_path"].map(
        lambda p: p.is_file() if p is not None else False
    )
]

invalid_videos.to_csv(
    OUT / "invalid_videos.csv", index=False
)

print("\n--- INVALID VIDEO ENTRIES ---")
print(invalid_videos.to_string(index=False))

# -------------------------------------------------
# 5. Save normalized annotation copies
# -------------------------------------------------
frames.to_csv(OUT / "frame_annotations_checked.csv", index=False)
videos.to_csv(OUT / "video_annotations_checked.csv", index=False)
glosses.to_csv(OUT / "gloss_annotations_checked.csv", index=False)

print(f"\nDiagnostic files saved to: {OUT}")
