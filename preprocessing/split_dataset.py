
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("ISL_CSLRT_Corpus")
INPUT = ROOT / "processed" / "frame_sequence_manifest.csv"
OUTPUT_DIR = ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

df = pd.read_csv(INPUT)

# Only use successful, non-empty sequences with gloss targets
df = df[
    df["success"].astype(str).str.lower().eq("true")
    & df["gloss_sequence"].notna()
    & df["gloss_sequence"].astype(str).str.strip().ne("")
].copy()

# Each row is one video sequence
df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

print("Usable video sequences:", len(df))
print("Unique sentence labels:", df["sentence"].nunique())

# Stratify only when each class has enough samples for all splits.
counts = df["sentence"].value_counts()
can_stratify = (
    len(counts) > 1
    and counts.min() >= 3
    and len(df) >= 10
)

stratify_labels = df["sentence"] if can_stratify else None

if not can_stratify:
    print(
        "Warning: some sentence labels have fewer than 3 videos. "
        "Using a random split; inspect label coverage afterward."
    )

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=stratify_labels
)

# Split the remaining 20% equally into validation and test
temp_counts = temp_df["sentence"].value_counts()
can_stratify_temp = (
    len(temp_counts) > 1
    and temp_counts.min() >= 2
)

temp_stratify = temp_df["sentence"] if can_stratify_temp else None

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_stratify
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

combined = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

combined.to_csv(
    OUTPUT_DIR / "dataset_split.csv",
    index=False
)

for name, part in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    part.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} videos")

print("\nTotal split rows:", len(combined))
print("Unique videos:", combined["video_id"].nunique())

# Check overlap between splits
train_ids = set(train_df["video_id"])
val_ids = set(val_df["video_id"])
test_ids = set(test_df["video_id"])

print("Train/validation overlap:", len(train_ids & val_ids))
print("Train/test overlap:", len(train_ids & test_ids))
print("Validation/test overlap:", len(val_ids & test_ids))

# Check sentence labels not represented in training
train_labels = set(train_df["sentence"])
all_labels = set(df["sentence"])
print("Labels missing from training:", len(all_labels - train_labels))
