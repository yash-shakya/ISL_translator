
from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path("ISL_CSLRT_Corpus")
INPUT = ROOT / "processed" / "dataset_split.csv"

df = pd.read_csv(INPUT)

def tokenize_glosses(value):
    if pd.isna(value):
        return []
    return str(value).strip().split()

df["tokens"] = df["gloss_sequence"].apply(tokenize_glosses)
df["target_length"] = df["tokens"].apply(len)

vocab = sorted({
    token
    for sequence in df["tokens"]
    for token in sequence
})

print("Total videos:", len(df))
print("Unique gloss tokens:", len(vocab))
print("Vocabulary:")
print(vocab)

print("\nGloss target length statistics:")
print(df["target_length"].describe())

print("\nShortest gloss sequences:")
print(
    df.nsmallest(10, "target_length")[
        ["sentence", "gloss_sequence", "target_length", "split"]
    ].to_string(index=False)
)

print("\nLongest gloss sequences:")
print(
    df.nlargest(10, "target_length")[
        ["sentence", "gloss_sequence", "target_length", "split"]
    ].to_string(index=False)
)

counts = Counter(token for seq in df["tokens"] for token in seq)
print("\nMost frequent glosses:")
print(counts.most_common(20))

vocab_path = ROOT / "processed" / "gloss_vocabulary.txt"
vocab_path.write_text("\n".join(vocab), encoding="utf-8")
print("\nSaved vocabulary to:", vocab_path)
