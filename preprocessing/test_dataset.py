
from pathlib import Path
import torch
from dataset import ISLSequenceDataset

ROOT = Path("ISL_CSLRT_Corpus")
PROCESSED = ROOT / "processed"

vocab_path = PROCESSED / "gloss_vocabulary.txt"

for split in ["train", "validation", "test"]:
    csv_path = PROCESSED / f"{split}.csv"

    dataset = ISLSequenceDataset(csv_path, vocab_path)
    video, target, video_id, sentence = dataset[0]

    print(f"\n{split.upper()}")
    print("Number of videos:", len(dataset))
    print("Video tensor shape:", tuple(video.shape))
    print("Target IDs:", target.tolist())
    print("Target length:", len(target))
    print("Video ID:", video_id)
    print("Sentence:", sentence)

    assert video.shape == (32, 3, 224, 224)
    assert target.ndim == 1
    assert len(target) > 0
