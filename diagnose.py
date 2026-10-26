
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn


CHECKPOINT_PATH = Path("checkpoints/isl_baseline_test.pt")
CACHE_PATH = Path("ISL_CSLRT_Corpus/processed/feature_cache.pt")


class SequenceHead(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, features):
        temporal_features, _ = self.lstm(features)
        return self.classifier(temporal_features)


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        CACHE_PATH, map_location="cpu", weights_only=False
    )

    vocab = checkpoint["vocab"]
    id_to_gloss = {int(i): token for token, i in vocab.items()}
    num_classes = max(id_to_gloss) + 1

    model = SequenceHead(num_classes, hidden_size=128)
    model.load_state_dict(checkpoint["head_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Number of output classes:", num_classes)
    print("Blank ID: 0")
    print("Vocabulary size:", len(vocab))

    for split in ("train", "validation"):
        if split not in cache:
            continue

        counts = Counter()
        total_frames = 0
        blank_probability_sum = 0.0
        examples_shown = 0

        with torch.no_grad():
            for features, targets, target_lengths in cache[split]:
                logits = model(features)
                probabilities = logits.softmax(dim=-1)
                predictions = logits.argmax(dim=-1)

                counts.update(predictions.reshape(-1).tolist())
                total_frames += predictions.numel()
                blank_probability_sum += (
                    probabilities[..., 0].sum().item()
                )

                if examples_shown < 3:
                    for row in predictions:
                        ids = row.tolist()
                        collapsed = []
                        previous = None

                        for token_id in ids:
                            if token_id != previous and token_id != 0:
                                collapsed.append(token_id)
                            previous = token_id

                        glosses = [
                            id_to_gloss[i]
                            for i in collapsed
                            if i in id_to_gloss
                        ]

                        print(
                            f"{split} decoded sample:",
                            " ".join(glosses) or "<EMPTY>",
                        )
                        examples_shown += 1

                    # Show only a few decoded sequences.
                    # Continue through all batches for aggregate stats.

        blank_frames = counts.get(0, 0)
        blank_probability = blank_probability_sum / total_frames

        print(f"\n--- {split.upper()} DIAGNOSTICS ---")
        print("Total frames:", total_frames)
        print(
            f"Frames predicted blank: "
            f"{blank_frames / total_frames:.2%}"
        )
        print(
            f"Mean probability assigned to blank: "
            f"{blank_probability:.2%}"
        )

        print("Most common predicted class IDs:")
        for class_id, count in counts.most_common(10):
            gloss = "<BLANK>" if class_id == 0 else id_to_gloss.get(
                class_id, "<UNKNOWN>"
            )
            print(
                f"  ID {class_id:3d} | {gloss:20s} | "
                f"{count / total_frames:.2%}"
            )


if __name__ == "__main__":
    main()
