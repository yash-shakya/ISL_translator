
from pathlib import Path
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path("ISL_CSLRT_Corpus")
PROCESSED = ROOT / "processed"
CHECKPOINTS = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 2
HIDDEN_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 3e-4

DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, csv_path, vocab):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.vocab = vocab

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        paths = json.loads(row["frame_paths"])

        frames = []
        for path in paths:
            with Image.open(path) as im:
                frames.append(self.transform(im.convert("RGB")))

        video = torch.stack(frames)
        tokens = str(row["gloss_sequence"]).split()
        target = torch.tensor(
            [self.vocab[token] for token in tokens],
            dtype=torch.long
        )

        return video, target


def collate_batch(batch):
    videos = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    target_lengths = torch.tensor(
        [len(t) for t in targets], dtype=torch.long
    )
    concatenated_targets = torch.cat(targets)
    return videos, concatenated_targets, target_lengths


# --------------------------------------------------
# Frozen pretrained CNN
# --------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(net.children())[:-1])
        self.cnn.eval()

        for p in self.cnn.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, videos):
        b, t, c, h, w = videos.shape
        x = videos.reshape(b * t, c, h, w)
        x = self.cnn(x).flatten(1)
        return x.reshape(b, t, 512)


# --------------------------------------------------
# BiLSTM + classifier
# --------------------------------------------------
class SequenceHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(
            HIDDEN_SIZE * 2, num_classes
        )

    def forward(self, features):
        x, _ = self.lstm(features)
        return self.classifier(x)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    vocab_tokens = [
        x.strip()
        for x in (PROCESSED / "gloss_vocabulary.txt")
        .read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    # CTC blank is class 0; gloss IDs start at 1
    vocab = {token: i + 1 for i, token in enumerate(vocab_tokens)}
    num_classes = len(vocab) + 1

    train_set = FeatureDataset(PROCESSED / "train.csv", vocab)
    val_set = FeatureDataset(PROCESSED / "validation.csv", vocab)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_batch
    )

    encoder = CNNEncoder().to(DEVICE)
    head = SequenceHead(num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(
    head.parameters(),
    lr=LEARNING_RATE
    )

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=False
    )

    # ResNet is frozen; cache its features to avoid repeating CNN work.
    cache_path = PROCESSED / "feature_cache.pt"

    if cache_path.exists():
        print("Loading cached CNN features...")
        cache = torch.load(cache_path, map_location="cpu",
                           weights_only=False)
    else:
        print("Extracting CNN features; this may take a while...")
        cache = {"train": [], "validation": []}

        for name, loader in [
            ("train", train_loader),
            ("validation", val_loader)
        ]:
            for videos, targets, target_lengths in loader:
                features = encoder(videos.to(DEVICE))
                cache[name].append((
                    features.cpu(),
                    targets.cpu(),
                    target_lengths.cpu()
                ))

        torch.save(cache, cache_path)
        print("Saved feature cache:", cache_path)

    def run_epoch(name, training):
        head.train(training)
        total_loss = 0.0
        batches = 0
        start = time.time()

        for features, targets, target_lengths in cache[name]:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = head(features)
                log_probs = logits.log_softmax(2).transpose(0, 1)

                batch_size, time_steps = features.shape[:2]
                input_lengths = torch.full(
                    (batch_size,), time_steps, dtype=torch.long
                )

                loss = criterion(
                    log_probs, targets,
                    input_lengths, target_lengths
                )

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        head.parameters(), 5.0
                    )
                    optimizer.step()

            total_loss += loss.item()
            batches += 1

        elapsed = time.time() - start
        return total_loss / max(batches, 1), elapsed

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_time = run_epoch("train", True)
        val_loss, val_time = run_epoch("validation", False)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"train: {train_time:.1f}s | "
            f"val: {val_time:.1f}s"
        )

        torch.save({
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, CHECKPOINTS / "isl_baseline_test.pt")

    print("\n5-epoch speed test completed.")


if __name__ == "__main__":
    main()
