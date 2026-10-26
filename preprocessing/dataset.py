
from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ISLSequenceDataset(Dataset):
    def __init__(self, csv_path, vocab_path, image_size=224):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        vocab_file = Path(vocab_path)
        tokens = [
            line.strip()
            for line in vocab_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # CTC blank is reserved as class index 0
        self.vocab = {token: i + 1 for i, token in enumerate(tokens)}
        self.blank_id = 0

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frame_paths = json.loads(row["frame_paths"])
        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                frames.append(self.transform(image))

        # Shape: [time, channels, height, width]
        video = torch.stack(frames)

        glosses = str(row["gloss_sequence"]).strip().split()

        unknown = [token for token in glosses if token not in self.vocab]
        if unknown:
            raise ValueError(f"Unknown gloss tokens: {unknown}")

        target = torch.tensor(
            [self.vocab[token] for token in glosses],
            dtype=torch.long
        )

        return video, target, row["video_id"], row["sentence"]
