
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ISLRecognizer(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size=256,
        pretrained=True,
        freeze_cnn=True
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # ResNet-18 feature vector: 512 dimensions
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    @torch.no_grad()
    def extract_features(self, videos):
        # videos: [B, T, C, H, W]
        batch_size, time_steps, channels, height, width = videos.shape

        frames = videos.reshape(
            batch_size * time_steps,
            channels, height, width
        )

        features = self.cnn(frames).flatten(1)

        return features.reshape(batch_size, time_steps, 512)

    def forward(self, videos):
        # videos: [batch, time, channels, height, width]
        batch_size, time_steps, channels, height, width = (
            videos.shape
        )

        frames = videos.reshape(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        features = self.cnn(frames)
        features = features.flatten(1)

        features = features.reshape(
            batch_size,
            time_steps,
            512
        )

        temporal_features, _ = self.lstm(features)

        # [batch, time, number of classes]
        logits = self.classifier(temporal_features)

        return logits
