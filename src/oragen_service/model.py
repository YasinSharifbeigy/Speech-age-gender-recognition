"""Audio architecture adapted from MIT-licensed markitantov/chimera_ml.

Upstream commit: 2f5429aa8ad6d0d7ec0a7acb9ed5fd7e337def8d.
See ORAGEN-LICENSE.txt. Layer names intentionally match the published checkpoint.
"""
import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .layers import AGenderClassificationHead, StatPoolLayer, TransformerLayer


class OragenAudioModel(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        size = config.hidden_size
        self.transformer_block1 = TransformerLayer(size, 4, 0.1, True)
        self.transformer_block2 = TransformerLayer(size, 4, 0.1, True)
        self.stp = StatPoolLayer(1)
        self.fc1 = nn.Linear(size * 2, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.6)
        self.cl_head = AGenderClassificationHead(256, 3)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.wav2vec2(waveform)[0]
        x = self.relu(self.transformer_block1(x, x, x))
        x = self.relu(self.transformer_block2(x, x, x))
        return self.cl_head(self.dp(self.relu(self.fc1(self.stp(x)))))
