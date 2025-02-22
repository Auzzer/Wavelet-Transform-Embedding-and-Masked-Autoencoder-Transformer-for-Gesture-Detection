import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_classes, segment_length, series_length, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512,
                 dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.segment_length = segment_length
        self.num_segments = series_length // segment_length
        self.d_model = d_model

        self.input_projection = nn.Linear(num_features * segment_length, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.positional_encoding = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [batch_size, channels, length]
        batch_size, num_channels, total_length = x.shape
        x = x.view(batch_size, num_channels, self.num_segments, self.segment_length)
        x = x.permute(2, 0, 1, 3).reshape(self.num_segments, batch_size,
                                          -1)  # [num_segments, batch_size, num_channels * segment_length]
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling over the sequence dimension
        x = self.fc_out(x)
        return x




