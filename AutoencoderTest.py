import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SegmentEmbed(nn.Module):
    """Embedding layer for segments of sequence data"""
    def __init__(self, num_channels, embed_dim, segment_length):
        super().__init__()
        self.segment_length = segment_length
        self.proj = nn.Linear(num_channels * segment_length, embed_dim)

    def forward(self, x):
        # x shape: (batch, num_segments, segment_length, num_channels)
        batch_size, num_channels, seq_length = x.shape
        x = x.reshape(batch_size, num_segments, -1)  # Flatten each segment
        return self.proj(x)

class MaskedAutoencoderSeqSegmented(nn.Module):
    """ Masked Autoencoder with Transformer backbone for segmented sequence data """
    def __init__(self, seq_length, num_channels, segment_length, embed_dim=256, depth=3, num_heads=16,
                 decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_segments = seq_length // segment_length
        self.segment_length = segment_length

        # --------------------------------------------------------------------------
        # MAE encoder
        self.segment_embed = SegmentEmbed(num_channels, embed_dim, segment_length)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_segments, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_segments, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads, dim_feedforward=int(decoder_embed_dim*mlp_ratio), batch_first=True)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_channels * segment_length)  # Output back to original segment size
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        torch.nn.init.xavier_uniform_(self.segment_embed.proj.weight)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (batch, sequence_length, num_channels)
        x = x.reshape(-1, self.num_segments, self.segment_length, x.shape[-1])  # Divide into segments
        x = self.segment_embed(x)  # Embed each segment
        x = x + self.pos_embed  # Add positional embedding

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Decoder (simplified)
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # Output projection to original feature size

        return x

# Example usage:
seq_length = 4000
num_channels = 22
segment_length = 50


model = MaskedAutoencoderSeqSegmented(seq_length, num_channels, segment_length)
input_tensor = torch.randn(64, 22, 4000)
output = model(input_tensor)