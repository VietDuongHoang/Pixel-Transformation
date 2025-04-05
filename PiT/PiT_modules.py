import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patchs = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, "Hình ảnh đầu vào sai kích thước"
        x = self.proj(x).flatten(2).permute(0, 2, 1)  # (B, C, H, W) -> (B, num_patches, embed_dim)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_patchs, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patchs, embed_dim))  # Khởi tạo bằng 0 để ổn định hơn

    def forward(self, x):
        return x + self.pos_embed
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dims, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(embed_dim, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, embed_dim)
        self.activation = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.fc2(self.activation(self.fc1(x)))
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dims, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dims, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class CLSEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Khởi tạo bằng 0 thay vì random

    def forward(self, x):
        N, _, E = x.shape
        cls_tokens = self.cls_token.expand(N, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)
    
class PiT(nn.Module):
    def __init__(self, img_size=224, patch_size=1, embed_dim=768, num_classes=1000,
                 num_layers=12, num_heads=12, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.num_patchs = self.patch_embed.num_patchs
        self.pos_embed = PositionalEncoding(self.num_patchs + 1, embed_dim)
        self.cls_embed = CLSEmbedding(embed_dim)
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, embed_dim * 4, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_embed(x)
        x = self.pos_embed(x)
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        cls_token = self.norm(x[:, 0, :]) 

        return self.fc(cls_token)