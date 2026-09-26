import numpy as np

import torch
import torch.nn as nn

from einops import rearrange


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


def img_to_uint8(img_tensor):
    return (img_tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim=384,        # Dimension from your Encoder
        decoder_embed_dim=1024, # Decoder is usually smaller/lighter
        decoder_depth=3, 
        decoder_num_heads=32,
        img_size=224,
        patch_size=14,
        num_regs=1,
    ):  
        super().__init__()

        self.grid_size = img_size // patch_size # 16
        self.num_patches = self.grid_size ** 2
        self.patch_size = patch_size
        self.num_regs = num_regs
        # 1. Project Encoder output to Decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # 2. Learnable tokens for the "missing" spatial patches
        if self.num_regs == 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 0, decoder_embed_dim))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        # 3. The Transformer "Body"
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 4. Prediction Head: Projects to pixel values (14*14*3 = 588)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)

        # 5. Fixed or learnable 2D Positional Embeddings
        self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, self.num_patches + self.num_regs, decoder_embed_dim), # +1 for CLS
                    requires_grad=False
        )
        
        self.initialize_pos_embed()

    def initialize_pos_embed(self):
        # Generate the 2D grid embeddings (256, decoder_embed_dim)
        pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=False)
        
        # We need to account for CLS + 4 Registers (5 tokens total)
        # We leave them as zeros so they don't have spatial bias
        full_pos_embed = np.concatenate([np.zeros([self.num_regs, self.decoder_pos_embed.shape[-1]]), pos_embed], axis=0)
        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(full_pos_embed).float().unsqueeze(0))

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * 3) -> (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        # 1. Reshape to (Batch, Grid_H, Grid_W, P_H, P_W, Channels)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        
        # 2. Permute to (Batch, Channels, Grid_H, P_H, Grid_W, P_W)
        x = torch.einsum('nhwpqc->nchpwq', x)
        
        # 3. Flatten into (Batch, 3, H, W)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward(self, tokens):
        """
        tokens: [B, T, 1, D] if CLS only
        tokens: [B, T, 5, D] if CLS + Registers
        """
        B, T, num_global, D = tokens.shape
        N = self.num_patches # 256 for 16x16 grid
        tokens = rearrange(tokens, 'b t n d -> (b t) n d')
        
        # 1. Project all input tokens to decoder dimension
        x_global = self.decoder_embed(tokens) # [B*T, 1 or 5, D_dec]
        
        # 2. Expand Mask Tokens for the spatial grid
        mask_tokens = self.mask_token.expand(B * T, -1, -1) # [B*T, 256, D_dec]
        
        # 3. Concatenate global context with spatial masks
        # Result: [B*T, 257, D_dec] or [B*T, 261, D_dec]
        x = torch.cat([x_global, mask_tokens], dim=1)
        
        # 4. Add Positional Encoding
        # Note: Your pos_embed must match the length (Global + 256)
        # Usually, we don't give pos_embed to CLS/Reg, only to the grid. 
        # Position embeds for Cls/Reg tokens initialzed as zeros, so they won't affect the global tokens. 
        x = x + self.decoder_pos_embed
        
        # 5. Transformer Blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        # 6. Prediction Head (Only on the 256 spatial slots)
        patches = self.decoder_pred(x[:, num_global:, :]) 
        
        return self.unpatchify(patches), torch.zeros(1).to(tokens.device) # dummy placeholder