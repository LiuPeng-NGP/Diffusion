# Keep: import math # Still needed for TimestepEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# --- Helper Functions (Attention, Modulate) --- Remain the same ---

def attention(query:Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    sqrt_dim_head = query.shape[-1]**0.5

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head
    # Shape of scores [batch_size, num_heads, sequence_length, sequence_length]

    if mask is not None:
        # Ensure mask has compatible dimensions for broadcasting
        # Mask shape: [B, 1, T, T] or [B, N_heads, T, T]
        scores = scores.masked_fill(mask == 0, -1e9) # Use large negative number

    weight = F.softmax(scores, dim=-1)
    # Handle potential NaNs from softmax if all scores are -inf (e.g., due to masking)
    # This shouldn't happen with -1e9 but good practice for -inf
    # weight = torch.nan_to_num(weight)
    return torch.matmul(weight, value)

def modulate(x, shift, scale):
    # x: (B, SeqLen, Dim)
    # shift, scale: (B, Dim)
    # Unsqueeze to broadcast across SeqLen: (B, 1, Dim)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --- Modules (PositionalEncoding Removed, MHA, PFFN, TimestepEmbedder) ---

# Removed PositionalEncoding class, will use learnable nn.Parameter

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()
        assert dim_embed % num_heads == 0

        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        self.query  = nn.Linear(dim_embed, dim_embed)
        self.key    = nn.Linear(dim_embed, dim_embed)
        self.value  = nn.Linear(dim_embed, dim_embed)
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor=None) -> Tensor:
        # x is query (B, Q_len, Dim), y is key/value (B, KV_len, Dim)
        query   = self.query(x)
        key     = self.key(y)
        value   = self.value(y)

        batch_size = x.size(0)
        query_len = x.size(1)
        kv_len = y.size(1)

        # Reshape for multi-head attention
        # (B, SeqLen, Dim) -> (B, SeqLen, N_Heads, Dim_Head) -> (B, N_Heads, SeqLen, Dim_Head)
        query   = query .view(batch_size, query_len, self.num_heads, self.dim_head).transpose(1,2)
        key     = key   .view(batch_size, kv_len,    self.num_heads, self.dim_head).transpose(1,2)
        value   = value .view(batch_size, kv_len,    self.num_heads, self.dim_head).transpose(1,2)

        if mask is not None:
             # Expected mask shape: (B, Q_len, KV_len) or broadcastable
             # Add head dim: (B, 1, Q_len, KV_len)
            mask = mask.unsqueeze(1)

        # Compute attention: Output shape (B, N_Heads, Q_len, Dim_Head)
        attn = attention(query, key, value, mask)

        # Concatenate heads and project
        # (B, N_Heads, Q_len, Dim_Head) -> (B, Q_len, N_Heads, Dim_Head) -> (B, Q_len, Dim)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, query_len, self.dim_embed)
        out = self.dropout(self.output(attn))
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float) -> None:
        super().__init__()
        # UViT uses GELU, but SiLU (Swish) is also very effective. Keep SiLU for now.
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn, bias=True), # Add bias=True (common)
            nn.SiLU(),
            nn.Dropout(drop_prob),
            nn.Linear(dim_pffn, dim_embed, bias=True), # Add bias=True (common)
            nn.Dropout(drop_prob),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.pffn(x)

class TimestepEmbedder(nn.Module):
    """ Embeds scalar timesteps into vector representations. """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """ Create sinusoidal timestep embeddings. """
        # t: 1D Tensor of N indices, possibly fractional. [0, 1] range assumed input.
        # Scale t to a range (e.g., 0-10000) for the embedding function
        # DiT multiplies normalized t by 1000. UViT uses raw timesteps?
        # Let's keep the scaling similar to DiT for now.
        t_scaled = t * max_period # Scale normalized t from [0,1] to [0, max_period]

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t_scaled[:, None].float() * freqs[None] # Use scaled t
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t expected shape [B], normalized to [0, 1]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# Removed RMSNorm, replaced with nn.LayerNorm

class TransformerBlock(nn.Module):
    '''
    A Transformer Block with adaptive layer norm zero (adaLN-Zero) conditioning
    and optional skip connection handling.
    '''
    def __init__(self, num_heads, dim_embed, mlp_ratio=4.0, drop_prob=0.1, use_skip=False)-> None:
        super().__init__()
        self.use_skip = use_skip
        # Replaced RMSNorm with LayerNorm
        self.norm1 = nn.LayerNorm(dim_embed, eps=1e-6) # Standard eps for LayerNorm often 1e-5 or 1e-6
        self.self_atten = MultiHeadAttention(num_heads=num_heads, dim_embed=dim_embed, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(dim_embed, eps=1e-6)
        dim_pwff = int(dim_embed * mlp_ratio)
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        # Layer for conditioning (time embedding c)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 6 * dim_embed, bias=True) # shift/scale/gate for Attn & FFN
        )
        # Optional linear layer to process skip connection, like UViT
        if self.use_skip:
            self.skip_linear = nn.Linear(2 * dim_embed, dim_embed, bias=True)

    def forward(self, x, c, skip=None, mask=None):
        # x: Input sequence (B, SeqLen, Dim)
        # c: Conditioning (time embedding) (B, Dim)
        # skip: Optional skip connection tensor (B, SeqLen, Dim) from encoder block

        # --- Skip Connection Fusion (if applicable) ---
        if self.use_skip and skip is not None:
            # Concatenate along the feature dimension and project
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        elif self.use_skip and skip is None:
            # This should ideally not happen if architecture is symmetric
             print("Warning: TransformerBlock expected skip connection but received None.")


        # Get modulation parameters (shift, scale, gate) from time embedding c
        shift_mha, scale_mha, gate_mha, shift_ffd, scale_ffd, gate_ffd = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Self-Attention path ---
        residual = x
        x_norm1 = self.norm1(x)
        x_norm1 = modulate(x_norm1, shift_mha, scale_mha) # Apply AdaLN modulation
        attn_output = self.self_atten(x_norm1, x_norm1, mask) # Self-attention
        x = residual + gate_mha.unsqueeze(1) * attn_output # Gated residual connection

        # --- Feed-Forward path ---
        residual = x
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_ffd, scale_ffd) # Apply AdaLN modulation
        ffn_output = self.feed_forward(x_norm2)
        x = residual + gate_ffd.unsqueeze(1) * ffn_output # Gated residual connection

        return x

class FinalLayer(nn.Module):
    """
    The final layer of Transformer. Applies LN, conditioning, and final projection for unpatching.
    """
    def __init__(self, dim_embed, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim_embed, eps=1e-6)
        # Final linear projection: map Dim -> (PatchSize^2 * OutChannels) for PixelShuffle
        self.linear = nn.Linear(dim_embed, (patch_size ** 2) * out_channels, bias=True)
        # Conditioning for the final layer (modulates *before* final projection)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 2 * dim_embed, bias=True) # Output: shift, scale
        )

    def forward(self, x, c):
        # x: (B, SeqLen, Dim)
        # c: (B, Dim)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_norm = self.norm_final(x)
        x_norm = modulate(x_norm, shift, scale) # Apply modulation
        x = self.linear(x_norm) # Final projection to shape (B, SeqLen, P*P*C)
        return x

# --- Main Transformer Model ---

class Transformer(nn.Module):
    """
    Transformer-based diffusion model with U-Net skips and improved unpatching.
    API remains: forward(self, x, t)
    """
    def __init__(
            self,
            img_resolution=32,
            in_channels=3,
            out_channels=3,
            patch_size=4,
            dim_embed=256,
            num_heads=8,
            depth=8,         # Total number of blocks (must be even for U-Net structure)
            mlp_ratio=4.0,
            drop_prob=0.1,
            final_conv=True, # Add optional final conv like UViT
            ):
        super().__init__()
        assert depth % 2 == 0, "Depth must be even for U-Net structure"
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dim_embed = dim_embed
        self.depth = depth
        self.final_conv_enabled = final_conv

        # --- 1. Patching and Embedding ---
        self.num_patches = (img_resolution // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, dim_embed,
            kernel_size=self.patch_size, stride=self.patch_size
        )

        # --- 2. Positional and Timestep Embedding ---
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim_embed))
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim_embed)

        # --- 3. Transformer Blocks (U-Net Structure) ---
        self.in_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                use_skip=False # Encoder blocks don't use skips internally
            )
            for _ in range(depth // 2)
        ])

        self.mid_block = TransformerBlock( # A single block in the middle
            num_heads=num_heads,
            dim_embed=dim_embed,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            use_skip=False
        )

        self.out_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                use_skip=True # Decoder blocks use skips
            )
            for _ in range(depth // 2)
        ])

        # --- 4. Final Layer ---
        # Processes the sequence output, conditioned on time, projects for unpatching
        self.final_layer = FinalLayer(dim_embed, patch_size, out_channels)

        # --- 5. Unpatching (PixelShuffle) ---
        self.unpatchify = nn.PixelShuffle(patch_size)

        # --- 6. Optional Final Convolution ---
        if self.final_conv_enabled:
            self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.final_conv = nn.Identity() # Does nothing if final_conv=False

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (from DiT)
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.bias is not None:
            torch.nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize learnable positional embedding
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.t_embedder.mlp[0], 'bias') and self.t_embedder.mlp[0].bias is not None: nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        if hasattr(self.t_embedder.mlp[2], 'bias') and self.t_embedder.mlp[2].bias is not None: nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        # Initialize transformer blocks:
        for block in self.in_blocks + [self.mid_block] + self.out_blocks:
            # Init linear layers (Attention & FFN)
            nn.init.xavier_uniform_(block.self_atten.query.weight)
            nn.init.xavier_uniform_(block.self_atten.key.weight)
            nn.init.xavier_uniform_(block.self_atten.value.weight)
            nn.init.xavier_uniform_(block.self_atten.output.weight)
            if block.self_atten.output.bias is not None: nn.init.constant_(block.self_atten.output.bias, 0)

            nn.init.xavier_uniform_(block.feed_forward.pffn[0].weight) # Linear 1
            nn.init.xavier_uniform_(block.feed_forward.pffn[3].weight) # Linear 2 (index assuming structure)
            if block.feed_forward.pffn[0].bias is not None: nn.init.constant_(block.feed_forward.pffn[0].bias, 0)
            if block.feed_forward.pffn[3].bias is not None: nn.init.constant_(block.feed_forward.pffn[3].bias, 0)

            # Init LayerNorms
            nn.init.constant_(block.norm1.weight, 1.0)
            nn.init.constant_(block.norm1.bias, 0)
            nn.init.constant_(block.norm2.weight, 1.0)
            nn.init.constant_(block.norm2.bias, 0)

            # Zero-out adaLN modulation layers weights (crucial for AdaLN-Zero)
            # Last layer of the sequential is the Linear layer
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            if block.adaLN_modulation[-1].bias is not None: nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Initialize skip linear layer if it exists
            if hasattr(block, 'skip_linear') and block.skip_linear is not None:
                 nn.init.xavier_uniform_(block.skip_linear.weight)
                 if block.skip_linear.bias is not None: nn.init.constant_(block.skip_linear.bias, 0)


        # Initialize final layer
        # Zero-out adaLN modulation layers weights
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        if self.final_layer.adaLN_modulation[-1].bias is not None: nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Initialize final projection linear layer (but maybe zero-init bias, small init weight?)
        # DiT zero-initializes the output projection. Let's follow that.
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None: nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize final LayerNorm
        nn.init.constant_(self.final_layer.norm_final.weight, 1.0)
        nn.init.constant_(self.final_layer.norm_final.bias, 0)

        # Initialize final convolution (if used) - Xavier is reasonable
        if self.final_conv_enabled and isinstance(self.final_conv, nn.Conv2d):
            nn.init.xavier_uniform_(self.final_conv.weight)
            if self.final_conv.bias is not None: nn.init.constant_(self.final_conv.bias, 0)


    def _pos_embed_reshape(self, B, H_patch, W_patch, D):
        # Ensure pos_embed matches the flattened sequence length
        # If pos_embed was learned on a different num_patches (e.g., different resolution fine-tuning)
        # this basic interpolation might be needed. Assumes square images/patch grids.
        # For now, we assume num_patches is fixed so simple broadcast works.
        # If dynamic resizing is needed, 2D interpolation on the grid before flattening is better.
        # Current pos_embed shape: (1, L, D) where L = num_patches
        # Required shape: (B, L, D)
        # If self.pos_embed.shape[1] != H_patch * W_patch:
            # Implement interpolation here if needed
            # pass
        return self.pos_embed.expand(B, -1, -1)


    def forward(self, x, t):
        """
        Forward pass of Transformer.
        x: (Batch_Size, Channels, Height, Width) tensor of images
        t: (Batch_Size,) tensor of diffusion timesteps (normalized to [0, 1])
        """
        B, C, H, W = x.shape
        assert H == self.img_resolution and W == self.img_resolution, "Input image resolution mismatch"

        # --- 1. Patch Embedding ---
        x = self.patch_embed(x)  # Output: (B, dim_embed, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3] # Height and Width of patch grid

        # --- 2. Flatten Patches into Sequence ---
        x = x.flatten(2).transpose(1, 2) # Output: (B, num_patches, dim_embed)
        assert x.shape[1] == self.num_patches

        # --- 3. Add Positional Encoding ---
        pos_embed = self._pos_embed_reshape(B, Hp, Wp, self.dim_embed)
        x = x + pos_embed # Add learnable positional encoding

        # --- 4. Get Timestep Embedding ---
        t_emb = self.t_embedder(t) # Output: (B, dim_embed) - Time embedding

        # --- 5. Apply Transformer Blocks (U-Net Structure) ---
        skips = []
        # Input blocks (Encoder)
        for block in self.in_blocks:
            x = block(x, t_emb)
            skips.append(x) # Store output for skip connection

        # Middle block
        x = self.mid_block(x, t_emb)

        # Output blocks (Decoder)
        for block in self.out_blocks:
            skip = skips.pop() # Get corresponding skip connection
            x = block(x, t_emb, skip=skip) # Pass skip connection

        # --- 6. Apply Final Layer ---
        # Final LN, modulation, and projection
        x = self.final_layer(x, t_emb) # Output: (B, num_patches, P*P*out_channels)

        # --- 7. Unpatch (Reshape + PixelShuffle) ---
        # Reshape for PixelShuffle: (B, L, P*P*C) -> (B, C*P*P, L) -> (B, C*P*P, Hp, Wp)
        x = x.transpose(1, 2).view(B, self.out_channels * (self.patch_size**2), Hp, Wp)
        # Apply PixelShuffle: (B, C*P*P, Hp, Wp) -> (B, C, Hp*P, Wp*P) = (B, C, H, W)
        x = self.unpatchify(x)

        # --- 8. Optional Final Convolution ---
        x = self.final_conv(x) # Output: (B, out_channels, H, W)

        return x

# # Example Usage (for checking shapes)
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Config matching roughly the UViT CIFAR-10 example but using DiT style parameters where needed
#     model_cfg = {
#         "img_resolution": 32,
#         "in_channels": 3,
#         "out_channels": 3,
#         "patch_size": 4, # UViT uses 2 for CIFAR, DiT often 2, 4, or 8
#         "dim_embed": 512, # UViT uses 512
#         "num_heads": 8,   # UViT uses 8
#         "depth": 12,      # UViT uses 12
#         "mlp_ratio": 4.0,
#         "drop_prob": 0.1, # UViT doesn't specify dropout, but common
#         "final_conv": True # Match UViT
#     }

#     model = Transformer(**model_cfg).to(device)
#     print(f"Model initialized on {device}")

#     # Check parameter count
#     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Parameter count: {params / 1e6:.2f} M") # Should be roughly comparable to UViT now

#     # Test forward pass
#     batch_size = 4
#     dummy_x = torch.randn(batch_size,
#                           model_cfg['in_channels'],
#                           model_cfg['img_resolution'],
#                           model_cfg['img_resolution']).to(device)
#     dummy_t = torch.rand(batch_size).to(device) # Timesteps normalized [0, 1]

#     try:
#         with torch.no_grad():
#             output = model(dummy_x, dummy_t)
#         print(f"Input shape: {dummy_x.shape}")
#         print(f"Timestep shape: {dummy_t.shape}")
#         print(f"Output shape: {output.shape}")
#         assert output.shape == dummy_x.shape, "Output shape mismatch!"
#         print("Forward pass successful!")
#     except Exception as e:
#         print(f"Forward pass failed: {e}")
#         import traceback
#         traceback.print_exc()