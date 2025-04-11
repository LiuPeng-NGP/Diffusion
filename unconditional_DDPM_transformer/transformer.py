import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --- Helper Functions (Attention, Modulate) --- Remain the same ---

def attention(query:Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    sqrt_dim_head = query.shape[-1]**0.5

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head
    # Shape of scores [batch_size, num_heads, sequence_length, sequence_length]

    if mask is not None:
        scores = scores.masked_fill(mask==0, -5e4)

    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --- Modules (PositionalEncoding, MultiHeadAttention, etc.) --- Remain the same ---

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embed: int, max_len: int=1024, drop_prob: float =0.1) -> None:
        super(PositionalEncoding, self).__init__()

        assert dim_embed % 2 == 0

        self.dim_embed = dim_embed
        # NOTE: max_len here should correspond to the number of patches
        #       It is set correctly when Transformer is initialized.
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / dim_embed))

        pe = torch.zeros(max_len, dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_positions, dim_embed)
        pe = pe.unsqueeze(0)

        # Register as non-learnable parameters
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor):
        # Add positional encoding to the sequence of patch embeddings
        # Slicing ensures we only use encoding up to the actual sequence length
        x = x + self.pe[:, :x.size(1)] # Use x.size(1) which is num_patches
        x = self.dropout(x)
        return x

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
        query   = self.query(x)
        key     = self.key(y)
        value   = self.value(y)

        batch_size = x.size(0)
        # Reshape for multi-head attention
        query   = query .view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        key     = key   .view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        value   = value .view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1) # Add head dimension for broadcasting

        # Compute attention
        attn = attention(query, key, value, mask)
        # Concatenate heads and project
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embed)
        out = self.dropout(self.output(attn))
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float) -> None:
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn, bias=False), # Often dim_pffn = mlp_ratio * dim_embed
            nn.SiLU(),
            nn.Dropout(drop_prob),
            nn.Linear(dim_pffn, dim_embed), # Project back to embedding dimension
            nn.Dropout(drop_prob),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.pffn(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scaler timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000): # Increased max_period typically
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t expected to be shape [batch_size]
        # The input t to the main model forward is often normalized (e.g., t/T)
        # But timestep_embedding expects integer indices, or scales appropriately.
        # Assuming input t is already scaled (e.g., 0 to T or 0 to 1)
        # If t is [0, 1], scale it: t * (self.frequency_embedding_size -1 ) ? No, timestep_embedding handles float t.
        # If t is [0, T], it's fine. Let's assume t is [0, T] or similar range appropriate for max_period.
        # The DDPM code passes t_normalized (t.float() / self.T) which is [0, 1].
        # We need to scale it appropriately or adjust timestep_embedding.
        # Original DiT paper uses t/1000. Let's assume t is already normalized [0,1]
        # and scale it slightly for the embedding function.
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size) # Scale normalized t
        t_emb = self.mlp(t_freq)
        return t_emb

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Compute RMS = sqrt(mean(x^2))
        # Normalize = x / (RMS + eps)
        # rsqrt = 1 / sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) # Normalize in float32 for stability
        return output * self.weight # Apply learnable scale

class TransformerBlock(nn.Module):
    '''
    A Transformer Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    '''
    def __init__(self, num_heads, dim_embed, mlp_ratio=4.0, drop_prob=0.1)-> None:
        super().__init__()
        self.norm1 = RMSNorm(dim_embed, eps=1e-6) # Changed from 1e-5 to 1e-6 often used with RMSNorm
        self.self_atten = MultiHeadAttention(num_heads=num_heads, dim_embed=dim_embed, drop_prob=drop_prob)
        self.norm2 = RMSNorm(dim_embed, eps=1e-6)
        dim_pwff = int(dim_embed * mlp_ratio)
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        # Layer for conditioning (time embedding c)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 6 * dim_embed, bias=True) # Output channels: 2*(scale+shift+gate) for Attn + FFN
        )

    def forward(self, x, c, mask=None):
        # x: Input sequence (B, SeqLen, Dim)
        # c: Conditioning (time embedding) (B, Dim)
        # Get modulation parameters (shift, scale, gate) from time embedding c
        shift_mha, scale_mha, gate_mha, shift_ffd, scale_ffd, gate_ffd = self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention path
        # Modulate -> Norm -> Attention -> Modulate by gate
        residual = x
        x_norm1 = self.norm1(x)
        x_norm1 = modulate(x_norm1, shift_mha, scale_mha)
        attn_output = self.self_atten(x_norm1, x_norm1, mask) # Self-attention
        x = residual + gate_mha.unsqueeze(1) * attn_output # Gated residual connection

        # Feed-Forward path
        # Modulate -> Norm -> FFN -> Modulate by gate
        residual = x
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_ffd, scale_ffd)
        ffn_output = self.feed_forward(x_norm2)
        x = residual + gate_ffd.unsqueeze(1) * ffn_output # Gated residual connection

        return x

class FinalLayer(nn.Module):
    """
    The final layer of Transformer applies conditioning and final projection.
    """
    def __init__(self, dim_embed, out_dim_embed): # Allow output dim change if needed
        super().__init__()
        self.norm_final = RMSNorm(dim_embed, eps=1e-6)
        # Final linear projection
        self.linear = nn.Linear(dim_embed, out_dim_embed, bias=True) # Changed bias to True, common here
        # Conditioning for the final layer
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
        x = self.linear(x_norm) # Final projection
        return x

class Transformer(nn.Module):
    """
    Transformer-based diffusion model (implements patching).
    """
    def __init__(
            self,
            img_resolution=32,
            in_channels=3,
            out_channels=3,
            patch_size=4,      # <-- Accepts patch_size from config
            dim_embed=256,     # Renamed from dim_embed for clarity in __init__
            num_heads=8,
            depth=8,
            mlp_ratio=4.0,
            drop_prob=0.1,
            ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dim_embed = dim_embed # Core embedding dimension

        # --- 1. Patching and Embedding ---
        # Calculate the number of patches
        self.num_patches = (img_resolution // self.patch_size) ** 2

        # Patch embedding layer: Conv2d with kernel_size=patch_size and stride=patch_size
        # Input: (B, C, H, W) -> (B, 3, 32, 32)
        # Output: (B, dim_embed, H/patch_size, W/patch_size) -> (B, 256, 8, 8)
        self.patch_embed = nn.Conv2d(
            in_channels, dim_embed,
            kernel_size=self.patch_size, stride=self.patch_size
        )

        # --- 2. Positional and Timestep Embedding ---
        # Positional encoding for the sequence of patches
        self.x_embedder = PositionalEncoding(dim_embed=dim_embed, max_len=self.num_patches, drop_prob=drop_prob)
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim_embed) # Embeds t into dim_embed

        # --- 3. Transformer Blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob
            )
            for _ in range(depth)
        ])

        # --- 4. Final Layer ---
        # Processes the sequence output, conditioned on time
        # The output dimension of FinalLayer linear needs to project back to something
        # that can be un-patched. This should be dim_embed.
        self.final_layer = FinalLayer(dim_embed, dim_embed) # Output dim is dim_embed

        # --- 5. Unpatching (Output Projection) ---
        # Convert sequence back to image grid using Transposed Convolution
        # This layer needs to output the correct number of channels for the final image.
        # Input: (B, dim_embed, H/patch_size, W/patch_size) -> (B, 256, 8, 8)
        # Output: (B, out_channels, H, W) -> (B, 3, 32, 32)
        # The output dim of the final_layer is dim_embed, which is the input to this proj layer.
        self.output_proj = nn.ConvTranspose2d(
            dim_embed,                  # Input channels = embedding dimension
            out_channels,               # Output channels = image channels (e.g., 3 for RGB)
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like a linear layer
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.bias is not None:
            torch.nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize positional encoding if it's learnable (it's fixed here)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None: nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        if self.t_embedder.mlp[2].bias is not None: nn.init.constant_(self.t_embedder.mlp[2].bias, 0)


        # Initialize transformer blocks:
        for block in self.blocks:
            # Init linear layers
            nn.init.xavier_uniform_(block.self_atten.query.weight)
            nn.init.xavier_uniform_(block.self_atten.key.weight)
            nn.init.xavier_uniform_(block.self_atten.value.weight)
            nn.init.xavier_uniform_(block.self_atten.output.weight)
            if block.self_atten.output.bias is not None: nn.init.constant_(block.self_atten.output.bias, 0)

            nn.init.xavier_uniform_(block.feed_forward.pffn[0].weight)
            # nn.init.xavier_uniform_(block.feed_forward.pffn[3].weight) # Correct index assuming structure
            if block.feed_forward.pffn[3].bias is not None: nn.init.constant_(block.feed_forward.pffn[3].bias, 0)

            # Zero-out adaLN modulation layers weights for stability at init
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            if block.adaLN_modulation[-1].bias is not None: nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        # Initialize final layer
        # Zero-out adaLN modulation layers weights
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        if self.final_layer.adaLN_modulation[-1].bias is not None: nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Zero-out output projection linear layer weight
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None: nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize output projection (unpatching layer)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None: nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, x, t):
        """
        Forward pass of Transformer.
        x: (Batch_Size, Channels, Height, Width) tensor of images
        t: (Batch_Size,) tensor of diffusion timesteps (normalized to [0, 1])
        """
        B, C, H, W = x.shape
        assert H == self.img_resolution and W == self.img_resolution, "Input image resolution mismatch"

        # --- 1. Patch Embedding ---
        # Convert image to patches and embed them.
        # Input: (B, C, H, W)
        x = self.patch_embed(x)  # Output: (B, dim_embed, H/patch_size, W/patch_size)
        Hp, Wp = x.shape[2], x.shape[3] # Height and Width of patch grid

        # --- 2. Flatten Patches into Sequence ---
        # Reshape: (B, dim_embed, Hp, Wp) -> (B, dim_embed, Hp*Wp) -> (B, Hp*Wp, dim_embed)
        x = x.flatten(2).transpose(1, 2) # Output: (B, num_patches, dim_embed)
        assert x.shape[1] == self.num_patches

        # --- 3. Add Positional Encoding ---
        x = self.x_embedder(x) # Output: (B, num_patches, dim_embed)

        # --- 4. Get Timestep Embedding ---
        # Input t: (B,) tensor of timesteps (normalized [0, 1])
        t_emb = self.t_embedder(t) # Output: (B, dim_embed) - Time embedding

        # --- 5. Apply Transformer Blocks ---
        # Pass sequence through transformer blocks, conditioning on time embedding
        for block in self.blocks:
            x = block(x, t_emb) # Output: (B, num_patches, dim_embed)

        # --- 6. Apply Final Layer ---
        # Final processing, conditioned on time embedding
        x = self.final_layer(x, t_emb) # Output: (B, num_patches, dim_embed)

        # --- 7. Reshape Sequence back to Grid ---
        # Reshape: (B, num_patches, dim_embed) -> (B, dim_embed, num_patches) -> (B, dim_embed, Hp, Wp)
        x = x.transpose(1, 2).view(B, self.dim_embed, Hp, Wp)

        # --- 8. Unpatch (Project back to Image) ---
        # Use transposed convolution to map patch embeddings back to image pixels
        # Input: (B, dim_embed, Hp, Wp)
        x = self.output_proj(x) # Output: (B, out_channels, H, W)

        return x