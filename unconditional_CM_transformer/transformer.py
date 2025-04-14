import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange # Often useful for transformers

# --- Helper Functions (Attention, Modulate) --- Remain the same ---

def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
    # Standard scaled dot-product attention
    sqrt_dim_head = query.shape[-1]**0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt_dim_head

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.finfo(scores.dtype).max) # Use finfo for better numerical stability

    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)

def modulate(x, shift, scale):
    # Applies adaLN modulation (scale and shift)
    # x: (B, SeqLen, Dim)
    # shift, scale: (B, Dim)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --- Modules ---

class PositionalEncoding(nn.Module):
    # Standard Sinusoidal Positional Encoding
    def __init__(self, dim_embed: int, max_len: int = 1024, drop_prob: float = 0.0) -> None: # Default drop_prob to 0 for PE
        super(PositionalEncoding, self).__init__()
        assert dim_embed % 2 == 0
        self.dim_embed = dim_embed
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2).float() # Ensure float for division
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / dim_embed))

        pe = torch.zeros(1, max_len, dim_embed) # Start with batch dim
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe, persistent=False) # Register as non-learnable buffer
        # Dropout after adding PE is optional, sometimes omitted for stability
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor):
        # x shape: (Batch, SeqLen, Dim)
        # Add positional encoding up to the sequence length
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()
        assert dim_embed % num_heads == 0
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        # Use a single projection for QKV for potential efficiency/parameter sharing pattern
        self.qkv = nn.Linear(dim_embed, dim_embed * 3, bias=True) # Bias often included
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout_attn = nn.Dropout(drop_prob) # Dropout on attention weights (optional)
        self.dropout_output = nn.Dropout(drop_prob) # Dropout on output

    def forward(self, x: Tensor, y: Tensor, mask: Tensor = None) -> Tensor:
        B, N, C = x.shape # Batch, SeqLen, Channels(dim_embed)
        _B, _N, _C = y.shape

        # Project x to Q, y to K,V
        # This assumes self-attention (x=y). If cross-attention needed, adjust.
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x.unbind(0) # Shape: (B, num_heads, N, dim_head)

        # If cross-attention is needed (y != x), project y separately for K, V
        # if x is not y:
        #    kv_y = self.kv(y).reshape(B, N, 2, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        #    k, v = kv_y.unbind(0)

        attn = attention(q, k, v, mask) # (B, num_heads, N, dim_head)
        attn = attn.transpose(1, 2).reshape(B, N, C) # (B, N, C) concat heads
        attn = self.dropout_attn(attn) # Optional dropout on attention scores

        out = self.output(attn)
        out = self.dropout_output(out) # Dropout before residual connection
        return out

class PositionwiseFeedForward(nn.Module):
    # Standard MLP block (Linear -> Activation -> Dropout -> Linear -> Dropout)
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float, bias: bool = True) -> None:
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn, bias=bias),
            nn.SiLU(), # SiLU (Swish) is common and effective
            nn.Dropout(drop_prob),
            nn.Linear(dim_pffn, dim_embed, bias=bias),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pffn(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings + MLP.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Initialize MLP layers
        self.init_weights()

    def init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Standard sinusoidal timestep embedding.
        t: tensor of shape [N]
        dim: embedding dimension
        max_period: max freq cycle
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: # Zero padding if dim is odd
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Assumes t is shape [batch_size] and represents values (e.g., noise levels or steps)
        # If t is normalized [0, 1], scaling might be needed depending on expected range for embedding
        # Example: t * 1000 if input t is normalized and timestep_embedding expects larger values
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# --- Normalization Options ---
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize in float32 for stability, then cast back
        output = self._norm(x.float()).type_as(x)
        if self.elementwise_affine:
            output = output * self.weight
        return output

class LayerNormWrapper(nn.Module):
    # Simple wrapper to match expected interface if using LayerNorm
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=False): # Defaulting affine=False for adaLN
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.norm(x)

# --- Transformer Block with adaLN-Zero ---

class TransformerBlock(nn.Module):
    '''
    A Transformer Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Uses modulate() for conditioning.
    '''
    def __init__(self, num_heads, dim_embed, mlp_ratio=4.0, drop_prob=0.1, norm_type="rmsnorm", norm_eps=1e-6)-> None:
        super().__init__()
        # Choose normalization layer type
        if norm_type == "rmsnorm":
            # RMSNorm doesn't need elementwise_affine=True if used with modulate
            self.norm1 = RMSNorm(dim_embed, eps=norm_eps, elementwise_affine=True) # Let RMSNorm have its scale
            self.norm2 = RMSNorm(dim_embed, eps=norm_eps, elementwise_affine=True)
        elif norm_type == "layernorm":
            # LayerNorm elementwise_affine should be False if using modulate
            self.norm1 = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=False)
            self.norm2 = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.self_atten = MultiHeadAttention(num_heads=num_heads, dim_embed=dim_embed, drop_prob=drop_prob)
        dim_pwff = int(dim_embed * mlp_ratio)
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)

        # adaLN_modulation: projects time embedding to 6 values (shift_mha, scale_mha, gate_mha, ...)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 6 * dim_embed, bias=True)
        )

    def forward(self, x, c): # c is the conditional embedding (e.g., time)
        # x: (B, SeqLen, Dim), c: (B, Dim)

        # Get modulation parameters: shift, scale, gate for Attn and FFN paths
        shift_mha, scale_mha, gate_mha, shift_ffd, scale_ffd, gate_ffd = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention path
        residual = x
        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_mha, scale_mha) # Modulate AFTER norm
        attn_output = self.self_atten(x_modulated1, x_modulated1) # Self-attention
        x = residual + gate_mha.unsqueeze(1) * attn_output # Gated residual

        # Feed-Forward path
        residual = x
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_ffd, scale_ffd) # Modulate AFTER norm
        ffn_output = self.feed_forward(x_modulated2)
        x = residual + gate_ffd.unsqueeze(1) * ffn_output # Gated residual

        return x


class FinalLayer(nn.Module):
    """
    The final layer applies adaLN modulation and projects to output dimension.
    """
    def __init__(self, dim_embed, out_channels, norm_type="rmsnorm", norm_eps=1e-6):
        super().__init__()
        if norm_type == "rmsnorm":
             # RMSNorm doesn't need elementwise_affine=True if used with modulate
            self.norm_final = RMSNorm(dim_embed, eps=norm_eps, elementwise_affine=True)
        elif norm_type == "layernorm":
            # LayerNorm elementwise_affine should be False if using modulate
            self.norm_final = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Modulation layer for shift/scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 2 * dim_embed, bias=True)
        )
        # Final linear projection to the dimension needed for unpatching
        self.linear = nn.Linear(dim_embed, out_channels, bias=True) # Output should match input to ConvTranspose2d

    def forward(self, x, c):
        # x: (B, SeqLen, Dim), c: (B, Dim)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_norm = self.norm_final(x)
        x_modulated = modulate(x_norm, shift, scale) # Modulate AFTER norm
        x = self.linear(x_modulated) # Project to output dimension
        return x


class Transformer(nn.Module):
    """
    Transformer-based diffusion model using Patching and adaLN-Zero conditioning.
    """
    def __init__(
            self,
            img_resolution=32,
            in_channels=3,
            out_channels=3,       # Typically same as in_channels for noise prediction
            patch_size=4,
            dim_embed=768,        # Embedding dimension (often larger in Transformers)
            num_heads=12,         # Number of attention heads
            depth=12,             # Number of Transformer blocks
            mlp_ratio=4.0,        # Ratio for FFN hidden dim
            drop_prob=0.1,        # Dropout probability
            learn_pe=False,       # Whether to learn positional embeddings
            norm_type="rmsnorm",  # Normalization type: "rmsnorm" or "layernorm"
            norm_eps=1e-6,        # Epsilon for normalization layers
            t_embed_dim=256       # Dimension for sinusoidal time embedding frequencies
            ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.depth = depth
        self.norm_type = norm_type
        self.learn_pe = learn_pe

        # --- 1. Patching ---
        self.patch_embed = nn.Conv2d(
            in_channels, dim_embed,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True # Bias is common here
        )
        self.num_patches = (img_resolution // self.patch_size) ** 2

        # --- 2. Positional and Timestep Embedding ---
        if learn_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim_embed))
        else:
            self.pos_embed = PositionalEncoding(dim_embed, max_len=self.num_patches, drop_prob=0.0)

        self.t_embedder = TimestepEmbedder(hidden_size=dim_embed, frequency_embedding_size=t_embed_dim)

        # --- 3. Transformer Blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                norm_type=norm_type,
                norm_eps=norm_eps
            )
            for _ in range(depth)
        ])

        # --- 4. Final Layer ---
        # The projection needs to output dim_embed features per patch for reshaping before unpatching.
        self.final_layer = FinalLayer(
            dim_embed=dim_embed,
            out_channels=dim_embed, # Output dim matches input dim for unpatching proj.
            norm_type=norm_type,
            norm_eps=norm_eps
        )

        # --- 5. Unpatching (Output Projection) ---
        # Projects patch features back to pixel space. Output channels = image channels.
        self.output_proj = nn.ConvTranspose2d(
            dim_embed,
            self.out_channels, # Target image channels
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Initialize weights
        self.initialize_weights()
        print(f"Transformer initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")


    def initialize_weights(self):
        # Initialize patch_embed like a linear layer
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize positional embedding if learnable
        if self.learn_pe:
            nn.init.normal_(self.pos_embed, std=.02)

        # Timestep embedder MLP is initialized within its class

        # Initialize transformer blocks:
        for block in self.blocks:
            # Attention projections (QKV and output)
            if hasattr(block.self_atten, 'qkv'):
                 nn.init.xavier_uniform_(block.self_atten.qkv.weight)
                 if block.self_atten.qkv.bias is not None: nn.init.constant_(block.self_atten.qkv.bias, 0)
            nn.init.xavier_uniform_(block.self_atten.output.weight)
            if block.self_atten.output.bias is not None: nn.init.constant_(block.self_atten.output.bias, 0)

            # FeedForward layers
            nn.init.xavier_uniform_(block.feed_forward.pffn[0].weight)
            if block.feed_forward.pffn[0].bias is not None: nn.init.constant_(block.feed_forward.pffn[0].bias, 0)
            nn.init.xavier_uniform_(block.feed_forward.pffn[3].weight)
            if block.feed_forward.pffn[3].bias is not None: nn.init.constant_(block.feed_forward.pffn[3].bias, 0)

            # *** Crucial: Zero-out the adaLN modulation output layer ***
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize final layer
        # *** Crucial: Zero-out the final adaLN modulation output layer ***
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # *** Crucial: Zero-out the final linear projection layer ***
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize output projection (unpatching layer) - Xavier is reasonable here
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)


    def unpatchify(self, x):
        """
        x: (B, N, C) N = num_patches, C = embed_dim
        imgs: (B, C_img, H, W)
        """
        B, N, C = x.shape
        Hp = Wp = int(N**0.5) # Assume square patch grid
        assert Hp * Wp == N
        assert C == self.dim_embed # Ensure correct dimension before reshape

        # Reshape sequence to grid: (B, N, C) -> (B, C, N) -> (B, C, Hp, Wp)
        x = x.transpose(1, 2).view(B, self.dim_embed, Hp, Wp)

        # Use ConvTranspose2d to upscale and reduce channels
        imgs = self.output_proj(x)
        return imgs

    def forward(self, x, t, class_labels=None): # class_labels currently unused
        """
        Forward pass of Transformer.
        x: (B, C_img, H, W) tensor of images
        t: (B,) tensor of diffusion timesteps (scalar values, potentially normalized)
        """
        B, C_img, H, W = x.shape
        assert H == self.img_resolution and W == self.img_resolution, "Input image resolution mismatch"

        # --- 1. Patch Embedding ---
        x = self.patch_embed(x)  # (B, dim_embed, Hp, Wp)
        # Reshape to sequence: (B, dim_embed, Hp*Wp) -> (B, Hp*Wp, dim_embed)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, dim_embed)

        # --- 2. Add Positional Encoding ---
        if self.learn_pe:
            x = x + self.pos_embed
        else:
            x = self.pos_embed(x) # Apply sinusoidal PE

        # --- 3. Get Timestep Embedding ---
        # Ensure t is correctly scaled if necessary before passing to embedder
        # Example: If t is [0, 1], scale by T=1000: t_input = t * 1000
        # If t is already in appropriate range (e.g., 0 to T), use directly: t_input = t
        # ** Assuming `t` passed from `sCM` is suitable for `TimestepEmbedder` **
        t_emb = self.t_embedder(t) # (B, dim_embed) - Time embedding

        # --- Class Embedding (Placeholder if needed later) ---
        # if class_labels is not None:
        #     c_emb = self.class_embedder(class_labels) # (B, dim_embed)
        #     t_emb = t_emb + c_emb # Combine embeddings

        # --- 4. Apply Transformer Blocks ---
        for block in self.blocks:
            x = block(x, t_emb) # (B, num_patches, dim_embed)

        # --- 5. Apply Final Layer ---
        x = self.final_layer(x, t_emb) # (B, num_patches, dim_embed) - Ready for unpatching

        # --- 6. Unpatch (Project back to Image) ---
        x = self.unpatchify(x) # (B, out_channels, H, W)

        # Ensure output shape matches input shape if predicting noise (out_channels = in_channels)
        assert x.shape == (B, self.out_channels, H, W)
        return x