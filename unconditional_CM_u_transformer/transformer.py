import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import rearrange # Often useful for transformers

# ANSI color codes
_RED = "\033[91m"
_RESET = "\033[0m"

# --- Debugging Module ---
_DEBUG_ENABLED = False  # Global flag to control debugging - SET TO FALSE BY DEFAULT
# _DEBUG_ENABLED = True  # Uncomment this line to enable debugging

def set_debug_enabled(enabled: bool):
    """Globally enable or disable debug printing."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled

def debug_print_stats(name: str, x: Tensor):
    """Prints tensor statistics if debugging is enabled, but only for NaNs/Infs or non-tensors/empty tensors."""
    if not _DEBUG_ENABLED: # This check remains, but _DEBUG_ENABLED is now False by default
        return

    if not isinstance(x, Tensor):
        # Use red color for non-tensor debug message
        print(f"{_RED}DEBUG: {name} is not a tensor ({type(x)}).{_RESET}")
        return
    if x.numel() == 0:
        print(f"DEBUG: {name} is empty. Shape: {x.shape}")
        return

    # Detach before calculating stats to avoid impacting gradients if any ops aren't no_grad
    with torch.no_grad():
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()

        if has_nan or has_inf:
            # Determine if we need red color for the main stats line
            stats_prefix = f"DEBUG: {name}"
            color_prefix = _RED
            color_suffix = _RESET

            if x.is_floating_point():
                 print(f"{color_prefix}{stats_prefix} - Shape: {tuple(x.shape)}, Dtype: {x.dtype}, Device: {x.device}, "
                       f"NaN: {has_nan}, Inf: {has_inf}, "
                       f"Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, "
                       f"Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}{color_suffix}")
            else:
                 print(f"{color_prefix}{stats_prefix} - Shape: {tuple(x.shape)}, Dtype: {x.dtype}, Device: {x.device}, "
                       f"NaN: {has_nan}, Inf: {has_inf}, "
                       f"Min: {x.min().item()}, Max: {x.max().item()}{color_suffix}") # No mean/std for non-float

            # Print additional warning in red if NaN/Inf detected
            print(f"{_RED}!!! WARNING: NaN or Inf detected in {name} !!!{_RESET}")


# --- Helper Functions (Attention) --- Modulate helper removed as it's integrated differently ---

def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
    # Standard scaled dot-product attention
    sqrt_dim_head = query.shape[-1]**0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt_dim_head

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.finfo(scores.dtype).max) # Use finfo for better numerical stability

    weight = F.softmax(scores, dim=-1)
    # Add NaN check for weights
    if torch.isnan(weight).any():
        if _DEBUG_ENABLED: # This print is now conditional
            print(f"{_RED}NaN detected in attention weights!{_RESET}")
        # Optionally handle (e.g., replace NaNs, though this might hide underlying issues)
        # weight = torch.nan_to_num(weight, nan=0.0) # Be cautious using this
    # Add NaN check for value - This check was already present, keeping it.
    if torch.isnan(value).any():
        if _DEBUG_ENABLED: # This print is now conditional
            print(f"{_RED}NaN detected in attention value tensor!{_RESET}")
        # value = torch.nan_to_num(value, nan=0.0)

    out = torch.matmul(weight, value)
    # Add NaN check for output
    if torch.isnan(out).any():
        if _DEBUG_ENABLED: # This print is now conditional
            print(f"{_RED}NaN detected after attention matmul!{_RESET}")
        # out = torch.nan_to_num(out, nan=0.0)
    return out

# --- Utility for Printing Stats (Keep as is) ---
# print_stats function is now debug_print_stats in the debug module


# --- Modules (PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward, TimestepEmbedder remain structurally similar) ---

class PositionalEncoding(nn.Module):
    # Standard Sinusoidal Positional Encoding (Keep as is)
    def __init__(self, dim_embed: int, max_len: int = 1024, drop_prob: float = 0.0) -> None:
        super(PositionalEncoding, self).__init__()
        assert dim_embed % 2 == 0
        self.dim_embed = dim_embed
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2).float() # Ensure float for division
        div_term = torch.exp(dim_pair * (-math.log(1024.0) / dim_embed))

        pe = torch.zeros(1, max_len, dim_embed) # Start with batch dim
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe, persistent=False)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor):
        # x shape: (Batch, SeqLen, Dim)
        debug_print_stats("PosEnc Input", x)
        pe_to_add = self.pe[:, :x.size(1)]
        debug_print_stats("PosEnc PE Added", pe_to_add)
        x = x + pe_to_add
        debug_print_stats("PosEnc After Add", x)
        x_drop = self.dropout(x)
        debug_print_stats("PosEnc Output (After Dropout)", x_drop)
        return x_drop


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()
        assert dim_embed % num_heads == 0
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        self.qkv = nn.Linear(dim_embed, dim_embed * 3, bias=True)
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout_attn = nn.Dropout(drop_prob) # Consider reducing or removing if instability persists
        self.dropout_output = nn.Dropout(drop_prob) # Consider reducing or removing

    def forward(self, x: Tensor, y: Tensor, mask: Tensor = None) -> Tensor:
        # Assuming self-attention (x=y) based on how it's used in the block
        debug_print_stats("MHA Input x", x)

        B, N, C = x.shape
        qkv_x = self.qkv(x)
        debug_print_stats("MHA QKV Proj", qkv_x)
        qkv_x = qkv_x.reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x.unbind(0)
        debug_print_stats("MHA Q", q)
        debug_print_stats("MHA K", k)
        debug_print_stats("MHA V", v)

        attn_output = attention(q, k, v, mask)
        debug_print_stats("MHA Attention Output Raw", attn_output)

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        debug_print_stats("MHA Attention Output Reshaped", attn_output)
        attn_output = self.dropout_attn(attn_output) # Optional dropout on attention scores
        debug_print_stats("MHA Attention Output After Dropout", attn_output)

        out = self.output(attn_output)
        debug_print_stats("MHA Final Proj Output", out)
        out = self.dropout_output(out) # Dropout before residual connection
        debug_print_stats("MHA Output (After Dropout)", out)
        return out

class PositionwiseFeedForward(nn.Module):
    # Keep as is
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float, bias: bool = True) -> None:
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn, bias=bias),      # Index 0
            nn.SiLU(),                                     # Index 1
            nn.Dropout(drop_prob),                         # Index 2 # Consider reducing or removing
            nn.Linear(dim_pffn, dim_embed, bias=bias),     # Index 3
            nn.Dropout(drop_prob),                         # Index 4 # Consider reducing or removing
        )

    def forward(self, x: Tensor) -> Tensor:
        debug_print_stats("PFFN Input", x)
        output = self.pffn(x)
        debug_print_stats("PFFN Output", output)
        return output

class TimestepEmbedder(nn.Module):
    # Keep as is
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.init_weights()

    def init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000): # Changed max_period to 10000 like UViT
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
        debug_print_stats("TimestepEmbedder Input t", t)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        debug_print_stats("TimestepEmbedder Freq Embed", t_freq)
        t_emb = self.mlp(t_freq)
        debug_print_stats("TimestepEmbedder Output MLP (t_emb)", t_emb)
        return t_emb

# --- Normalization Options (Keep as is) ---
class LayerNormWrapper(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True): # Use affine parameters
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        debug_print_stats("Norm Input", x)
        out = self.norm(x)
        debug_print_stats("Norm Output", out)
        return out


# --- LayerScale Helper (Keep as is) ---
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor):
        debug_print_stats("LayerScale Input", x)
        debug_print_stats("LayerScale Gamma", self.gamma)
        gamma_unsqueezed = self.gamma.unsqueeze(0).unsqueeze(0) if x.ndim == 3 else self.gamma
        out = x.mul_(gamma_unsqueezed) if self.inplace else x * gamma_unsqueezed
        debug_print_stats("LayerScale Output", out)
        return out


# --- MODIFIED Transformer Block ---

class TransformerBlock(nn.Module):
    '''
    A Transformer Block inspired by DiT, with AdaLN modulation applied *before* Norm,
    and LayerScale applied to residual paths. Gating is removed.
    MODIFIED: Can optionally accept and integrate a skip connection.
    '''
    def __init__(self, num_heads, dim_embed, mlp_ratio=4.0, drop_prob=0.1, norm_eps=1e-6,
                 ls_init_value=1e-5, use_skip_connection=False)-> None: # Added use_skip_connection
        super().__init__()
        self.use_skip_connection = use_skip_connection
        # Use LayerNorm with affine=True, modulation happens before norm
        self.norm1 = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=True)
        self.norm2 = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=True)

        self.self_atten = MultiHeadAttention(num_heads=num_heads, dim_embed=dim_embed, drop_prob=drop_prob)
        dim_pwff = int(dim_embed * mlp_ratio)
        # Ensure PositionwiseFeedForward matches checkpoint (using 'pffn' sequential)
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)

        # LayerScale
        self.ls1 = LayerScale(dim_embed, init_values=ls_init_value)
        self.ls2 = LayerScale(dim_embed, init_values=ls_init_value)

        # adaLN_modulation: projects time embedding to 4 values (shift/scale for Attn, shift/scale for FFN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 4 * dim_embed, bias=True) # Output 4 values
        )
        # ** Crucial: Zero-init the final layer for AdaLN-Zero style **
        # Will be handled in Transformer.initialize_weights

        # --- NEW: Skip Connection Layer ---
        if self.use_skip_connection:
            self.skip_linear = nn.Linear(dim_embed * 2, dim_embed, bias=True)
            # Initialization will be handled in Transformer.initialize_weights
        else:
            self.skip_linear = None


    def forward(self, x: Tensor, c: Tensor, skip: Optional[Tensor] = None): # c is the conditional embedding, Added skip
        debug_print_stats("Block Input x", x)
        debug_print_stats("Block Input c", c)
        if skip is not None:
             debug_print_stats("Block Input skip", skip)

        # --- NEW: Integrate Skip Connection (if applicable) ---
        if self.use_skip_connection and self.skip_linear is not None and skip is not None:
             if _DEBUG_ENABLED:
                 print("--- Block: Integrating Skip Connection ---")
             debug_print_stats("Block x before skip concat", x)
             debug_print_stats("Block skip before skip concat", skip)
             # Ensure shapes match except for feature dim (should be handled by concat)
             assert x.shape[0] == skip.shape[0] and x.shape[1] == skip.shape[1], \
                 f"Shape mismatch for skip connection: x {x.shape}, skip {skip.shape}"
             x_concat = torch.cat([x, skip], dim=-1)
             debug_print_stats("Block Concat(x, skip)", x_concat)
             x = self.skip_linear(x_concat)
             debug_print_stats("Block x after skip_linear", x)
        elif self.use_skip_connection and skip is None:
             if _DEBUG_ENABLED:
                 print(f"{_RED}WARNING: Block expects skip connection but received None.{_RESET}")
        # --- End NEW ---


        # Get modulation parameters
        mod_params = self.adaLN_modulation(c)
        debug_print_stats("Block adaLN Raw Output", mod_params)
        # Expecting shift_attn, scale_attn, shift_mlp, scale_mlp
        shift_attn, scale_attn, shift_mlp, scale_mlp = mod_params.chunk(4, dim=1)

        # --- Print Modulation Parameter Stats ---
        debug_print_stats("Block shift_attn", shift_attn)
        debug_print_stats("Block scale_attn", scale_attn)
        debug_print_stats("Block shift_mlp", shift_mlp)
        debug_print_stats("Block scale_mlp", scale_mlp)
        # ---

        # Self-Attention path: Modulate -> Norm -> Attention -> LayerScale -> Residual
        residual = x
        if _DEBUG_ENABLED: # Conditional print
            print("--- Block: Entering Self-Attention Path ---")
        # Apply AdaLN modulation (scale and shift) *before* norm
        x_mod_attn = x * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        debug_print_stats("Block Modulated Attn Input", x_mod_attn)

        attn_output = self.self_atten(self.norm1(x_mod_attn), self.norm1(x_mod_attn)) # Norm after modulation
        debug_print_stats("Block Attn Raw Output", attn_output)

        # Apply LayerScale to the attention output before adding residual
        scaled_attn_output = self.ls1(attn_output)
        debug_print_stats("Block LayerScaled Attn Output", scaled_attn_output)
        x = residual + scaled_attn_output
        debug_print_stats("Block After Attn Residual Add", x)
        if _DEBUG_ENABLED: # Conditional print
            print("--- Block: Exiting Self-Attention Path ---")

        # Feed-Forward path: Modulate -> Norm -> FFN -> LayerScale -> Residual
        residual = x
        if _DEBUG_ENABLED: # Conditional print
            print("--- Block: Entering Feed-Forward Path ---")
        # Apply AdaLN modulation (scale and shift) *before* norm
        x_mod_mlp = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        debug_print_stats("Block Modulated FFN Input", x_mod_mlp)

        ffn_output = self.feed_forward(self.norm2(x_mod_mlp)) # Norm after modulation
        debug_print_stats("Block FFN Raw Output", ffn_output)

        # Apply LayerScale to the FFN output before adding residual
        scaled_ffn_output = self.ls2(ffn_output)
        debug_print_stats("Block LayerScaled FFN Output", scaled_ffn_output)
        x = residual + scaled_ffn_output
        debug_print_stats("Block After FFN Residual Add", x)
        if _DEBUG_ENABLED: # Conditional print
            print("--- Block: Exiting Feed-Forward Path ---")

        debug_print_stats("Block Final Output x", x)
        return x

# --- Revised Final Layer (Keep as is) ---

class FinalLayer(nn.Module):
    """
    The final layer applies AdaLN modulation, normalization, and projects to output dimension.
    Follows the DiT-style block structure.
    """
    def __init__(self, dim_embed, out_channels, norm_eps=1e-6):
        super().__init__()
        # Use LayerNorm with affine=True
        self.norm_final = LayerNormWrapper(dim_embed, eps=norm_eps, elementwise_affine=True)

        # Modulation layer for shift/scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 2 * dim_embed, bias=True) # Output 2 values (shift, scale)
        )
        # Final linear projection
        self.linear = nn.Linear(dim_embed, out_channels, bias=True)
        # ** Crucial: Zero-init the final layers **
        # Will be handled in Transformer.initialize_weights

    def forward(self, x, c):
        debug_print_stats("FinalLayer Input x", x)
        debug_print_stats("FinalLayer Input c", c)

        mod_params = self.adaLN_modulation(c)
        debug_print_stats("FinalLayer adaLN Raw Output", mod_params)
        shift, scale = mod_params.chunk(2, dim=1)

        # --- Print Modulation Parameter Stats ---
        debug_print_stats("FinalLayer shift", shift)
        debug_print_stats("FinalLayer scale", scale)
        # ---

        # Apply AdaLN modulation *before* norm
        x_mod = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        debug_print_stats("FinalLayer After Modulate", x_mod)

        x_norm = self.norm_final(x_mod) # Norm after modulation
        debug_print_stats("FinalLayer After Norm", x_norm)

        x_proj = self.linear(x_norm) # Final projection
        debug_print_stats("FinalLayer Output (After Linear)", x_proj)
        return x_proj


# --- MODIFIED Main Transformer Model ---

class Transformer(nn.Module):
    """
    Transformer-based diffusion model using Patching, revised Transformer Blocks,
    and revised Final Layer for improved stability.
    MODIFIED: Incorporates a U-Net like structure with skip connections.
    """
    def __init__(
            self,
            img_resolution=32,
            in_channels=3,
            out_channels=3,
            patch_size=4,
            dim_embed=768,
            num_heads=12,
            depth=12,
            mlp_ratio=4.0,
            drop_prob=0.1, # Consider reducing dropout if needed
            learn_pe=True,
            # norm_type="layernorm", # Removed as it's fixed in block
            norm_eps=1e-6,
            ls_init_value=1e-5, # LayerScale init value
            t_embed_dim=256
            ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.depth = depth
        self.learn_pe = learn_pe
        # norm_type argument kept for potential future flexibility, but block forces LayerNorm now.

        # --- 1. Patching ---
        self.patch_embed = nn.Conv2d(
            in_channels, dim_embed,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True
        )
        self.num_patches = (img_resolution // self.patch_size) ** 2

        # --- 2. Positional and Timestep Embedding ---
        if learn_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim_embed))
        else:
            self.pos_embed_module = PositionalEncoding(dim_embed, max_len=self.num_patches, drop_prob=0.0)

        self.t_embedder = TimestepEmbedder(hidden_size=dim_embed, frequency_embedding_size=t_embed_dim)

        # --- 3. Transformer Blocks (MODIFIED: U-Net Structure) ---
        in_depth = depth // 2
        out_depth = depth - in_depth
        if _DEBUG_ENABLED:
            print(f"Creating U-Net structure: {in_depth} in_blocks, {out_depth} out_blocks")

        # --- Encoder Blocks ---
        self.in_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                norm_eps=norm_eps,
                ls_init_value=ls_init_value,
                use_skip_connection=False # Encoder blocks don't use skips
            )
            for _ in range(in_depth)
        ])

        # --- Decoder Blocks ---
        self.out_blocks = nn.ModuleList([
            TransformerBlock(
                num_heads=num_heads,
                dim_embed=dim_embed,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                norm_eps=norm_eps,
                ls_init_value=ls_init_value,
                use_skip_connection=True # Decoder blocks use skips
            )
            for _ in range(out_depth)
        ])

        # --- 4. Final Layer (Revised - Keep as is) ---
        self.final_layer = FinalLayer(
            dim_embed=dim_embed,
            out_channels=dim_embed, # Still output embed_dim for unpatching
            norm_eps=norm_eps
        )

        # --- 5. Unpatching (Output Projection - Keep as is) ---
        self.output_proj = nn.ConvTranspose2d(
            dim_embed,
            self.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Initialize weights
        self.initialize_weights(ls_init_value) # Pass ls_init_value
        if _DEBUG_ENABLED: # Conditional print for parameter count
            print(f"Transformer initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters")


    def initialize_weights(self, ls_init_value=1e-5): # Added ls_init_value
        # Initialize patch_embed like a linear layer
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize positional embedding if learnable
        if self.learn_pe:
            nn.init.normal_(self.pos_embed, std=.02)

        # --- MODIFIED: Initialize transformer blocks (in and out) ---
        # Initialize encoder blocks
        for block in self.in_blocks:
            # Attention projections (QKV and output)
            if hasattr(block.self_atten, 'qkv'):
                 nn.init.xavier_uniform_(block.self_atten.qkv.weight)
                 if block.self_atten.qkv.bias is not None: nn.init.constant_(block.self_atten.qkv.bias, 0)
            nn.init.xavier_uniform_(block.self_atten.output.weight)
            if block.self_atten.output.bias is not None: nn.init.constant_(block.self_atten.output.bias, 0)

            # FeedForward layers (using nn.Sequential)
            if hasattr(block.feed_forward, 'pffn'):
                 nn.init.xavier_uniform_(block.feed_forward.pffn[0].weight)
                 if block.feed_forward.pffn[0].bias is not None: nn.init.constant_(block.feed_forward.pffn[0].bias, 0)
                 nn.init.xavier_uniform_(block.feed_forward.pffn[3].weight)
                 if block.feed_forward.pffn[3].bias is not None: nn.init.constant_(block.feed_forward.pffn[3].bias, 0)

            # Zero-out the adaLN modulation output layer
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Initialize LayerScale gamma parameters
            nn.init.constant_(block.ls1.gamma, ls_init_value)
            nn.init.constant_(block.ls2.gamma, ls_init_value)

        # Initialize decoder blocks
        for block in self.out_blocks:
             # Standard Attention/FFN/AdaLN/LayerScale init (same as in_blocks)
            if hasattr(block.self_atten, 'qkv'):
                 nn.init.xavier_uniform_(block.self_atten.qkv.weight)
                 if block.self_atten.qkv.bias is not None: nn.init.constant_(block.self_atten.qkv.bias, 0)
            nn.init.xavier_uniform_(block.self_atten.output.weight)
            if block.self_atten.output.bias is not None: nn.init.constant_(block.self_atten.output.bias, 0)
            if hasattr(block.feed_forward, 'pffn'):
                 nn.init.xavier_uniform_(block.feed_forward.pffn[0].weight)
                 if block.feed_forward.pffn[0].bias is not None: nn.init.constant_(block.feed_forward.pffn[0].bias, 0)
                 nn.init.xavier_uniform_(block.feed_forward.pffn[3].weight)
                 if block.feed_forward.pffn[3].bias is not None: nn.init.constant_(block.feed_forward.pffn[3].bias, 0)
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.ls1.gamma, ls_init_value)
            nn.init.constant_(block.ls2.gamma, ls_init_value)

            # --- NEW: Initialize the skip connection linear layer ---
            if block.skip_linear is not None:
                # Replace the line above with this one:
                nn.init.xavier_uniform_(block.skip_linear.weight) # Use Xavier initialization
                if block.skip_linear.bias is not None:
                    nn.init.constant_(block.skip_linear.bias, 0)
        # --- End MODIFIED block init ---


        # Initialize final layer
        # *** Crucial: Zero-out the final adaLN modulation output layer ***
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # *** Crucial: Zero-out the final linear projection layer ***
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize output projection (unpatching layer)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)


    def unpatchify(self, x):
        """
        x: (B, N, C) N = num_patches, C = embed_dim
        imgs: (B, C_img, H, W)
        (Remains the same)
        """
        debug_print_stats("Unpatchify Input x", x)
        B, N, C = x.shape
        Hp = Wp = int(N**0.5) # Assume square patch grid
        assert Hp * Wp == N, f"Number of patches {N} is not a perfect square."
        # assert C == self.dim_embed, f"Input C ({C}) != dim_embed ({self.dim_embed})" # C might change if final layer outputs diff dim
        assert x.shape[-1] == self.final_layer.linear.out_features, \
               f"Input C ({x.shape[-1]}) to unpatchify != final_layer output dim ({self.final_layer.linear.out_features})"


        # Adapt view based on the output dimension of the final layer
        # x = x.transpose(1, 2).view(B, self.dim_embed, Hp, Wp)
        x = x.transpose(1, 2).view(B, self.final_layer.linear.out_features, Hp, Wp)
        debug_print_stats("Unpatchify Reshaped x", x)

        imgs = self.output_proj(x)
        debug_print_stats("Unpatchify Output imgs", imgs)
        return imgs

    def forward(self, x, t, class_labels=None): # Class labels still unused
        """
        Forward pass of Transformer using revised blocks AND U-Net structure.
        """
        if _DEBUG_ENABLED: # Conditional print
            print("\n--- Transformer Forward Start ---")
        debug_print_stats("Input x", x)
        debug_print_stats("Input t", t)

        B, C_img, H, W = x.shape
        if not (H == self.img_resolution and W == self.img_resolution):
            if _DEBUG_ENABLED: # Conditional print
                # Use red color for resolution warning
                print(f"{_RED}WARNING: Input image resolution ({H}x{W}) mismatch with expected ({self.img_resolution}x{self.img_resolution}){_RESET}")

        # --- 1. Patch Embedding ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 1: Patch Embedding ---")
        x = self.patch_embed(x) # Shape: B, C, H/P, W/P
        debug_print_stats("After Patch Embed (Conv2d)", x)
        x = x.flatten(2).transpose(1, 2) # Shape: B, N, C
        debug_print_stats("After Patch Embed (Reshaped)", x)


        # --- 2. Add Positional Encoding ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 2: Positional Encoding ---")
        if self.learn_pe:
            pe_to_add = self.pos_embed # Shape: 1, N, C
            debug_print_stats("Learnable PE", pe_to_add)
            x = x + pe_to_add
            debug_print_stats("After Learnable PE Add", x)
        else:
            x = self.pos_embed_module(x) # Applies PE internally
            debug_print_stats("After Sinusoidal PE Module", x)

        # --- 3. Get Timestep Embedding ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 3: Timestep Embedding ---")
        t_emb = self.t_embedder(t) # Shape: B, C
        debug_print_stats("Timestep Embedding (t_emb)", t_emb)

        # --- 4. Apply Transformer Blocks (MODIFIED: U-Net Flow) ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 4: Transformer Blocks (U-Net Flow) ---")

        skips = []
        # --- Encoder Path ---
        if _DEBUG_ENABLED: print("--- Entering Encoder Blocks ---")
        for i, block in enumerate(self.in_blocks):
            if _DEBUG_ENABLED: print(f"--- Encoder Block {i+1}/{len(self.in_blocks)} ---")
            x = block(x, t_emb, skip=None) # No skip connection in encoder
            debug_print_stats(f"Output from Encoder Block {i+1}", x)
            skips.append(x)
            # Instability Check
            if torch.isnan(x).any() or torch.isinf(x).any():
                 if _DEBUG_ENABLED: print(f"{_RED}!!! Instability detected after Encoder Block {i+1} !!!{_RESET}")
                 raise ValueError(f"NaN or Inf detected after Encoder Block {i+1}")
        if _DEBUG_ENABLED: print("--- Exiting Encoder Blocks ---")

        # --- Bottleneck --- (Implicitly the transition)

        # --- Decoder Path ---
        if _DEBUG_ENABLED: print("--- Entering Decoder Blocks ---")
        for i, block in enumerate(self.out_blocks):
             if _DEBUG_ENABLED: print(f"--- Decoder Block {i+1}/{len(self.out_blocks)} ---")
             skip_connection = skips.pop() # Get corresponding skip from encoder (LIFO)
             x = block(x, t_emb, skip=skip_connection) # Pass skip connection
             debug_print_stats(f"Output from Decoder Block {i+1}", x)
             # Instability Check
             if torch.isnan(x).any() or torch.isinf(x).any():
                 if _DEBUG_ENABLED: print(f"{_RED}!!! Instability detected after Decoder Block {i+1} !!!{_RESET}")
                 raise ValueError(f"NaN or Inf detected after Decoder Block {i+1}")
        if _DEBUG_ENABLED: print("--- Exiting Decoder Blocks ---")
        assert len(skips) == 0, "Mismatch in skip connections!"


        # --- 5. Apply Final Layer (Revised - Keep as is) ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 5: Final Layer ---")
        x = self.final_layer(x, t_emb) # Pass t_emb for final modulation
        debug_print_stats("After Final Layer", x)
        # Check for instability *after* the final layer
        if torch.isnan(x).any() or torch.isinf(x).any():
            if _DEBUG_ENABLED: # Conditional print
                # Use red color for instability warning
                print(f"{_RED}!!! Instability detected after Final Layer !!!{_RESET}")
            # Keep the raise ValueError
            raise ValueError(f"NaN or Inf detected after Final Layer")

        # --- 6. Unpatch (Project back to Image - Keep as is, but check logic) ---
        if _DEBUG_ENABLED: # Conditional print
            print("--- Step 6: Unpatchify ---")
        x = self.unpatchify(x)
        debug_print_stats("Final Output Image", x)
        # Check for instability *after* unpatchify
        if torch.isnan(x).any() or torch.isinf(x).any():
            if _DEBUG_ENABLED: # Conditional print
                # Use red color for instability warning
                print(f"{_RED}!!! Instability detected after Unpatchify !!!{_RESET}")
            # Keep the raise ValueError
            raise ValueError(f"NaN or Inf detected after Unpatchify")

        # Check final output shape
        expected_shape = (B, self.out_channels, self.img_resolution, self.img_resolution)
        if x.shape != expected_shape:
            if _DEBUG_ENABLED: # Conditional print
                # Use red color for shape mismatch warning
                print(f"{_RED}WARNING: Final output shape {x.shape} doesn't match expected {expected_shape}{_RESET}")

        if _DEBUG_ENABLED: # Conditional print
            print("--- Transformer Forward End ---\n")
        return x

# --- Example Usage (Keep as is) ---
# set_debug_enabled(False) # Default: No verbose logs
# model = Transformer(depth=12, dim_embed=768, patch_size=4, img_resolution=32)
# print(f"Model created with U-Net structure.")
# dummy_input = torch.randn(2, 3, 32, 32)
# dummy_timestep = torch.randint(0, 1000, (2,))
# try:
#     # set_debug_enabled(True) # Uncomment to debug forward pass
#     output = model(dummy_input, dummy_timestep)
#     print(f"Run completed successfully. Output shape: {output.shape}")
# except ValueError as e:
#     print(f"{_RED}Run terminated due to instability: {e}{_RESET}")
# except Exception as e:
#      print(f"{_RED}Run terminated due to error: {e}{_RESET}")
# set_debug_enabled(False) # Ensure debug is off after example