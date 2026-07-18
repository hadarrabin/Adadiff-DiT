# models.py
import torch
import torch.nn as nn
import math

class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        """
        Upgraded DiT Block utilizing PyTorch's Hardware-Adaptive Scaled Dot-Product Attention
        to bypass quadratic memory overhead across different GPU generations.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Normalization layers
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Attention Linear Projections
        self.qkv_project = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_project = nn.Linear(hidden_dim, hidden_dim)

        # Pointwise Feedforward Component
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # adaLN-Zero Parameter Regression Engine
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)
        )

        # Zero-initialize the modulation network to force identity initialization
        nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: Input sequence tokens [Batch, Tokens, HiddenDim]
        c: Time conditioning vector [Batch, HiddenDim]
        """
        B, T, D = x.shape

        # Regress the 6 modulation configurations from condition vector
        mod = self.adaLN_modulation(c).unsqueeze(1) # [Batch, 1, HiddenDim * 6]
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(mod, 6, dim=-1)

        # First Residual Path: Hardware-Adaptive Scaled Dot-Product Attention
        norm_x1 = self.ln1(x) * (1.0 + gamma1) + beta1

        # Project and split into Q, K, V shapes: [B, T, 3, num_heads, head_dim]
        qkv = self.qkv_project(norm_x1).reshape(B, T, 3, self.num_heads, self.head_dim)
        # Permute to layout expected by scaled_dot_product_attention: [B, num_heads, T, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # FIXED: Removed the restrictive sdp_kernel flags to allow PyTorch to use
        # C++ Memory-Efficient math kernels natively optimized for T4 (sm_75) devices.
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Merge heads back into standard sequence layout: [B, T, HiddenDim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)
        attn_out = self.out_project(attn_out)

        x = x + alpha1 * attn_out

        # Second Residual Path: Pointwise Feedforward Network
        norm_x2 = self.ln2(x) * (1.0 + gamma2) + beta2
        mlp_out = self.mlp(norm_x2)
        x = x + alpha2 * mlp_out

        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, in_channels: int, input_size: int, patch_size: int = 2, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6):
        """
        Polymorphic DiT Core Backbone.
        Scales seamlessly across IXI and fastMRI spatial formats.
        """
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # 1. Patchify Operation (Converts spatial input dimensions into patch tokens)
        self.patchify = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Calculate grid sequence token count: T = (l / p)^2
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # 2. Continuous Time Embedding Network
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Stacked adaLN-Zero Transformer Blocks
        self.blocks = nn.ModuleList([AdaLNZeroBlock(hidden_dim, num_heads) for _ in range(depth)])

        # 4. Linear Decoder to return tensors back to their native spatial canvas shapes
        self.final_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, (patch_size ** 2) * in_channels)

        # Weight initialization
        nn.init.normal_(self.pos_embed, std=0.02)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.final_linear.weight, 0.0)
        nn.init.constant_(self.final_linear.bias, 0.0)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: Noisy tensor [Batch, Channels, H, W]
        t_emb: Time step embedded vector [Batch, HiddenDim]
        """
        B, C, H, W = x.shape

        # Patchify and add position embedding coordinates
        x_tokens = self.patchify(x).flatten(2).transpose(1, 2) # [B, Tokens, HiddenDim]
        x_tokens = x_tokens + self.pos_embed

        # Projects time condition vector
        c = self.time_mlp(t_emb)

        # Unroll tokens sequentially through the adaptive attention DiT blocks
        for block in self.blocks:
            x_tokens = block(x_tokens, c)

        # Decode tokens back into structural shape
        x_tokens = self.final_ln(x_tokens)
        decoded = self.final_linear(x_tokens) # [B, Tokens, P*P*C]

        # Rearrange tokens back to native multi-channel image layouts
        p = self.patch_size
        h_patches, w_patches = H // p, W // p
        x_out = decoded.view(B, h_patches, w_patches, p, p, C)

        # Corrected permute sequence dimensions
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

        return x_out