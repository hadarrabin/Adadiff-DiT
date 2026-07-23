import copy
import gc
import json
import math
import os
import time
import h5py
import matplotlib

matplotlib.use("Agg")  # Prevent display errors on cluster/SSH environments
import matplotlib.pyplot as plt
import numpy as np
import skimage.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CORE ARCHITECTURE & SCHEDULER
# =============================================================================
def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    args = timesteps.unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class AdaDiffScheduler:
    def __init__(
        self,
        step_size_k: int = 125,
        total_steps_T: int = 1000,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        self.k = step_size_k
        self.T = total_steps_T
        self.num_stages = total_steps_T // step_size_k
        self.stages = np.arange(0, self.T + 1, self.k)

        self.alpha_bar = []
        for t in self.stages:
            integrated_beta = beta_min * (t / self.T) + 0.5 * (beta_max - beta_min) * (
                (t / self.T) ** 2
            )
            self.alpha_bar.append(np.exp(-integrated_beta))
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)

    def add_noise(self, x_0: torch.Tensor, stage_idx: int):
        alpha_t_bar = self.alpha_bar[stage_idx].to(x_0.device)
        noise = torch.randn_like(x_0)
        return (
            torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1.0 - alpha_t_bar) * noise,
            noise,
        )


class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.qkv_project = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_project = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mod = self.adaLN_modulation(c).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(mod, 6, dim=-1)

        norm_x1 = self.ln1(x) * (1.0 + gamma1) + beta1
        qkv = self.qkv_project(norm_x1).reshape(
            B, T, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)
        attn_out = self.out_project(attn_out)
        x = x + alpha1 * attn_out

        norm_x2 = self.ln2(x) * (1.0 + gamma2) + beta2
        mlp_out = self.mlp(norm_x2)
        x = x + alpha2 * mlp_out
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: int,
        patch_size: int = 2,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patchify = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, num_heads) for _ in range(depth)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, (patch_size**2) * in_channels)

    def forward(self, x: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_tokens = self.patchify(x).flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.pos_embed

        t_sin = get_timestep_embedding(t_tensor, embedding_dim=self.hidden_dim)
        c = self.time_mlp(t_sin)

        for block in self.blocks:
            x_tokens = block(x_tokens, c)

        x_tokens = self.final_ln(x_tokens)
        decoded = self.final_linear(x_tokens)

        p = self.patch_size
        h_patches, w_patches = H // p, W // p
        x_out = decoded.view(B, h_patches, w_patches, p, p, C)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x_out


def load_trained_generator(checkpoint_path, device="cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint missing at '{checkpoint_path}'!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    else:
        state_dict = checkpoint

    hidden_dim, in_channels, patch_size, _ = state_dict["patchify.weight"].shape
    num_patches = state_dict["pos_embed"].shape[1]
    input_size = int(math.sqrt(num_patches)) * patch_size

    model = DiffusionTransformer(
        in_channels=in_channels,
        input_size=input_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        depth=12,
        num_heads=6,
    )
    model.load_state_dict(state_dict)
    return model.to(device)


# =============================================================================
# 2. OPERATORS, DATA CONSISTENCY & ADAPTATION PIPELINE
# =============================================================================
def get_variable_density_mask(H, W, R=4, center_fraction=0.08, device="cuda"):
    mask = torch.zeros((H, W), device=device)
    num_low_freq_h = int(H * center_fraction)
    num_low_freq_w = int(W * center_fraction)
    ch_s, ch_e = H // 2 - num_low_freq_h // 2, H // 2 + num_low_freq_h // 2
    cw_s, cw_e = W // 2 - num_low_freq_w // 2, W // 2 + num_low_freq_w // 2
    mask[ch_s:ch_e, cw_s:cw_e] = 1.0

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    dist = torch.sqrt(
        ((y_coords - H // 2) / (H // 2)) ** 2
        + ((x_coords - W // 2) / (W // 2)) ** 2
    )
    prob_mask = torch.exp(-dist * (R * 0.75))
    random_matrix = torch.rand((H, W), device=device)
    mask[random_matrix < prob_mask] = 1.0
    mask[ch_s:ch_e, cw_s:cw_e] = 1.0
    return mask.unsqueeze(0).unsqueeze(0)


def get_imaging_operator(x_true, R=4):
    kspace = torch.fft.fftshift(
        torch.fft.fft2(x_true, dim=(-2, -1)), dim=(-2, -1)
    )
    mask_2d = get_variable_density_mask(
        x_true.shape[2], x_true.shape[3], R=R, device=x_true.device
    )
    y = kspace * mask_2d
    return y, mask_2d


def adjoint_operator(y):
    return torch.fft.ifft2(torch.fft.ifftshift(y, dim=(-2, -1)), dim=(-2, -1))


def apply_data_consistency(x_pred, y_measured, mask, eta=1.0):
    kspace_pred = torch.fft.fftshift(
        torch.fft.fft2(x_pred, dim=(-2, -1)), dim=(-2, -1)
    )
    kspace_dc = (
        kspace_pred * (1.0 - mask)
        + (kspace_pred - eta * (kspace_pred - y_measured)) * mask
    )
    return adjoint_operator(kspace_dc).real.float()


def get_noised_measured_kspace(y_clean, stage_idx, scheduler):
    if stage_idx == 0:
        return y_clean
    spatial_clean = adjoint_operator(y_clean).real
    spatial_noisy, _ = scheduler.add_noise(spatial_clean, stage_idx)
    return torch.fft.fftshift(
        torch.fft.fft2(spatial_noisy.to(torch.complex64), dim=(-2, -1)),
        dim=(-2, -1),
    )


def run_prior_adaptation(
    base_model,
    scheduler,
    y,
    mask,
    J,
    lr,
    eta=1.0,
    param_scope="adaln_only",
    device="cuda",
):
    netG = copy.deepcopy(base_model).to(device)
    netG.eval()

    start_time = time.time()
    x_current = adjoint_operator(y).real
    x_current, _ = scheduler.add_noise(
        x_current, stage_idx=scheduler.num_stages - 1
    )

    x_input_tta = None
    t_final = torch.full(
        (1,), scheduler.stages[1], dtype=torch.long, device=device
    )

    with torch.no_grad():
        for stage_idx in reversed(range(scheduler.num_stages - 1)):
            if stage_idx == 0:
                x_input_tta = x_current.detach().clone()

            t_parent = scheduler.stages[stage_idx + 1]
            t_tensor = torch.full((1,), t_parent, dtype=torch.long, device=device)

            x_pred = netG(x_current, t_tensor)
            y_noised_kspace = get_noised_measured_kspace(y, stage_idx, scheduler)
            x_current = apply_data_consistency(x_pred, y_noised_kspace, mask, eta=eta)

    netG.train()
    if param_scope == "adaln_only":
        for name, param in netG.named_parameters():
            param.requires_grad = (
                True
                if (
                    "adaLN" in name
                    or "ln" in name
                    or "norm" in name
                    or "pos_embed" in name
                )
                else False
            )
    else:
        for param in netG.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, netG.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_x_0_pred = None

    for step in range(1, J + 1):
        optimizer.zero_grad()
        x_recon = netG(x_input_tta, t_final)
        kspace_recon = torch.fft.fftshift(
            torch.fft.fft2(x_recon.to(torch.complex64), dim=(-2, -1)), dim=(-2, -1)
        )

        masked_recon = kspace_recon * mask
        masked_target = y * mask
        loss = torch.mean(
            torch.abs(masked_recon.real - masked_target.real)
        ) + torch.mean(torch.abs(masked_recon.imag - masked_target.imag))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_0_pred = x_recon.detach().clone()

    netG.eval()
    with torch.no_grad():
        if best_x_0_pred is None:
            best_x_0_pred = netG(x_input_tta, t_final)
        final_recon = apply_data_consistency(best_x_0_pred, y, mask, eta=eta)

    execution_time = time.time() - start_time
    del netG, optimizer
    torch.cuda.empty_cache()

    return torch.clamp(final_recon, -1.0, 1.0), execution_time


# =============================================================================
# 3. SPATIAL CENTERING & METRIC EVALUATION
# =============================================================================
def get_centered_spatial_magnitude(t_img):
    """Extracts root-sum-of-squares spatial magnitude with smart auto-centering."""
    arr = torch.clamp((t_img.squeeze() + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()
    if arr.ndim == 3:
        arr = np.sqrt(np.sum(arr**2, axis=0))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # 🌟 SMART AUTO-CENTERING: Only shift if the energy is concentrated in the corners
    H, W = arr.shape
    center_energy = np.mean(arr[H//4:3*H//4, W//4:3*W//4])
    corner_energy = (np.mean(arr[:H//4, :W//4]) + np.mean(arr[:H//4, 3*W//4:]) + 
                     np.mean(arr[3*H//4:, :W//4]) + np.mean(arr[3*H//4:, 3*W//4:])) / 4.0
    
    if corner_energy > center_energy:
        arr = np.fft.fftshift(arr, axes=(-2, -1))
        
    return arr


def compute_metrics(recon_img, gt_img):
    """Computes PSNR and SSIM on centered magnitude images."""
    psnr = float(
        skimage.metrics.peak_signal_noise_ratio(gt_img, recon_img, data_range=1.0)
    )
    ssim = float(
        skimage.metrics.structural_similarity(
            gt_img, recon_img, data_range=1.0, channel_axis=None
        )
    )
    return psnr, ssim


def load_and_preprocess_volume(
    file_path, target_channels=10, target_size=256
):
    if file_path.endswith((".h5", ".hdf5")):
        with h5py.File(file_path, "r") as hf:
            possible_keys = [
                "reconstruction_rss",
                "reconstruction_esc",
                "image",
                "data",
                "kspace",
                "target",
            ]
            data_key = None
            for k in possible_keys:
                if k in hf.keys():
                    data_key = k
                    break
            if not data_key:
                data_key = list(hf.keys())[0]
            matrix = hf[data_key][()]
    elif file_path.endswith(".npy"):
        matrix = np.load(file_path)
    elif file_path.endswith(".npz"):
        npz = np.load(file_path)
        matrix = npz[npz.files[0]]
    elif file_path.endswith(".pt"):
        matrix = torch.load(file_path, map_location="cpu").numpy()

    if np.iscomplexobj(matrix):
        matrix = np.real(matrix)

    tensor_data = torch.tensor(matrix, dtype=torch.float32)

    if tensor_data.ndim == 2:
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
    elif tensor_data.ndim == 3:
        if (
            tensor_data.shape[-1] < tensor_data.shape[0]
            and tensor_data.shape[-1] <= 32
        ):
            tensor_data = tensor_data.permute(2, 0, 1)
        tensor_data = tensor_data.unsqueeze(1)
    elif tensor_data.ndim == 4:
        if (
            tensor_data.shape[-1] < tensor_data.shape[1]
            and tensor_data.shape[-1] <= 32
        ):
            tensor_data = tensor_data.permute(0, 3, 1, 2)

    B, C, H, W = tensor_data.shape
    if C > target_channels:
        tensor_data = tensor_data[:, :target_channels, :, :]
    elif C < target_channels:
        pad_channels = target_channels - C
        tensor_data = F.pad(
            tensor_data, (0, 0, 0, 0, 0, pad_channels), mode="constant", value=0
        )

    B, C, H, W = tensor_data.shape
    if H != target_size or W != target_size:
        h_start = max(0, (H - target_size) // 2)
        h_end = h_start + min(H, target_size)
        w_start = max(0, (W - target_size) // 2)
        w_end = w_start + min(W, target_size)

        cropped = tensor_data[:, :, h_start:h_end, w_start:w_end]

        pad_h = max(0, target_size - cropped.shape[2])
        pad_w = max(0, target_size - cropped.shape[3])
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bot = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            cropped = F.pad(
                cropped,
                (pad_left, pad_right, pad_top, pad_bot),
                mode="constant",
                value=0,
            )
        tensor_data = cropped

    return tensor_data


# =============================================================================
# 4. FIGURE GENERATION FUNCTION
# =============================================================================
def plot_and_save_figure(
    gt, zero_filled, recon, psnr, ssim, runtime, title_str, save_filename
):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Calculate absolute error map
    error_map = np.abs(gt - recon)

    axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(zero_filled, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Zero-Filled Input", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(recon, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("DiT Reconstructed", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    im_err = axes[3].imshow(error_map, cmap="hot", vmin=0, vmax=0.15)
    axes[3].set_title("Absolute Error Map", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    cbar = fig.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("|Error|", fontsize=11)

    plt.suptitle(
        f"{title_str}\nPSNR: {psnr:.2f} dB | SSIM: {ssim*100:.2f}% | Runtime:"
        f" {runtime:.2f}s",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" 📸 Saved clean figure: {save_filename}")


# =============================================================================
# 5. MAIN EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Generating Paper Visualizations on: {str(device).upper()}")

    # Load Best Hyperparameters
    with open("best_hyperparameters_fastmri.json", "r") as f:
        best_hyperparams = json.load(f)

    CHECKPOINT_FILE = "/ugproj/sipl-prj10848/Idan/fastmri_checkpoints_sinusoidal/adadiff_fastmri_sinusoidal_latest.pt"
    base_model = load_trained_generator(
        checkpoint_path=CHECKPOINT_FILE, device=device
    )
    scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)

    test_base_dir = "/ugproj/sipl-prj10848/Idan/fastMRI_adadiff/test"
    contrasts = ["T1", "T2", "FLAIR"]

    # Locate one test file per contrast
    selected_samples = {}
    if os.path.exists(test_base_dir):
        for root, dirs, files in os.walk(test_base_dir):
            for f in sorted(files):
                if f.endswith((".h5", ".hdf5", ".npy", ".npz", ".pt")):
                    file_path = os.path.join(root, f)
                    search_string = (f + "_" + root).lower()
                    for c in contrasts:
                        if c not in selected_samples and c.lower() in search_string:
                            selected_samples[c] = file_path
                            break

    print("\n" + "=" * 60)
    print("🎯 SELECTED TEST VOLUMES FOR VISUALIZATION:")
    for c, path in selected_samples.items():
        print(f"   -> {c}: {path}")
    print("=" * 60)

    output_dir = "paper_figures_fastmri_new"
    os.makedirs(output_dir, exist_ok=True)

    acceleration_rates = [4, 8]

    for contrast, file_path in selected_samples.items():
        print(f"\n🧠 Processing Contrast: {contrast} ...")
        volume_data = load_and_preprocess_volume(
            file_path,
            target_channels=base_model.in_channels,
            target_size=base_model.input_size,
        )

        # 🎯 SELECT BEST SLICE (Highest energy ensures full brain anatomy instead of empty padding)
        num_slices = volume_data.shape[0]
        slice_energies = [torch.mean(torch.abs(volume_data[i])).item() for i in range(num_slices)]
        best_slice_idx = np.argmax(slice_energies)
        print(
            f"   📌 Volume contains {num_slices} slices. Selected highest-energy slice #{best_slice_idx}"
        )

        # 🎯 SLICE-LEVEL NORMALIZATION (Ensures vivid contrast in [-1, 1] range)
        x_true = volume_data[best_slice_idx : best_slice_idx + 1]
        min_val, max_val = x_true.min(), x_true.max()
        x_true = ((x_true - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0
        x_true = x_true.to(device)

        gt_centered = get_centered_spatial_magnitude(x_true)

        for R in acceleration_rates:
            y, mask = get_imaging_operator(x_true, R=R)
            zero_filled_centered = get_centered_spatial_magnitude(
                adjoint_operator(y).real.float()
            )

            # ---------------------------------------------------------------------
            # EXPERIMENT 1: Ours (Best Hyperparameters)
            # ---------------------------------------------------------------------
            best_cfg = best_hyperparams[f"R_{R}"]
            print(
                f"   ⚡ Running Ours (R={R}x) | J={best_cfg['J']}, lr={best_cfg['lr']},"
                f" eta={best_cfg['eta']}, scope={best_cfg['scope']}..."
            )

            recon_ours_tensor, time_ours = run_prior_adaptation(
                base_model,
                scheduler,
                y,
                mask,
                J=best_cfg["J"],
                lr=best_cfg["lr"],
                eta=best_cfg["eta"],
                param_scope=best_cfg["scope"],
                device=device,
            )
            recon_ours = get_centered_spatial_magnitude(recon_ours_tensor)
            psnr_ours, ssim_ours = compute_metrics(recon_ours, gt_centered)

            plot_and_save_figure(
                gt=gt_centered,
                zero_filled=zero_filled_centered,
                recon=recon_ours,
                psnr=psnr_ours,
                ssim=ssim_ours,
                runtime=time_ours,
                title_str=f"Ours (DiT Prior) - fastMRI {contrast} (R={R}x)",
                save_filename=os.path.join(
                    output_dir, f"figure_Ours_{contrast}_R{R}x.png"
                ),
            )

            # ---------------------------------------------------------------------
            # EXPERIMENT 2: Baseline Literature (J=1000)
            # ---------------------------------------------------------------------
            print(f"   🐢 Running Baseline (R={R}x) | J=1000, lr=1e-4, eta=0.8...")

            recon_base_tensor, time_base = run_prior_adaptation(
                base_model,
                scheduler,
                y,
                mask,
                J=1000,
                lr=1e-4,
                eta=0.8,
                param_scope="adaln_only",
                device=device,
            )
            recon_base = get_centered_spatial_magnitude(recon_base_tensor)
            psnr_base, ssim_base = compute_metrics(recon_base, gt_centered)

            plot_and_save_figure(
                gt=gt_centered,
                zero_filled=zero_filled_centered,
                recon=recon_base,
                psnr=psnr_base,
                ssim=ssim_base,
                runtime=time_base,
                title_str=(
                    f"AdaDiff Literature Baseline (J=1000) - fastMRI {contrast}"
                    f" (R={R}x)"
                ),
                save_filename=os.path.join(
                    output_dir, f"figure_Baseline_J1000_{contrast}_R{R}x.png"
                ),
            )

            del y, mask, recon_ours_tensor, recon_base_tensor
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 60)
    print(
        f"🎉 ALL FIGURES GENERATED SUCCESSFULLY! Saved to folder: '{output_dir}/'"
    )
    print("=" * 60)