import copy
import gc
import json
import math
import os
import time
import h5py
import lpips
import matplotlib.pyplot as plt
import numpy as np
import skimage.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. CORE ARCHITECTURE & SCHEDULER
# =============================================================================
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    args = timesteps.unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class AdaDiffScheduler:
    def __init__(self, step_size_k: int = 125, total_steps_T: int = 1000, beta_min: float = 0.1, beta_max: float = 20.0):
        self.k = step_size_k
        self.T = total_steps_T
        self.num_stages = total_steps_T // step_size_k
        self.stages = np.arange(0, self.T + 1, self.k)

        self.alpha_bar = []
        for t in self.stages:
            integrated_beta = beta_min * (t / self.T) + 0.5 * (beta_max - beta_min) * ((t / self.T) ** 2)
            self.alpha_bar.append(np.exp(-integrated_beta))
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)

    def add_noise(self, x_0: torch.Tensor, stage_idx: int):
        alpha_t_bar = self.alpha_bar[stage_idx].to(x_0.device)
        noise = torch.randn_like(x_0)
        return torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1.0 - alpha_t_bar) * noise, noise

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
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mod = self.adaLN_modulation(c).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(mod, 6, dim=-1)

        norm_x1 = self.ln1(x) * (1.0 + gamma1) + beta1
        qkv = self.qkv_project(norm_x1).reshape(B, T, 3, self.num_heads, self.head_dim)
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
    def __init__(self, in_channels: int, input_size: int, patch_size: int = 2, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.patchify = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks = nn.ModuleList([AdaLNZeroBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, (patch_size ** 2) * in_channels)

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
    state_dict = checkpoint['generator_state_dict'] if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint else checkpoint

    hidden_dim, in_channels, patch_size, _ = state_dict['patchify.weight'].shape
    num_patches = state_dict['pos_embed'].shape[1]
    input_size = int(math.sqrt(num_patches)) * patch_size

    print(f"📦 Loaded fastMRI Generator Checkpoint:")
    print(f"   -> in_channels={in_channels} | input_size={input_size} | patch_size={patch_size} | hidden_dim={hidden_dim}")

    model = DiffusionTransformer(
        in_channels=in_channels, 
        input_size=input_size, 
        patch_size=patch_size, 
        hidden_dim=hidden_dim, 
        depth=12, 
        num_heads=6
    )
    model.load_state_dict(state_dict)
    return model.to(device)

# =============================================================================
# 2. OPERATORS & TTA ADAPTATION
# =============================================================================
def get_variable_density_mask(H, W, R=4, center_fraction=0.08, device="cuda"):
    mask = torch.zeros((H, W), device=device)
    num_low_freq_h = int(H * center_fraction)
    num_low_freq_w = int(W * center_fraction)
    ch_s, ch_e = H // 2 - num_low_freq_h // 2, H // 2 + num_low_freq_h // 2
    cw_s, cw_e = W // 2 - num_low_freq_w // 2, W // 2 + num_low_freq_w // 2
    mask[ch_s:ch_e, cw_s:cw_e] = 1.0
    
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    dist = torch.sqrt(((y_coords - H//2) / (H//2))**2 + ((x_coords - W//2) / (W//2))**2)
    prob_mask = torch.exp(-dist * (R * 0.75))
    random_matrix = torch.rand((H, W), device=device)
    mask[random_matrix < prob_mask] = 1.0
    mask[ch_s:ch_e, cw_s:cw_e] = 1.0
    return mask.unsqueeze(0).unsqueeze(0)

def get_imaging_operator(x_true, R=4):
    kspace = torch.fft.fftshift(torch.fft.fft2(x_true, dim=(-2, -1)), dim=(-2, -1))
    mask_2d = get_variable_density_mask(x_true.shape[2], x_true.shape[3], R=R, device=x_true.device)
    y = kspace * mask_2d
    return y, mask_2d

def adjoint_operator(y):
    return torch.fft.ifft2(torch.fft.ifftshift(y, dim=(-2, -1)), dim=(-2, -1))

def apply_data_consistency(x_pred, y_measured, mask, eta=1.0):
    kspace_pred = torch.fft.fftshift(torch.fft.fft2(x_pred, dim=(-2, -1)), dim=(-2, -1))
    kspace_dc = kspace_pred * (1.0 - mask) + (kspace_pred - eta * (kspace_pred - y_measured)) * mask
    return adjoint_operator(kspace_dc).real.float()

def get_noised_measured_kspace(y_clean, stage_idx, scheduler):
    if stage_idx == 0: return y_clean
    spatial_clean = adjoint_operator(y_clean).real
    spatial_noisy, _ = scheduler.add_noise(spatial_clean, stage_idx)
    return torch.fft.fftshift(torch.fft.fft2(spatial_noisy.to(torch.complex64), dim=(-2, -1)), dim=(-2, -1))

def run_prior_adaptation(base_model, scheduler, y, mask, J, lr, eta=1.0, param_scope="adaln_only", device="cuda"):
    netG = copy.deepcopy(base_model).to(device)
    netG.eval()
    
    start_time = time.time()
    x_current = adjoint_operator(y).real
    x_current, _ = scheduler.add_noise(x_current, stage_idx=scheduler.num_stages - 1)

    x_input_tta = None
    t_final = torch.full((1,), scheduler.stages[1], dtype=torch.long, device=device)
    
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
            param.requires_grad = True if ("adaLN" in name or "ln" in name or "norm" in name or "pos_embed" in name) else False
    else:
        for param in netG.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, netG.parameters()), lr=lr, betas=(0.5, 0.9), weight_decay=1e-4)
    
    best_loss = float('inf')
    best_x_0_pred = None

    for step in range(1, J + 1):
        optimizer.zero_grad()
        x_recon = netG(x_input_tta, t_final)
        kspace_recon = torch.fft.fftshift(torch.fft.fft2(x_recon.to(torch.complex64), dim=(-2, -1)), dim=(-2, -1))
        
        masked_recon = kspace_recon * mask
        masked_target = y * mask
        loss = torch.mean(torch.abs(masked_recon.real - masked_target.real)) + torch.mean(torch.abs(masked_recon.imag - masked_target.imag))
               
        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_0_pred = x_recon.detach().clone()

    netG.eval()
    with torch.no_grad():
        if best_x_0_pred is None: best_x_0_pred = netG(x_input_tta, t_final)
        final_recon = apply_data_consistency(best_x_0_pred, y, mask, eta=eta)
        
    execution_time = time.time() - start_time
    del netG, optimizer
    torch.cuda.empty_cache()
    
    return torch.clamp(final_recon, -1.0, 1.0), execution_time

# =============================================================================
# 3. HELPER & PREPROCESSING UTILITIES
# =============================================================================
def get_spatial_magnitude(t_img):
    arr = torch.clamp((t_img.squeeze() + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()
    if arr.ndim == 3:
        arr = np.sqrt(np.sum(arr**2, axis=0))
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def ensure_centered_image(img):
    H, W = img.shape
    center_energy = np.mean(img[H//4:3*H//4, W//4:3*W//4])
    corner_energy = (np.mean(img[:H//4, :W//4]) + np.mean(img[:H//4, 3*W//4:]) + 
                     np.mean(img[3*H//4:, :W//4]) + np.mean(img[3*H//4:, 3*W//4:])) / 4.0
    if corner_energy > center_energy:
        return np.fft.fftshift(img)
    return img

@torch.inference_mode()
def evaluate_slice(x_recon, x_true, lpips_model, device):
    """
    CLINICAL GOLD STANDARD: Collapses multi-coil / multi-channel outputs via 
    Root-Sum-of-Squares (RSS) BEFORE calculating PSNR, SSIM, and LPIPS.
    """
    # 1. Collapse multi-channel arrays to a single 2D spatial magnitude map via RSS
    if x_recon.shape[1] > 1:
        rec_mag = torch.sqrt(torch.sum(x_recon**2, dim=1, keepdim=True))
        true_mag = torch.sqrt(torch.sum(x_true**2, dim=1, keepdim=True))
    else:
        rec_mag = torch.abs(x_recon)
        true_mag = torch.abs(x_true)

    # 2. Normalize intensities strictly to [0, 1] range for accurate metric evaluation
    rec_01 = (rec_mag - rec_mag.min()) / (rec_mag.max() - rec_mag.min() + 1e-8)
    true_01 = (true_mag - true_mag.min()) / (true_mag.max() - true_mag.min() + 1e-8)
    
    rec_np = rec_01.squeeze().cpu().numpy()
    true_np = true_01.squeeze().cpu().numpy()
    
    # 3. Compute quantitative metrics on the unified, anatomical 2D image
    psnr = float(skimage.metrics.peak_signal_noise_ratio(true_np, rec_np, data_range=1.0))
    ssim = float(skimage.metrics.structural_similarity(true_np, rec_np, data_range=1.0))
    
    # 4. Format for LPIPS (requires 3-channel RGB [-1, 1] tensors)
    tensor_rec = (rec_01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
    tensor_true = (true_01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
    lpips_score = float(lpips_model(tensor_rec.to(device), tensor_true.to(device)).item())
    
    return psnr, ssim, lpips_score

def load_and_preprocess_volume(file_path, target_channels=10, target_size=256):
    """
    Loads IXI files and adapts spatial dimensions/channels to match the fastMRI model requirements.
    Uses CHANNEL REPETITION to fill all expected model channels with anatomical data.
    """
    if file_path.endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as hf:
            possible_keys = ['reconstruction_rss', 'reconstruction_esc', 'image', 'data', 'kspace', 'target']
            data_key = next((k for k in possible_keys if k in hf.keys()), list(hf.keys())[0])
            matrix = hf[data_key][()]
    elif file_path.endswith('.npy'):
        matrix = np.load(file_path)
    elif file_path.endswith('.npz'):
        npz = np.load(file_path); matrix = npz[npz.files[0]]
    elif file_path.endswith('.pt'):
        matrix = torch.load(file_path, map_location="cpu").numpy()
        
    if np.iscomplexobj(matrix):
        matrix = np.real(matrix)
        
    tensor_data = torch.tensor(matrix, dtype=torch.float32)
    
    if tensor_data.ndim == 2:
        tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
    elif tensor_data.ndim == 3:
        if tensor_data.shape[-1] < tensor_data.shape[0] and tensor_data.shape[-1] <= 32:
            tensor_data = tensor_data.permute(2, 0, 1)
        tensor_data = tensor_data.unsqueeze(1)
    elif tensor_data.ndim == 4:
        if tensor_data.shape[-1] < tensor_data.shape[1] and tensor_data.shape[-1] <= 32:
            tensor_data = tensor_data.permute(0, 3, 1, 2)
            
    # Channel Adaptation: Duplicate IXI's single channel across target_channels (e.g., 1 -> 10)
    B, C, H, W = tensor_data.shape
    if C > target_channels:
        tensor_data = tensor_data[:, :target_channels, :, :]
    elif C < target_channels:
        tensor_data = tensor_data.repeat(1, target_channels // C, 1, 1)
        
    # Spatial Adaptation: Center crop or Pad IXI slices to match target_size
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
            pad_top = pad_h // 2; pad_bot = pad_h - pad_top
            pad_left = pad_w // 2; pad_right = pad_w - pad_left
            cropped = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=0)
        tensor_data = cropped
        
    min_val, max_val = tensor_data.min(), tensor_data.max()
    tensor_data = ((tensor_data - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0
    return tensor_data

# =============================================================================
# 4. MAIN CROSS-DOMAIN EVALUATION SCRIPT
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running CROSS-DOMAIN V3 (RSS COLLAPSED + REPEATED CHANNELS) on: {str(device).upper()}")
    
    # 1. Load fastMRI winning hyperparameters
    with open("best_hyperparameters_fastmri.json", "r") as f:
        best_hyperparams = json.load(f)
        
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
    
    # 2. Load pre-trained fastMRI model checkpoint
    CHECKPOINT_FILE = "/ugproj/sipl-prj10848/Idan/fastmri_checkpoints_sinusoidal/adadiff_fastmri_sinusoidal_latest.pt"
    base_model = load_trained_generator(checkpoint_path=CHECKPOINT_FILE, device=device)
    scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)

    # 3. Index IXI test dataset
    TEST_BASE_DIR = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/test"
    contrasts = ["PD", "T2", "T1"]
    
    test_set = {c: [] for c in contrasts}
    if os.path.exists(TEST_BASE_DIR):
        for root, dirs, files in os.walk(TEST_BASE_DIR):
            for f in sorted(files):
                if f.endswith(('.h5', '.hdf5', '.npy', '.npz', '.pt')):
                    file_path = os.path.join(root, f)
                    search_string = (f + "_" + root).lower()
                    for c in contrasts:
                        if c.lower() in search_string:
                            test_set[c].append(file_path)
                            break

    MAX_VOLUMES_PER_CONTRAST = 5
    for c in test_set:
        test_set[c] = test_set[c][:MAX_VOLUMES_PER_CONTRAST]

    print("\n" + "="*50)
    print("📊 CROSS-DOMAIN V3 TEST SUBSET INVENTORY:")
    for c, samples in test_set.items():
        print(f"   -> IXI Modality [{c}]: {len(samples)} patient volumes selected")
    print("="*50)

    acceleration_rates = [4, 8]
    final_results = {}

    for R in acceleration_rates:
        config = best_hyperparams[f"R_{R}"]
        print(f"\n=========================================================")
        print(f" EVALUATING CROSS-DOMAIN (fastMRI -> IXI) AT R = {R}x")
        print(f" Config: lr={config['lr']}, J={config['J']}, eta={config['eta']}, scope={config['scope']}")
        print(f"=========================================================")
        
        final_results[f"R_{R}"] = {}
        
        for contrast, file_paths in test_set.items():
            if not file_paths: continue
                
            c_psnrs, c_ssims, c_lpips, c_times = [], [], [], []
            best_contrast_score = -1.0
            best_visual_data = None
            
            for vol_idx, fpath in enumerate(file_paths):
                volume_data = load_and_preprocess_volume(
                    fpath, 
                    target_channels=base_model.in_channels, 
                    target_size=base_model.input_size
                )
                
                for slice_idx in range(volume_data.shape[0]):
                    x_true = volume_data[slice_idx:slice_idx+1].to(device)
                    y, mask = get_imaging_operator(x_true, R=R)
                    
                    recon, exec_time = run_prior_adaptation(
                        base_model, scheduler, y, mask, 
                        J=config["J"], lr=config["lr"], eta=config["eta"], 
                        param_scope=config["scope"], device=device
                    )
                    
                    psnr, ssim, lpips_val = evaluate_slice(recon, x_true, loss_fn_alex, device)
                    
                    c_psnrs.append(psnr); c_ssims.append(ssim); c_lpips.append(lpips_val); c_times.append(exec_time)
                    
                    slice_energy = torch.mean(torch.abs(x_true)).item()
                    if slice_energy > 0.05:
                        sample_score = (psnr / 40.0) + ssim  
                        if sample_score > best_contrast_score:
                            best_contrast_score = sample_score
                            best_visual_data = {
                                "true": get_spatial_magnitude(x_true),
                                "aliased": get_spatial_magnitude(adjoint_operator(y).real.float()),
                                "recon": get_spatial_magnitude(recon),
                                "psnr": psnr, "ssim": ssim
                            }

                    del x_true, y, mask, recon
                    torch.cuda.empty_cache()
                    gc.collect()

            mean_psnr, std_psnr = np.mean(c_psnrs), np.std(c_psnrs)
            mean_ssim, std_ssim = np.mean(c_ssims) * 100, np.std(c_ssims) * 100
            mean_lpips, std_lpips = np.mean(c_lpips), np.std(c_lpips)
            mean_time, std_time = np.mean(c_times), np.std(c_times)

            final_results[f"R_{R}"][contrast] = {
                "PSNR_mean": float(mean_psnr), "PSNR_std": float(std_psnr),
                "SSIM_mean": float(mean_ssim), "SSIM_std": float(std_ssim),
                "LPIPS_mean": float(mean_lpips), "LPIPS_std": float(std_lpips),
                "Time_mean": float(mean_time), "Time_std": float(std_time)
            }

            print(f"  [{contrast}] PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB | SSIM: {mean_ssim:.2f} ± {std_ssim:.2f}% | LPIPS: {mean_lpips:.4f} ± {std_lpips:.4f} | Time: {mean_time:.2f}s")

            # Save distinct images with '_rss_' prefix so no prior plots are overwritten
            if best_visual_data is not None:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
                fig.suptitle(
                    f"Cross-Domain V3 (RSS Collapsed) Best Recon: {contrast} (R={R}x)\n"
                    f"PSNR: {best_visual_data['psnr']:.2f} dB | SSIM: {best_visual_data['ssim']*100:.2f}%", 
                    fontsize=13, fontweight='bold', y=1.02
                )
                
                t_img = ensure_centered_image(best_visual_data["true"])
                aliased_img = ensure_centered_image(best_visual_data["aliased"])
                r_img = ensure_centered_image(best_visual_data["recon"])
                e_img = np.abs(t_img - r_img)

                titles = ["Ground Truth", "Zero-Filled Input", "DiT Reconstructed", "Absolute Error Map"]
                images = [t_img, aliased_img, r_img, e_img]
                cmaps = ["gray", "gray", "gray", "hot"]

                for i, ax in enumerate(axes):
                    im = ax.imshow(images[i], cmap=cmaps[i], vmin=0.0, vmax=1.0 if i < 3 else 0.15, origin="upper")
                    ax.set_title(titles[i], fontsize=11, fontweight="bold", pad=8)
                    ax.set_xlabel("Spatial Position X (pixels)", fontsize=9)
                    if i == 0:
                        ax.set_ylabel("Spatial Position Y (pixels)", fontsize=9)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    
                    if i == 3:
                        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("|Error|", fontsize=9)
                
                plt.tight_layout()
                plt.savefig(f"cross_domain_rss_golden_sample_{contrast}_R{R}.png", dpi=300, bbox_inches='tight')
                plt.close()

    # Save brand new JSON file so prior experimental logs remain untouched
    with open("fastmri_to_ixi_cross_domain_results_rss.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "="*80)
    print("🏆 FINAL CROSS-DOMAIN V3 (RSS COLLAPSED) RESULTS SUMMARY")
    print("="*80)
    
    print("\n--- LATEX FORMAT ---")
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{cccccc}")
    print(r"\hline")
    print(r"\textbf{Accel} & \textbf{Contrast} & \textbf{PSNR (dB)} & \textbf{SSIM (\%)} & \textbf{LPIPS} & \textbf{Time (s)} \\")
    print(r"\hline")
    for r_key, r_data in final_results.items():
        r_num = r_key.replace("R_", "")
        for c_key, m in r_data.items():
            print(f"{r_num}x & {c_key} & {m['PSNR_mean']:.2f} $\\pm$ {m['PSNR_std']:.2f} & {m['SSIM_mean']:.2f} $\\pm$ {m['SSIM_std']:.2f} & {m['LPIPS_mean']:.4f} $\\pm$ {m['LPIPS_std']:.4f} & {m['Time_mean']:.2f} $\\pm$ {m['Time_std']:.2f} \\\\")
        print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Cross-domain V3 (Root-Sum-of-Squares collapsed) reconstruction metrics.}")
    print(r"\label{tab:cross_domain_v3_results}")
    print(r"\end{table}")
    print("="*80)
    print("\n✅ Execution complete! Results saved to 'fastmri_to_ixi_cross_domain_results_rss.json' and 'cross_domain_rss_golden_sample_*.png'.")