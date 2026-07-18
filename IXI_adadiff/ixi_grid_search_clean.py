import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# =========================================================================
# 1. NATIVE DIT ARCHITECTURE FOR IXI (STANDALONE ISOLATION)
# =========================================================================

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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_tokens = self.patchify(x).flatten(2).transpose(1, 2) 
        x_tokens = x_tokens + self.pos_embed
        c = self.time_mlp(t_emb)

        for block in self.blocks:
            x_tokens = block(x_tokens, c)

        x_tokens = self.final_ln(x_tokens)
        decoded = self.final_linear(x_tokens)

        p = self.patch_size
        h_patches, w_patches = H // p, W // p
        x_out = decoded.view(B, h_patches, w_patches, p, p, C)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x_out

# =========================================================================
# 2. MODEL WEIGHTS LOADER
# =========================================================================

def load_trained_generator(checkpoint_path, device="cuda"):
    print(f"🔄 Initializing DiffusionTransformer and loading weights from: {checkpoint_path}...")
    model = DiffusionTransformer(
        in_channels=1,
        input_size=200,
        patch_size=2,
        hidden_dim=384,
        depth=12,
        num_heads=6
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print("✅ Real IXI generator weights restored successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Raw state dict loaded successfully!")
    else:
        raise FileNotFoundError(f"❌ CRITICAL ERROR: Checkpoint file '{checkpoint_path}' not found!")
    return model.to(device)

# =========================================================================
# 3. SINGLE-CHANNEL IMAGING OPERATORS
# =========================================================================

def get_ixi_imaging_operator(x_true, R=4):
    B, C, H, W = x_true.shape
    kspace = torch.fft.fftshift(torch.fft.fft2(x_true, dim=(-2, -1)), dim=(-2, -1))
    
    mask = torch.zeros(W, device=x_true.device)
    center_fraction = 0.08
    num_low_freq = int(W * center_fraction)
    mask[W//2 - num_low_freq//2 : W//2 + num_low_freq//2] = 1.0
    
    for i in range(0, W, R):
        mask[i] = 1.0
        
    mask_2d = mask.unsqueeze(0).repeat(H, 1).unsqueeze(0).unsqueeze(0)
    y = kspace * mask_2d
    return y, mask_2d

def adjoint_operator(y):
    shifted_y = torch.fft.ifftshift(y, dim=(-2, -1))
    return torch.fft.ifft2(shifted_y, dim=(-2, -1))

# =========================================================================
# 4. PRIOR ADAPTATION ENGINE
# =========================================================================

def run_ixi_prior_adaptation(base_model, y, mask, J, lr, device="cuda"):
    import copy
    netG = copy.deepcopy(base_model).to(device)
    netG.train()
    
    optimizer = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))
    
    x_init_complex = adjoint_operator(y)
    x_init_input = torch.abs(x_init_complex).float()
    t_emb = torch.zeros(1, netG.hidden_dim, device=device)
    
    for step in range(J):
        optimizer.zero_grad()
        x_recon = netG(x_init_input, t_emb)
        
        kspace_recon = torch.fft.fftshift(torch.fft.fft2(x_recon, dim=(-2, -1)), dim=(-2, -1))
        y_recon = kspace_recon * mask
        
        loss = torch.mean(torch.abs(y_recon - y))
        loss.backward()
        optimizer.step()
        
    netG.eval()
    with torch.no_grad():
        final_recon = netG(x_init_input, t_emb).detach()
        
    del netG, optimizer
    torch.cuda.empty_cache()
    return torch.clamp(final_recon, 0.0, 1.0)

def evaluate_slice(x_recon, x_true, lpips_model, device):
    rec_np = x_recon.squeeze().cpu().numpy()
    true_np = x_true.squeeze().cpu().numpy()
    
    psnr = skimage.metrics.peak_signal_noise_ratio(true_np, rec_np, data_range=1.0)
    ssim = skimage.metrics.structural_similarity(true_np, rec_np, data_range=1.0)
    
    tensor_rec = (x_recon.repeat(1, 3, 1, 1) * 2.0) - 1.0
    tensor_true = (x_true.repeat(1, 3, 1, 1) * 2.0) - 1.0
    
    with torch.no_grad():
        lpips_score = lpips_model(tensor_rec.to(device), tensor_true.to(device)).item()
    return psnr, ssim, lpips_score

# =========================================================================
# 5. RUNNING THE SEARCH
# =========================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    ixi_weights = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/ixi_checkpoints/adadiff_ixi_epoch_20_final.pt"
    base_model = load_trained_generator(checkpoint_path=ixi_weights, device=device)
    
    print("\n📁 Scanning isolated IXI validation folder structural categories...")
    val_base_dir = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/val" 
    val_set = {"T1": [], "T2": [], "PD": []}
    max_samples_per_contrast = 10
    
    for contrast in val_set.keys():
        contrast_dir = os.path.join(val_base_dir, contrast)
        if os.path.exists(contrast_dir):
            all_files = sorted([f for f in os.listdir(contrast_dir) if f.endswith('.npy')])
            
            for f in all_files:
                if len(val_set[contrast]) >= max_samples_per_contrast:
                    break
                    
                file_path = os.path.join(contrast_dir, f)
                try:
                    matrix = np.load(file_path)
                    tensor_data = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    min_val, max_val = tensor_data.min(), tensor_data.max()
                    tensor_data = (tensor_data - min_val) / (max_val - min_val + 1e-8)
                    val_set[contrast].append(tensor_data)
                except Exception as e:
                    print(f"  ⚠️ Skipping file {f} inside {contrast}: {e}")
        else:
            print(f"  ⚠️ Warning: Contrast directory missing at: {contrast_dir}")

    for contrast, samples in val_set.items():
        print(f"  -> Successfully processed {len(samples)} physical .npy maps for structural category {contrast}")
        if len(samples) == 0:
            raise ValueError(f"❌ CRITICAL ERROR: Category folder '{contrast}' contains 0 valid scans!")

    learning_rates = [5e-4, 1e-4]
    adaptation_steps = [100, 250, 500]
    grid_results = {}
    
    for lr in learning_rates:
        for J in adaptation_steps:
            grid_key = f"LR_{lr}_J_{J}"
            print(f"\n==========================================")
            print(f"🚀 Evaluating Configuration: LR = {lr} | J = {J} Steps")
            print(f"==========================================")
            
            run_psnrs, run_ssims, run_lpips, run_times = [], [], [], []
            
            for contrast, samples in val_set.items():
                if len(samples) == 0:
                    continue
                print(f"\n🧠 Running contrast modality: {contrast}")
                for idx, x_true in enumerate(samples):
                    x_true = x_true.to(device)
                    y, mask = get_ixi_imaging_operator(x_true, R=4)
                    
                    t0 = time.time()
                    x_recon = run_ixi_prior_adaptation(base_model, y, mask, J=J, lr=lr, device=device)
                    elapsed = time.time() - t0
                    
                    psnr, ssim, lpips_score = evaluate_slice(x_recon, x_true, loss_fn_alex, device)
                    
                    run_psnrs.append(psnr)
                    run_ssims.append(ssim)
                    run_lpips.append(lpips_score)
                    run_times.append(elapsed)
                    
                    print(f"  [{contrast}] Slice {idx+1}/{len(samples)} -> PSNR: {psnr:.2f} dB | SSIM: {ssim*100:.2f}% | LPIPS: {lpips_score:.4f} | Time: {elapsed:.2f}s")
                    
            grid_results[grid_key] = {
                "lr": lr,
                "J": J,
                "PSNR": float(np.mean(run_psnrs)),
                "SSIM": float(np.mean(run_ssims)),
                "LPIPS": float(np.mean(run_lpips)),
                "Time": float(np.mean(run_times))
            }
            print(f"\n📊 CONFIG COMPLETE: {grid_key}")
            print(f"   -> Mean PSNR:  {grid_results[grid_key]['PSNR']:.2f} dB")
            print(f"   -> Mean SSIM:  {grid_results[grid_key]['SSIM']*100:.2f}%")
            print(f"   -> Mean LPIPS: {grid_results[grid_key]['LPIPS']:.4f}")
            print(f"   -> Mean Time:  {grid_results[grid_key]['Time']:.2f}s")

    # =========================================================================
    # 6. MULTI-CRITERIA UTILITY OPTIMIZATION (EXACT ALIGNMENT WITH fastMRI WEIGHTS)
    # =========================================================================
    w_psnr, w_ssim, w_lpips, w_time = 0.1, 0.3, 0.4, 0.2
    
    psnr_vals = [res["PSNR"] for res in grid_results.values()]
    ssim_vals = [res["SSIM"] for res in grid_results.values()]
    lpips_vals = [res["LPIPS"] for res in grid_results.values()]
    time_vals = [res["Time"] for res in grid_results.values()]
    
    eps = 1e-8
    best_score = -1
    best_config = None
    
    for key, res in grid_results.items():
        n_psnr = (res["PSNR"] - min(psnr_vals)) / (max(psnr_vals) - min(psnr_vals) + eps)
        n_ssim = (res["SSIM"] - min(ssim_vals)) / (max(ssim_vals) - min(ssim_vals) + eps)
        n_lpips = (max(lpips_vals) - res["LPIPS"]) / (max(lpips_vals) - min(lpips_vals) + eps)
        n_time = (max(time_vals) - res["Time"]) / (max(time_vals) - min(time_vals) + eps)
        
        utility_score = (w_psnr * n_psnr) + (w_ssim * n_ssim) + (w_lpips * n_lpips) + (w_time * n_time)
        res["Utility_Score"] = utility_score
        
        if utility_score > best_score:
            best_score = utility_score
            best_config = res

    best_params_path = "best_hyperparameters_ixi.json"
    with open(best_params_path, "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(f"\n🏆 WINNING CONFIGURATION FOR IXI: LR = {best_config['lr']} | J = {best_config['J']} Steps")

    # =========================================================================
    # 7. POST-RUN SELECTION GRAPH PLOTTING
    # =========================================================================
    print("\n📈 Generating dashboard metrics figure...")
    configs = list(grid_results.keys())
    psnrs = [grid_results[c]["PSNR"] for c in configs]
    ssims = [grid_results[c]["SSIM"] for c in configs]
    lpipss = [grid_results[c]["LPIPS"] for c in configs]
    times = [grid_results[c]["Time"] for c in configs]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DiT Hyperparameter Optimization Dashboard (IXI Validation)", fontsize=16, fontweight='bold')
    
    axs[0, 0].bar(configs, psnrs, color='skyblue', edgecolor='black')
    axs[0, 0].set_title("Average PSNR (Higher is Better)")
    axs[0, 0].set_ylabel("PSNR (dB)")
    axs[0, 0].tick_params(axis='x', rotation=15)
    
    axs[0, 1].bar(configs, ssims, color='lightgreen', edgecolor='black')
    axs[0, 1].set_title("Average SSIM (Higher is Better)")
    axs[0, 1].set_ylabel("SSIM Score")
    axs[0, 1].tick_params(axis='x', rotation=15)
    
    axs[1, 0].bar(configs, lpipss, color='salmon', edgecolor='black')
    axs[1, 0].set_title("Average LPIPS (Lower is Better)")
    axs[1, 0].set_ylabel("Perceptual Distance")
    axs[1, 0].tick_params(axis='x', rotation=15)
    
    axs[1, 1].bar(configs, times, color='plum', edgecolor='black')
    axs[1, 1].set_title("Inference Latency (Lower is Better)")
    axs[1, 1].set_ylabel("Inference Time (seconds/slice)")
    axs[1, 1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig("ixi_grid_search_metrics.png", dpi=300)
    print("💾 Dashboard metrics successfully plotted and saved to 'ixi_grid_search_metrics.png'!")