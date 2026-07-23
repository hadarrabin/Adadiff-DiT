import os
import time
import json
import math
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# =============================================================================
# 1. SINUSOIDAL TIMESTEP EMBEDDING & SCHEDULER
# =============================================================================
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
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

# =============================================================================
# 2. DIFFUSION TRANSFORMER ARCHITECTURE
# =============================================================================
class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.qkv_project = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_project = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim * 4, hidden_dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mod = self.adaLN_modulation(c).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(mod, 6, dim=-1)

        norm_x1 = self.ln1(x) * (1.0 + gamma1) + beta1
        qkv = self.qkv_project(norm_x1).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_out = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + alpha1 * self.out_project(attn_out)

        norm_x2 = self.ln2(x) * (1.0 + gamma2) + beta2
        x = x + alpha2 * self.mlp(norm_x2)
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, in_channels: int = 10, input_size: int = 256, patch_size: int = 4, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patchify = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([AdaLNZeroBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, (patch_size ** 2) * in_channels)

    def forward(self, x: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_tokens = self.patchify(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp(get_timestep_embedding(t_tensor, embedding_dim=self.hidden_dim))
        for block in self.blocks: x_tokens = block(x_tokens, c)
        decoded = self.final_linear(self.final_ln(x_tokens))
        p = self.patch_size
        return decoded.view(B, H // p, W // p, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

# =============================================================================
# 3. FOURIER OPERATORS & DUAL-DOMAIN PROJECTIONS
# =============================================================================
def get_cartesian_mask(cols: int, R: int = 4, center_fraction: float = 0.08, device: str = "cuda") -> torch.Tensor:
    mask = torch.zeros((1, 1, 1, cols), device=device)
    num_low_freqs = int(cols * center_fraction)
    pad = (cols - num_low_freqs) // 2
    mask[:, :, :, pad:pad + num_low_freqs] = 1.0
    for i in range(0, cols, R):
        mask[:, :, :, i] = 1.0
    return mask

def tensor_to_complex(x_tensor: torch.Tensor) -> torch.Tensor:
    num_coils = x_tensor.shape[1] // 2
    return torch.complex(x_tensor[:, :num_coils], x_tensor[:, num_coils:])

def complex_to_tensor(x_complex: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_complex.real, x_complex.imag], dim=1)

def rss_combine(x_complex: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(x_complex) ** 2, dim=1, keepdim=True))

def apply_spatial_data_consistency(x_pred_spatial: torch.Tensor, y_measured_complex: torch.Tensor, mask: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    x_pred_complex = tensor_to_complex(x_pred_spatial)
    kspace_pred = torch.fft.fftshift(torch.fft.fft2(x_pred_complex, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    kspace_dc = kspace_pred * (1.0 - mask) + (kspace_pred - eta * (kspace_pred - y_measured_complex)) * mask
    x_dc_complex = torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    return complex_to_tensor(x_dc_complex)

def get_noised_measured_kspace(y_clean_kspace: torch.Tensor, stage_idx: int, scheduler: AdaDiffScheduler) -> torch.Tensor:
    if stage_idx == 0:
        return y_clean_kspace
    spatial_clean = torch.fft.ifft2(torch.fft.ifftshift(y_clean_kspace, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    spatial_tensor = complex_to_tensor(spatial_clean)
    spatial_noisy, _ = scheduler.add_noise(spatial_tensor, stage_idx=stage_idx)
    spatial_noisy_complex = tensor_to_complex(spatial_noisy)
    return torch.fft.fftshift(torch.fft.fft2(spatial_noisy_complex, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

# =============================================================================
# 4. FULL ADADIFF 2-PHASE INFERENCE ENGINE (BEST-LOSS TRACKING & FP32 FIX)
# =============================================================================
def run_adadiff_two_phase(base_model: nn.Module, scheduler: AdaDiffScheduler, y_undersampled_kspace: torch.Tensor, mask: torch.Tensor, J: int, lr: float, eta: float, param_scope: str = "adaln_only", device: str = "cuda", record_step_intervals: list = None):
    netG = copy.deepcopy(base_model).to(device)
    netG.eval()
    
    start_time = time.time()
    step_reconstructions = {}
    
    zf_spatial_complex = torch.fft.ifft2(torch.fft.ifftshift(y_undersampled_kspace, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    x_current = complex_to_tensor(zf_spatial_complex).float()

    x_current, _ = scheduler.add_noise(x_current, stage_idx=scheduler.num_stages - 1)

    x_input_tta = None
    t_final = torch.full((1,), scheduler.stages[1], dtype=torch.long, device=device) # t=125
    
    # PHASE 1: Generative Reverse Diffusion (Zero-Shot)
    with torch.no_grad():
        for stage_idx in reversed(range(scheduler.num_stages - 1)):
            if stage_idx == 0:
                # Intercept precisely before the final step so Phase 2 gets x_{125}
                x_input_tta = x_current.detach().clone() 

            t_parent = scheduler.stages[stage_idx + 1]
            t_tensor = torch.full((1,), t_parent, dtype=torch.long, device=device)
            
            x_pred = netG(x_current, t_tensor)
            y_noised_kspace = get_noised_measured_kspace(y_undersampled_kspace, stage_idx, scheduler)
            x_current = apply_spatial_data_consistency(x_pred, y_noised_kspace, mask, eta=eta)

    if record_step_intervals and 0 in record_step_intervals:
        with torch.no_grad():
            step_reconstructions[0] = (rss_combine(tensor_to_complex(x_current)), time.time() - start_time)

    # PHASE 2: Test-Time Adaptation
    netG.train()
    if param_scope == "adaln_only":
        for name, param in netG.named_parameters():
            param.requires_grad = True if ("adaLN" in name or "ln" in name or "norm" in name or "pos_embed" in name) else False
    else:
        for param in netG.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, netG.parameters()), lr=lr, betas=(0.5, 0.9), weight_decay=1e-4)
    loss_history = []
    
    # BEST-STATE TRACKER: Prevents PSNR collapse
    best_loss = float('inf')
    best_x_0_pred = None

    for step in range(1, J + 1):
        optimizer.zero_grad()
        
        # PURE FP32 FOR GRADIENT STABILITY IN K-SPACE
        x_recon_spatial = netG(x_input_tta, t_final)
        x_recon_complex = tensor_to_complex(x_recon_spatial)
        kspace_recon = torch.fft.fftshift(torch.fft.fft2(x_recon_complex, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        
        # EXACT MATCH TO TRAINING LOSS: L1 on Real and Imag separately on valid mask
        masked_recon = kspace_recon * mask
        masked_target = y_undersampled_kspace * mask
        loss = torch.mean(torch.abs(masked_recon.real - masked_target.real)) + \
               torch.mean(torch.abs(masked_recon.imag - masked_target.imag))
               
        loss.backward()
        
        # GRADIENT CLIPPING: Prevents massive K-space Center DC jumps
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

        # Save optimal mapping!
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_0_pred = x_recon_spatial.detach().clone()

        if record_step_intervals and step in record_step_intervals:
            with torch.no_grad():
                netG.eval()
                # Use current best to track stable trajectory
                temp_spatial = apply_spatial_data_consistency(best_x_0_pred, y_undersampled_kspace, mask, eta=eta)
                step_reconstructions[step] = (rss_combine(tensor_to_complex(temp_spatial)), time.time() - start_time)
                netG.train()

    netG.eval()
    with torch.no_grad():
        # Fallback in case J=0 (shouldn't happen, but safe)
        if best_x_0_pred is None:
            best_x_0_pred = netG(x_input_tta, t_final)
            
        final_spatial_tensor = apply_spatial_data_consistency(best_x_0_pred, y_undersampled_kspace, mask, eta=eta)
        final_spatial_rss = rss_combine(tensor_to_complex(final_spatial_tensor))

    del netG, optimizer
    torch.cuda.empty_cache()
    return final_spatial_rss, loss_history, step_reconstructions

@torch.inference_mode()
def evaluate_metrics(x_recon_rss: torch.Tensor, x_true_rss: torch.Tensor, lpips_model: nn.Module, device: str):
    rec_np = x_recon_rss.squeeze().cpu().numpy()
    true_np = x_true_rss.squeeze().cpu().numpy()
    psnr = float(skimage.metrics.peak_signal_noise_ratio(true_np, rec_np, data_range=1.0))
    ssim = float(skimage.metrics.structural_similarity(true_np, rec_np, data_range=1.0))
    tensor_rec = (x_recon_rss.repeat(1, 3, 1, 1) * 2.0) - 1.0
    tensor_true = (x_true_rss.repeat(1, 3, 1, 1) * 2.0) - 1.0
    lpips_val = float(lpips_model(tensor_rec.to(device), tensor_true.to(device)).item())
    return psnr, ssim, lpips_val

# =============================================================================
# 5. MAIN EVALUATION WITH FULL 48-SWEEP CONFIGURATION
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Robust Phase-2 Adaptive Grid Search on: {str(device).upper()}")
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
    VIRTUAL_COILS = 5
    INPUT_SIZE = 256
    
    CHECKPOINT_FILE = "/ugproj/sipl-prj10848/Idan/fastmri_checkpoints_sinusoidal/adadiff_fastmri_sinusoidal_epoch_50.pt"
    base_model = DiffusionTransformer(in_channels=2*VIRTUAL_COILS, input_size=INPUT_SIZE).to(device)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    base_model.load_state_dict(checkpoint['generator_state_dict'] if 'generator_state_dict' in checkpoint else checkpoint)
    scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)

    val_dir = "/ugproj/sipl-prj10848/Idan/fastMRI_adadiff/val/val"
    val_set = {"AXT1": [], "AXT1PRE": [], "AXT1POST": [], "AXT2": [], "AXFLAIR": []}
    max_samples = 10

    for f in sorted(os.listdir(val_dir)):
        if not (f.endswith('.h5') or f.endswith('.npy')): continue
        parts = f.split('_')
        if len(parts) <= 2: continue
        contrast = parts[2]
        if contrast not in val_set or len(val_set[contrast]) >= max_samples: continue

        filepath = os.path.join(val_dir, f)
        try:
            if f.endswith('.h5'):
                with h5py.File(filepath, 'r') as hf:
                    kspace = np.array(hf['kspace'], dtype=np.complex64)
                slice_data = torch.from_numpy(kspace[kspace.shape[0] // 2])
            else:
                slice_data = torch.from_numpy(np.load(filepath))

            num_coils, h, w = slice_data.shape
            flattened = slice_data.view(num_coils, -1)
            U, _, _ = torch.linalg.svd(flattened, full_matrices=False)
            compressed = torch.matmul(U[:, :VIRTUAL_COILS].H, flattened).view(VIRTUAL_COILS, h, w)

            real_res = F.interpolate(compressed.real.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
            imag_res = F.interpolate(compressed.imag.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
            compressed_res = torch.complex(real_res.squeeze(0), imag_res.squeeze(0))

            spatial_coils = torch.fft.ifft2(torch.fft.ifftshift(compressed_res.unsqueeze(0), dim=(-2, -1)), dim=(-2, -1), norm="ortho").squeeze(0)
            max_val = torch.max(torch.abs(spatial_coils))
            spatial_coils_norm = spatial_coils / (max_val + 1e-8)
            
            y_full_complex = torch.fft.fftshift(torch.fft.fft2(spatial_coils_norm.unsqueeze(0), dim=(-2, -1), norm="ortho"), dim=(-2, -1)).squeeze(0)
            
            val_set[contrast].append((y_full_complex.unsqueeze(0), spatial_coils_norm.unsqueeze(0)))
        except Exception as e:
            pass

    # =========================================================================
    # FULL 48-SWEEP CONFIGURATION
    # =========================================================================
    accel_rates = [4, 8]
    learning_rates = [1e-4, 5e-4]
    adapt_steps = [25, 50, 100]  
    eta_vals = [0.8, 1.0]              
    scopes = ["adaln_only", "all"]  
    # =========================================================================
    
    step_checkpoints = [0, 10, 25, 50, 100]
    grid_summary = {}
    detailed_export = {}
    total_runs = len(accel_rates) * len(learning_rates) * len(adapt_steps) * len(eta_vals) * len(scopes)
    run_idx = 0

    for R in accel_rates:
        for lr in learning_rates:
            for J in adapt_steps:
                for eta in eta_vals:
                    for scope in scopes:
                        run_idx += 1
                        key = f"R{R}_LR{lr}_J{J}_ETA{eta}_SCOPE-{scope}"
                        print(f"\n====================================================================")
                        print(f"📊 [{run_idx}/{total_runs}] Sweep: R={R}x | LR={lr} | J={J} | ETA={eta} | Scope={scope}")
                        print(f"====================================================================")
                        
                        run_psnr, run_ssim, run_lpips, run_time = [], [], [], []
                        loss_curves = []
                        trajectory_data = {step: {"psnr": [], "ssim": [], "lpips": [], "time": []} for step in step_checkpoints if step <= J}
                        
                        for contrast, samples in val_set.items():
                            if not samples: continue
                            print(f"  🧠 Modality -> {contrast}")
                            
                            for idx, (y_full, spatial_target) in enumerate(samples):
                                y_full = y_full.to(device)
                                mask = get_cartesian_mask(INPUT_SIZE, R=R, device=device)
                                y_under = y_full * mask
                                
                                true_rss = rss_combine(spatial_target.to(device))
                                rss_max = torch.max(true_rss)
                                true_rss_norm = torch.clamp(true_rss / (rss_max + 1e-8), 0.0, 1.0)
                                
                                t0 = time.time()
                                final_rss, losses, step_recons = run_adadiff_two_phase(
                                    base_model, scheduler, y_under, mask, J=J, lr=lr, eta=eta, param_scope=scope, device=device, record_step_intervals=step_checkpoints
                                )
                                elapsed = time.time() - t0
                                
                                final_rss_norm = torch.clamp(final_rss / (rss_max + 1e-8), 0.0, 1.0)
                                
                                psnr, ssim, lpips_val = evaluate_metrics(final_rss_norm, true_rss_norm, loss_fn_alex, device)
                                print(f"      [Slice {idx+1}/{len(samples)}] PSNR: {psnr:.2f} dB | SSIM: {ssim*100:.2f}% | LPIPS: {lpips_val:.4f} | Time: {elapsed:.2f}s")
                                
                                run_psnr.append(psnr); run_ssim.append(ssim); run_lpips.append(lpips_val); run_time.append(elapsed)
                                loss_curves.append(losses)

                                for stp, (rec_rss, stp_time) in step_recons.items():
                                    rec_rss_norm = torch.clamp(rec_rss / (rss_max + 1e-8), 0.0, 1.0)
                                    sp_psnr, sp_ssim, sp_lpips = evaluate_metrics(rec_rss_norm, true_rss_norm, loss_fn_alex, device)
                                    trajectory_data[stp]["psnr"].append(sp_psnr)
                                    trajectory_data[stp]["ssim"].append(sp_ssim)
                                    trajectory_data[stp]["lpips"].append(sp_lpips)
                                    trajectory_data[stp]["time"].append(stp_time)

                        mean_psnr = float(np.mean(run_psnr))
                        mean_ssim = float(np.mean(run_ssim))
                        mean_lpips = float(np.mean(run_lpips))
                        mean_time = float(np.mean(run_time))
                        mean_loss_curve = np.mean(np.array(loss_curves), axis=0).tolist() if loss_curves else []
                        
                        mean_trajectory = {
                            str(stp): {
                                "psnr": float(np.mean(vals["psnr"])),
                                "ssim": float(np.mean(vals["ssim"])),
                                "lpips": float(np.mean(vals["lpips"])),
                                "time": float(np.mean(vals["time"]))
                            } for stp, vals in trajectory_data.items()
                        }

                        print(f"  -> SUMMARY: PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim*100:.2f}% | LPIPS: {mean_lpips:.4f} | Avg Time: {mean_time:.2f}s")

                        grid_summary[key] = {
                            "R": R, "LR": lr, "J": J, "ETA": eta, "Scope": scope,
                            "PSNR": mean_psnr, "SSIM": mean_ssim, "LPIPS": mean_lpips, "Time": mean_time
                        }
                        
                        detailed_export[key] = {
                            "config": grid_summary[key],
                            "mean_loss_trajectory": mean_loss_curve,
                            "step_convergence": mean_trajectory
                        }

                        with open("dit_adadiff_grid_summary.json", "w") as f:
                            json.dump(grid_summary, f, indent=4)
                        with open("dit_adadiff_plotting_data.json", "w") as f:
                            json.dump(detailed_export, f, indent=4)

    print("\n🏆 SWEEP COMPLETE! Generating benchmark comparison figures...")
    
    # Plotting Automation
    configs = list(grid_summary.keys())
    psnrs = [grid_summary[c]["PSNR"] for c in configs]
    ssims = [grid_summary[c]["SSIM"] * 100 for c in configs]
    times = [grid_summary[c]["Time"] for c in configs]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("fastMRI DiT-GAN vs. AdaDiff 2-Phase Optimization Benchmark", fontsize=15, fontweight="bold")
    
    axs[0].barh(configs[:10], psnrs[:10], color="cornflowerblue", edgecolor="black")
    axs[0].set_title("Top 10 Average PSNR (dB)"); axs[0].set_xlabel("dB")
    
    axs[1].barh(configs[:10], ssims[:10], color="mediumseagreen", edgecolor="black")
    axs[1].set_title("Top 10 Average SSIM (%)"); axs[1].set_xlabel("%")
    
    for c in configs[:3]:  
        traj = detailed_export[c]["step_convergence"]
        stps = sorted([int(k) for k in traj.keys()])
        t_vals = [traj[str(s)]["time"] for s in stps]
        p_vals = [traj[str(s)]["psnr"] for s in stps]
        axs[2].plot(t_vals, p_vals, marker="o", linewidth=2, label=f"J={grid_summary[c]['J']} | {grid_summary[c]['Scope']}")
        
    axs[2].set_title("PSNR vs. Latency (DiT Speed Advantage)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("PSNR (dB)")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("fastmri_dit_adadiff_benchmark_summary.png", dpi=300)
    plt.close()
    print("📈 Saved benchmark summary dashboard to: 'fastmri_dit_adadiff_benchmark_summary.png'.")