import os
import time
import json
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# =============================================================================
# 1. GLOBAL SINUSOIDAL EMBEDDING ENGINE
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

# =============================================================================
# 2. MODEL ARCHITECTURE
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

# =============================================================================
# 3. LOADER & IMAGING OPERATORS
# =============================================================================
def load_trained_generator(checkpoint_path, device="cuda"):
    model = DiffusionTransformer(in_channels=1, input_size=200, patch_size=2, hidden_dim=384, depth=12, num_heads=6)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"❌ Checkpoint missing at '{checkpoint_path}'!")
    return model.to(device)

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

def get_ixi_imaging_operator(x_true, R=4):
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

def get_noised_measured_kspace_ixi(y_clean, stage_idx, scheduler):
    if stage_idx == 0: return y_clean
    spatial_clean = adjoint_operator(y_clean).real
    spatial_noisy, _ = scheduler.add_noise(spatial_clean, stage_idx)
    return torch.fft.fftshift(torch.fft.fft2(spatial_noisy.to(torch.complex64), dim=(-2, -1)), dim=(-2, -1))

# =============================================================================
# 4. TTA CORE WITH TRAJECTORY SAVING & FP32 FIX
# =============================================================================
def run_ixi_prior_adaptation(base_model, scheduler, y, mask, J, lr, eta=1.0, param_scope="adaln_only", device="cuda", record_step_intervals=None):
    netG = copy.deepcopy(base_model).to(device)
    netG.eval()
    
    start_time = time.time()
    step_reconstructions = {}
    
    x_current = adjoint_operator(y).real
    x_current, _ = scheduler.add_noise(x_current, stage_idx=scheduler.num_stages - 1)

    x_input_tta = None
    t_final = torch.full((1,), scheduler.stages[1], dtype=torch.long, device=device) # t=125
    
    # PHASE 1: Generative Reverse Diffusion
    with torch.no_grad():
        for stage_idx in reversed(range(scheduler.num_stages - 1)):
            if stage_idx == 0:
                x_input_tta = x_current.detach().clone() 

            t_parent = scheduler.stages[stage_idx + 1]
            t_tensor = torch.full((1,), t_parent, dtype=torch.long, device=device)
            
            x_pred = netG(x_current, t_tensor)
            y_noised_kspace = get_noised_measured_kspace_ixi(y, stage_idx, scheduler)
            x_current = apply_data_consistency(x_pred, y_noised_kspace, mask, eta=eta)

    if record_step_intervals and 0 in record_step_intervals:
        step_reconstructions[0] = (torch.clamp(x_current, -1.0, 1.0).detach(), time.time() - start_time)

    # PHASE 2: Test-Time Adaptation
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
    loss_history = []

    for step in range(1, J + 1):
        optimizer.zero_grad()
        
        # FP32 K-Space Loss ensuring gradient stability
        x_recon = netG(x_input_tta, t_final)
        kspace_recon = torch.fft.fftshift(torch.fft.fft2(x_recon.to(torch.complex64), dim=(-2, -1)), dim=(-2, -1))
        
        masked_recon = kspace_recon * mask
        masked_target = y * mask
        
        # Real & Imaginary exact matched L1 error
        loss = torch.mean(torch.abs(masked_recon.real - masked_target.real)) + torch.mean(torch.abs(masked_recon.imag - masked_target.imag))
               
        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_history.append(loss.item())

        # BEST-STATE TRACKER
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_0_pred = x_recon.detach().clone()

        if record_step_intervals and step in record_step_intervals:
            with torch.no_grad():
                netG.eval()
                temp_dc = apply_data_consistency(best_x_0_pred, y, mask, eta=eta)
                step_reconstructions[step] = (torch.clamp(temp_dc, -1.0, 1.0).detach(), time.time() - start_time)
                netG.train()

    netG.eval()
    with torch.no_grad():
        if best_x_0_pred is None: best_x_0_pred = netG(x_input_tta, t_final)
        final_recon = apply_data_consistency(best_x_0_pred, y, mask, eta=eta)
        
    del netG, optimizer
    torch.cuda.empty_cache()
    
    return torch.clamp(final_recon, -1.0, 1.0), loss_history, step_reconstructions

@torch.inference_mode()
def evaluate_slice(x_recon, x_true, lpips_model, device):
    rec_01 = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
    true_01 = torch.clamp((x_true + 1.0) / 2.0, 0.0, 1.0)
    rec_np = rec_01.squeeze().cpu().numpy()
    true_np = true_01.squeeze().cpu().numpy()
    
    psnr = float(skimage.metrics.peak_signal_noise_ratio(true_np, rec_np, data_range=1.0))
    ssim = float(skimage.metrics.structural_similarity(true_np, rec_np, data_range=1.0))
    
    # LPIPS architecture strict [-1, 1] range expectation
    tensor_rec = x_recon.repeat(1, 3, 1, 1)
    tensor_true = x_true.repeat(1, 3, 1, 1)
    lpips_score = float(lpips_model(tensor_rec.to(device), tensor_true.to(device)).item())
    return psnr, ssim, lpips_score

# =============================================================================
# 5. MAIN SWEEP EVALUATION (LIVE SAVING ENABLED)
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Unified IXI Sweep on target device: {str(device).upper()}")
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
    
    CHECKPOINT_FILE = "/ugproj/sipl-prj10848/Idan/ixi_checkpoints_sinusoidal/adadiff_ixi_sinusoidal_latest.pt"
    base_model = load_trained_generator(checkpoint_path=CHECKPOINT_FILE, device=device)
    scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)
    
    print("\n📁 Scanning isolated IXI validation metrics folder categories...")
    val_base_dir = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/val" 
    val_set = {"T1": [], "T2": [], "PD": []}
    max_samples_per_contrast = 6
    
    for contrast in val_set.keys():
        contrast_dir = os.path.join(val_base_dir, contrast)
        if os.path.exists(contrast_dir):
            all_files = sorted([f for f in os.listdir(contrast_dir) if f.endswith('.npy')])
            for f in all_files:
                if len(val_set[contrast]) >= max_samples_per_contrast: break
                try:
                    matrix = np.load(os.path.join(contrast_dir, f))
                    tensor_data = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    min_val, max_val = tensor_data.min(), tensor_data.max()
                    tensor_data = ((tensor_data - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0
                    val_set[contrast].append(tensor_data)
                except Exception as e:
                    pass

    # EXACT 48-SWEEP MATCH TO FASTMRI
    acceleration_rates = [4, 8]
    learning_rates = [1e-4, 5e-4]
    adaptation_steps = [25, 50, 100]  # <-- ALIGNED EXACTLY TO YOUR FASTMRI J-ARRAY!
    step_lengths_eta = [0.8, 1.0]
    param_scopes = ["adaln_only", "all"]
    
    step_checkpoints = [0, 10, 25, 50, 100]
    grid_summary = {}
    detailed_export = {}
    best_visual_candidate = {"psnr": -1.0, "true": None, "aliased": None, "recon": None, "title": ""}
    
    total_runs = len(acceleration_rates) * len(learning_rates) * len(adaptation_steps) * len(step_lengths_eta) * len(param_scopes)
    run_idx = 0

    for R in acceleration_rates:
        for lr in learning_rates:
            for J in adaptation_steps:
                for eta in step_lengths_eta:
                    for scope in param_scopes:
                        run_idx += 1
                        grid_key = f"R{R}_LR{lr}_J{J}_ETA{eta}_SCOPE-{scope}"
                        print(f"\n====================================================================")
                        print(f"📊 [{run_idx}/{total_runs}] Sweep: R={R}x | LR={lr} | J={J} | ETA={eta} | Scope={scope}")
                        print(f"====================================================================")
                        
                        run_psnrs, run_ssims, run_lpips, run_times = [], [], [], []
                        loss_curves = []
                        trajectory_data = {step: {"psnr": [], "ssim": [], "lpips": [], "time": []} for step in step_checkpoints if step <= J}
                        
                        for contrast, samples in val_set.items():
                            if not samples: continue
                            print(f"  🧠 Modality -> {contrast}")
                            for idx, x_true in enumerate(samples):
                                x_true = x_true.to(device)
                                y, mask = get_ixi_imaging_operator(x_true, R=R)
                                
                                final_recon, losses, step_recons = run_ixi_prior_adaptation(
                                    base_model, scheduler, y, mask, J=J, lr=lr, eta=eta, 
                                    param_scope=scope, device=device, record_step_intervals=step_checkpoints
                                )
                                
                                # Evaluate Final
                                psnr, ssim, lpips_score = evaluate_slice(final_recon, x_true, loss_fn_alex, device)
                                print(f"      [Slice {idx+1}/{len(samples)}] PSNR: {psnr:.2f} dB | SSIM: {ssim*100:.2f}% | LPIPS: {lpips_score:.4f}")
                                
                                run_psnrs.append(psnr); run_ssims.append(ssim); run_lpips.append(lpips_score)
                                run_times.append(step_recons.get(J, (None, 0.0))[1] if J in step_recons else 0.0)
                                loss_curves.append(losses)

                                # Evaluate Step Intervals
                                for stp, (rec_img, stp_time) in step_recons.items():
                                    sp_psnr, sp_ssim, sp_lpips = evaluate_slice(rec_img, x_true, loss_fn_alex, device)
                                    trajectory_data[stp]["psnr"].append(sp_psnr)
                                    trajectory_data[stp]["ssim"].append(sp_ssim)
                                    trajectory_data[stp]["lpips"].append(sp_lpips)
                                    trajectory_data[stp]["time"].append(stp_time)
                                
                                if psnr > best_visual_candidate["psnr"] and R == 4:
                                    best_visual_candidate.update({
                                        "psnr": psnr, 
                                        "true": torch.clamp((x_true.squeeze() + 1.0)/2.0, 0, 1).cpu().numpy(),
                                        "aliased": torch.clamp((adjoint_operator(y).real.float().squeeze() + 1.0)/2.0, 0, 1).cpu().numpy(),
                                        "recon": torch.clamp((final_recon.squeeze() + 1.0)/2.0, 0, 1).cpu().numpy(),
                                        "title": f"Modality: {contrast} | R={R}x | PSNR: {psnr:.2f} dB"
                                    })

                        mean_psnr = float(np.mean(run_psnrs))
                        mean_ssim = float(np.mean(run_ssims))
                        mean_lpips = float(np.mean(run_lpips))
                        mean_time = float(np.mean(run_times))
                        mean_loss_curve = np.mean(np.array(loss_curves), axis=0).tolist() if loss_curves else []
                        
                        mean_trajectory = {
                            str(stp): {
                                "psnr": float(np.mean(vals["psnr"])), "ssim": float(np.mean(vals["ssim"])),
                                "lpips": float(np.mean(vals["lpips"])), "time": float(np.mean(vals["time"]))
                            } for stp, vals in trajectory_data.items()
                        }

                        print(f"  -> SUMMARY: PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim*100:.2f}% | LPIPS: {mean_lpips:.4f} | Avg Time: {mean_time:.2f}s")
                                    
                        grid_summary[grid_key] = {
                            "R": R, "lr": lr, "J": J, "eta": eta, "scope": scope,
                            "PSNR": mean_psnr, "SSIM": mean_ssim, "LPIPS": mean_lpips, "Time": mean_time
                        }
                        
                        detailed_export[grid_key] = {
                            "config": grid_summary[grid_key],
                            "mean_loss_trajectory": mean_loss_curve,
                            "step_convergence": mean_trajectory
                        }

                        # LIVE SAVING: Overwrites cleanly inside the loop so you can monitor live!
                        with open("ixi_adadiff_grid_summary.json", "w") as f:
                            json.dump(grid_summary, f, indent=4)
                        with open("ixi_adadiff_plotting_data.json", "w") as f:
                            json.dump(detailed_export, f, indent=4)

    # Plotting Automation at the very end
    print("\n🏆 SWEEP COMPLETE! Generating benchmark comparison figures...")
    configs = list(grid_summary.keys())
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IXI DiT-GAN Optimization Benchmark", fontsize=15, fontweight="bold")
    
    axs[0].barh(configs[:10], [grid_summary[c]["PSNR"] for c in configs[:10]], color="skyblue", edgecolor="black")
    axs[0].set_title("Top 10 Average PSNR (dB)"); axs[0].set_xlabel("dB")
    
    axs[1].barh(configs[:10], [grid_summary[c]["SSIM"]*100 for c in configs[:10]], color="lightgreen", edgecolor="black")
    axs[1].set_title("Top 10 Average SSIM (%)"); axs[1].set_xlabel("%")
    
    for c in configs[:3]:  
        traj = detailed_export[c]["step_convergence"]
        stps = sorted([int(k) for k in traj.keys()])
        t_vals = [traj[str(s)]["time"] for s in stps]
        p_vals = [traj[str(s)]["psnr"] for s in stps]
        axs[2].plot(t_vals, p_vals, marker="o", linewidth=2, label=f"J={grid_summary[c]['J']} | {grid_summary[c]['scope']}")
        
    axs[2].set_title("PSNR vs. Latency (Convergence Trajectory)")
    axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("PSNR (dB)")
    axs[2].legend(); axs[2].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig("ixi_adadiff_benchmark_summary.png", dpi=300); plt.close()
    
    if best_visual_candidate["true"] is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle(f"Qualitative Reconstruction & Error Analysis ({best_visual_candidate['title']})", fontsize=15, fontweight='bold')
        
        t_img = best_visual_candidate["true"]
        r_img = best_visual_candidate["recon"]
        e_img = np.abs(t_img - r_img)

        axes[0].imshow(t_img, cmap='gray', vmin=0, vmax=1); axes[0].set_title("Ground Truth ($x_{true}$)"); axes[0].axis("off")
        axes[1].imshow(best_visual_candidate["aliased"], cmap='gray', vmin=0, vmax=1); axes[1].set_title("Zero-Filled Aliased ($x_{init}$)"); axes[1].axis("off")
        axes[2].imshow(r_img, cmap='gray', vmin=0, vmax=1); axes[2].set_title("DiT Reconstructed ($\hat{x}_{final}$)"); axes[2].axis("off")
        im_err = axes[3].imshow(e_img, cmap='hot', vmin=0, vmax=0.15); axes[3].set_title("Absolute Error Map ($|x_{true} - \hat{x}|$)"); axes[3].axis("off")
        fig.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig("ixi_golden_trio_error_map.png", dpi=300); plt.close()
