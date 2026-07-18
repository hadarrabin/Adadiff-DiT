# fastmri_grid_search_clean.py
import os
import time
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.metrics
import lpips

# Native imports from models.py (Ensure models.py is copied to this directory)
from models import DiffusionTransformer

# =========================================================================
# 1. MODEL WEIGHTS LOADER (ALIGNED WITH 10-CHANNEL TRAINING)
# =========================================================================

def load_trained_generator(checkpoint_path="/ugproj/sipl-prj10848/Idan/fastmri_checkpoints/adadiff_fastmri_epoch_20_final.pt", device="cuda"):
    """
    Loads your actual trained DiffusionTransformer architecture for fastMRI.
    Aligned with your training code settings: DATASET_TYPE=0, VIRTUAL_COILS=5 (in_channels=10).
    """
    print(f"🔄 Initializing DiffusionTransformer and loading weights from: {checkpoint_path}...")
    
    # Matching your exact training setup:
    model = DiffusionTransformer(
        in_channels=10,  # 2 channels * 5 virtual coils
        input_size=256,
        patch_size=4,
        hidden_dim=384,
        depth=12,
        num_heads=6
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handles nested dictionary weights from your training saver
        if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print("✅ Real fastMRI generator weights restored successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Raw state dict loaded successfully!")
    else:
        raise FileNotFoundError(
            f"❌ CRITICAL ERROR: Checkpoint file '{checkpoint_path}' not found! "
            "Please confirm your training run output path."
        )
        
    return model.to(device)

# =========================================================================
# 2. RAW H5 PARSER & COIL SENSITIVITY ESTIMATOR (FROM TRAINING PIPELINE)
# =========================================================================

def process_raw_h5_file(filepath, virtual_coils=5, input_size=256):
    """
    Loads a raw fastMRI .h5 file, extracts the center slice, performs Geometric Coil Compression (GCC),
    resizes to input_size, and computes self-consistent coil sensitivity maps (S-maps) using the central k-space.
    """
    with h5py.File(filepath, 'r') as hf:
        kspace = np.array(hf['kspace'], dtype=np.complex64)

    mid_slice = kspace.shape[0] // 2
    slice_data = kspace[mid_slice]

    kspace_tensor = torch.from_numpy(slice_data)
    num_coils, h, w = kspace_tensor.shape

    # --- GEOMETRIC COIL COMPRESSION (GCC) ---
    flattened_kspace = kspace_tensor.view(num_coils, -1)
    U, S, Vh = torch.linalg.svd(flattened_kspace, full_matrices=False)
    compression_matrix = U[:, :virtual_coils].H

    compressed_flattened = torch.matmul(compression_matrix, flattened_kspace)
    compressed_kspace = compressed_flattened.view(virtual_coils, h, w)

    # --- RESIZE K-SPACE TO A FIXED INPUT_SIZE ---
    compressed_kspace_real = compressed_kspace.real.unsqueeze(0)
    compressed_kspace_imag = compressed_kspace.imag.unsqueeze(0)

    resized_kspace_real = F.interpolate(compressed_kspace_real, size=(input_size, input_size), mode='bilinear', align_corners=False)
    resized_kspace_imag = F.interpolate(compressed_kspace_imag, size=(input_size, input_size), mode='bilinear', align_corners=False)

    resized_kspace = torch.complex(resized_kspace_real.squeeze(0), resized_kspace_imag.squeeze(0))

    # --- GENERATE PHYSICAL SENSE MAPS (S-MAPS) FROM LOW-FREQUENCY CENTER ---
    # Crop central low frequencies to estimate coil sensitivity profiles (SENSE calibration)
    center_fraction = 0.08
    num_low_freq = int(input_size * center_fraction)
    pad = (input_size - num_low_freq) // 2
    
    # Calculate low-frequency central region
    low_freq_kspace = torch.zeros_like(resized_kspace)
    low_freq_kspace[:, :, pad:pad+num_low_freq] = resized_kspace[:, :, pad:pad+num_low_freq]
    
    # Inverse FFT to obtain low-frequency (smoothed) coil images
    coil_images_low = torch.fft.ifft2(torch.fft.ifftshift(low_freq_kspace, dim=(-2, -1)), dim=(-2, -1))
    
    # Compute Root-Sum-of-Squares (RSS) of low frequency images
    rss = torch.sqrt(torch.sum(torch.abs(coil_images_low)**2, dim=0, keepdim=True)) + 1e-8
    
    # Sensitivities = Smooth Coil Image / RSS
    S_maps = coil_images_low / rss
    
    # Compute the ground-truth combined complex image from full k-space via SENSE adjoint
    full_coil_images = torch.fft.ifft2(torch.fft.ifftshift(resized_kspace, dim=(-2, -1)), dim=(-2, -1))
    x_true = torch.sum(torch.conj(S_maps) * full_coil_images, dim=0)
    
    # Normalize ground truth to keep values on a standard stable [0, 1] scale
    x_true = x_true / (torch.max(torch.abs(x_true)) + 1e-8)
    
    return x_true, S_maps

# =========================================================================
# 3. MULTI-COIL IMAGING OPERATORS
# =========================================================================

def get_fastmri_imaging_operator(x_true, S_maps, R=4):
    """
    Simulates multi-coil forward acquisition: y_c = Mask * FFT(S_c * x)
    """
    C, H, W = S_maps.shape
    
    # 1. Multiply combined image by each coil's sensitivity map
    coil_images = S_maps * x_true.unsqueeze(0)
    
    # 2. Convert to k-space using 2D FFT
    kspace = torch.fft.fftshift(torch.fft.fft2(coil_images, dim=(-2, -1)), dim=(-2, -1))
    
    # 3. Generate deterministic 1D sampling mask (R=4) matching UnifiedMRIDataset
    mask = torch.zeros(W, device=x_true.device)
    center_fraction = 0.08  # 8% central ACS
    num_low_freq = int(W * center_fraction)
    mask[W//2 - num_low_freq//2 : W//2 + num_low_freq//2] = 1.0
    
    stride = R
    for i in range(0, W, stride):
        mask[i] = 1.0
        
    mask_2d = mask.unsqueeze(0).repeat(H, 1)
    
    # 4. Under-sample multi-coil k-space
    y = kspace * mask_2d.unsqueeze(0)
    return y, mask_2d


def adjoint_sense_operator(y, S_maps):
    """
    Reconstructs a coil-combined complex image from multi-coil k-space data.
    Applying SENSE adjoint: x_adjoint = Sum_c ( S_c* . IFFT(y_c) )
    """
    shifted_y = torch.fft.ifftshift(y, dim=(-2, -1))
    coil_images_recon = torch.fft.ifft2(shifted_y, dim=(-2, -1))
    
    S_maps_conj = torch.conj(S_maps)
    x_combined = torch.sum(S_maps_conj * coil_images_recon, dim=0)
    return x_combined

# =========================================================================
# 4. PRIOR ADAPTATION ENGINE
# =========================================================================

def run_fastmri_prior_adaptation(base_model, y, mask, S_maps, J, lr, device="cuda"):
    """
    Runs prior adaptation. Correctly converts complex reconstructions into 
    the 10-channel representation [Real, Imag] for your 5 virtual coils.
    """
    import copy
    netG = copy.deepcopy(base_model).to(device)
    netG.train()
    
    optimizer = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))
    
    # 1. Compute physical initial SENSE image
    x_init_complex = adjoint_sense_operator(y, S_maps)
    
    # 2. Re-project back to the 5 virtual coils to match the 10-channel training input
    # Shape: [5, H, W] complex
    virtual_coil_representation = S_maps * x_init_complex.unsqueeze(0)
    
    # 3. Concatenate Real and Imag components along channel dim -> [10, H, W]
    x_init_real_imag = torch.cat([virtual_coil_representation.real, virtual_coil_representation.imag], dim=0)
    x_init_input = x_init_real_imag.unsqueeze(0).float() # [1, 10, H, W]
    
    # Generate time step embedding vector for t=0 (shape: [1, hidden_dim])
    t_emb = torch.zeros(1, netG.hidden_dim, device=device)
    
    for step in range(J):
        optimizer.zero_grad()
        
        # Generator processes 10-channel complex representation along with t_emb
        x_recon_10ch = netG(x_init_input, t_emb) # [1, 10, H, W]
        
        # Split back into Real and Imag (first 5 real, last 5 imag)
        recon_real, recon_imag = torch.chunk(x_recon_10ch.squeeze(0), 2, dim=0)
        recon_complex_coils = torch.complex(recon_real, recon_imag) # [5, H, W]
        
        # Combine multi-coil representation back into a single image using S_maps conjugate
        S_maps_conj = torch.conj(S_maps)
        x_recon_complex = torch.sum(S_maps_conj * recon_complex_coils, dim=0) # [H, W]
        
        # Apply physical Coil Sensitivity multiplication to obtain raw data-domain representation
        coil_recon_images = S_maps * x_recon_complex.unsqueeze(0)
        
        # Bring back to k-space
        kspace_recon = torch.fft.fftshift(torch.fft.fft2(coil_recon_images, dim=(-2, -1)), dim=(-2, -1))
        y_recon = kspace_recon * mask.unsqueeze(0)
        
        # Multi-coil complex L1 Data Consistency Loss in k-space
        loss = torch.mean(torch.abs(y_recon - y))
        
        loss.backward()
        optimizer.step()
        
    # Reconstruct final step
    recon_real, recon_imag = torch.chunk(x_recon_10ch.squeeze(0), 2, dim=0)
    recon_complex_coils = torch.complex(recon_real, recon_imag)
    final_complex = torch.sum(torch.conj(S_maps) * recon_complex_coils, dim=0).detach()
    
    del netG, optimizer
    torch.cuda.empty_cache()
    
    # Return normalized magnitude image for evaluation
    return torch.clamp(torch.abs(final_complex), 0.0, 1.0)


def evaluate_slice(x_recon, x_true, lpips_model, device):
    """
    Calculates physical and perceptual evaluation metrics between recon and ground truth magnitude.
    """
    rec_np = x_recon.cpu().numpy()
    true_np = torch.abs(x_true).cpu().numpy()
    
    psnr = skimage.metrics.peak_signal_noise_ratio(true_np, rec_np, data_range=1.0)
    ssim = skimage.metrics.structural_similarity(true_np, rec_np, data_range=1.0)
    
    # Prepare normalized magnitude vectors for LPIPS [1, 3, H, W] scaled to [-1, 1]
    tensor_rec = (x_recon.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2.0) - 1.0
    tensor_true = (torch.abs(x_true).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2.0) - 1.0
    
    with torch.no_grad():
        lpips_score = lpips_model(tensor_rec.to(device), tensor_true.to(device)).item()
        
    return psnr, ssim, lpips_score

# =========================================================================
# 5. RUNNING THE SEARCH
# =========================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")
    
    # Initialize LPIPS using AlexNet
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # Point to your actual fastmri-trained weight path
    fastmri_weights = "/ugproj/sipl-prj10848/Idan/fastmri_checkpoints/adadiff_fastmri_epoch_20_final.pt"
    base_model = load_trained_generator(checkpoint_path=fastmri_weights, device=device)
    
    # Loading physical fastMRI validation slices and sensitivity maps
    print("\n📁 Scanning mixed fastMRI validation folder...")
    val_base_dir = "/ugproj/sipl-prj10848/Idan/fastMRI_adadiff/val/val" # Corrected absolute path
    val_set = {"T1": [], "T2": [], "FLAIR": []}
    
    # Aligned with IXI sample counts for overnight training
    max_samples_per_contrast = 10
    
    if os.path.exists(val_base_dir):
        # Scan all raw H5 files inside the folder
        all_files = sorted([f for f in os.listdir(val_base_dir) if f.endswith('.h5')])
        
        for f in all_files:
            file_path = os.path.join(val_base_dir, f)
            
            # Identify the contrast by checking the filename (case-insensitive)
            filename_upper = f.upper()
            if "T1" in filename_upper or "T1PR" in filename_upper:
                target_contrast = "T1"
            elif "T2" in filename_upper:
                target_contrast = "T2"
            elif "FLAIR" in filename_upper:
                target_contrast = "FLAIR"
            else:
                # Fallback skip if the file name doesn't match any contrast category
                continue
            
            # Skip if we already have 10 files for this contrast category
            if len(val_set[target_contrast]) >= max_samples_per_contrast:
                continue
                
            # Process the raw H5 file to extract x_true and S_maps
            try:
                x_true_cpu, S_maps_cpu = process_raw_h5_file(file_path, virtual_coils=5, input_size=256)
                val_set[target_contrast].append((x_true_cpu, S_maps_cpu))
            except Exception as e:
                print(f"  ⚠️ Skipping file {f} due to read error: {e}")
            
        for contrast, samples in val_set.items():
            print(f"  -> Successfully processed and loaded {len(samples)} real raw .h5 scans for {contrast}")
    else:
        raise FileNotFoundError(f"❌ CRITICAL ERROR: Validation directory '{val_base_dir}' not found!")

    # Multi-Coil Optimization search parameters
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
                for idx, (x_true_cpu, S_maps_cpu) in enumerate(samples):
                    x_true = x_true_cpu.to(device)
                    S_maps = S_maps_cpu.to(device)
                    
                    # Under-sample multi-coil k-space (using deterministic mask)
                    y, mask = get_fastmri_imaging_operator(x_true, S_maps, R=4)
                    
                    # Measure adaptation speed
                    t0 = time.time()
                    x_recon = run_fastmri_prior_adaptation(base_model, y, mask, S_maps, J=J, lr=lr, device=device)
                    elapsed = time.time() - t0
                    
                    # Calculate metrics against complex ground-truth
                    psnr, ssim, lpips_score = evaluate_slice(x_recon, x_true, loss_fn_alex, device)
                    
                    run_psnrs.append(psnr)
                    run_ssims.append(ssim)
                    run_lpips.append(lpips_score)
                    run_times.append(elapsed)
                    
                    print(f"  [{contrast}] Slice {idx+1}/{len(samples)} -> PSNR: {psnr:.2f} dB | SSIM: {ssim*100:.2f}% | LPIPS: {lpips_score:.4f} | Time: {elapsed:.2f}s")
                    
            if len(run_psnrs) == 0:
                print(f"  ⚠️ No active samples loaded for config: {grid_key}. Skipping...")
                continue

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
    # 6. MULTI-CRITERIA UTILITY OPTIMIZATION
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

    best_params_path = "best_hyperparameters_fastmri.json"
    with open(best_params_path, "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(f"\n🏆 WINNING CONFIGURATION FOR fastMRI: LR = {best_config['lr']} | J = {best_config['J']} Steps")
    print(f"Optimal parameter file exported to '{best_params_path}'")

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
    fig.suptitle("DiT Hyperparameter Optimization Dashboard (fastMRI Validation)", fontsize=16, fontweight='bold')
    
    # PSNR Bar Graph
    axs[0, 0].bar(configs, psnrs, color='skyblue', edgecolor='black')
    axs[0, 0].set_title("Average PSNR (Higher is Better)")
    axs[0, 0].set_ylabel("PSNR (dB)")
    axs[0, 0].tick_params(axis='x', rotation=15)
    
    # SSIM Bar Graph
    axs[0, 1].bar(configs, ssims, color='lightgreen', edgecolor='black')
    axs[0, 1].set_title("Average SSIM (Higher is Better)")
    axs[0, 1].set_ylabel("SSIM Score")
    axs[0, 1].tick_params(axis='x', rotation=15)
    
    # LPIPS Bar Graph
    axs[1, 0].bar(configs, lpipss, color='salmon', edgecolor='black')
    axs[1, 0].set_title("Average LPIPS (Lower is Better)")
    axs[1, 0].set_ylabel("Perceptual Distance")
    axs[1, 0].tick_params(axis='x', rotation=15)
    
    # Inference Time Bar Graph
    axs[1, 1].bar(configs, times, color='plum', edgecolor='black')
    axs[1, 1].set_title("Inference Latency (Lower is Better)")
    axs[1, 1].set_ylabel("Inference Time (seconds/slice)")
    axs[1, 1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig("fastmri_grid_search_metrics.png", dpi=300)
    print("💾 Dashboard metrics successfully plotted and saved to 'fastmri_grid_search_metrics.png'!")

    