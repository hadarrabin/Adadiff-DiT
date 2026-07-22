import os
import time
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. GLOBAL SINUSOIDAL EMBEDDING ENGINE (ADADIFF / DDPM STANDARD)
# =============================================================================
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Transforms integer diffusion stages into deterministic sinusoidal wave embeddings."""
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    args = timesteps.unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# =============================================================================
# 2. DATASET PROCESSING
# =============================================================================
class UnifiedMRIDataset(Dataset):
    def __init__(self, data_dir, dataset_type: int, virtual_coils: int = 5, acceleration: int = 4, input_size: int = 200):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.virtual_coils = virtual_coils
        self.acceleration = acceleration
        self.input_size = input_size 
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy') or f.endswith('.h5')])

    def __len__(self):
        return len(self.file_list)

    def _generate_mri_mask(self, cols, acceleration):
        mask = torch.zeros(cols, dtype=torch.float32)
        num_low_freqs = int(cols * 0.08)
        pad = (cols - num_low_freqs) // 2
        mask[pad:pad + num_low_freqs] = 1.0
        for i in range(0, cols, acceleration):
            mask[i] = 1.0
        return mask

    def _process_fastmri(self, filepath):
        with h5py.File(filepath, 'r') as hf:
            kspace = np.array(hf['kspace'], dtype=np.complex64)
        slice_data = kspace[kspace.shape[0] // 2]
        kspace_tensor = torch.from_numpy(slice_data)
        num_coils, h, w = kspace_tensor.shape

        flattened_kspace = kspace_tensor.view(num_coils, -1)
        U, _, _ = torch.linalg.svd(flattened_kspace, full_matrices=False)
        compressed_kspace = torch.matmul(U[:, :self.virtual_coils].H, flattened_kspace).view(self.virtual_coils, h, w)

        real_res = F.interpolate(compressed_kspace.real.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        imag_res = F.interpolate(compressed_kspace.imag.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        resized_kspace = torch.complex(real_res.squeeze(0), imag_res.squeeze(0))

        mask = self._generate_mri_mask(self.input_size, self.acceleration)
        resized_kspace = resized_kspace * mask.unsqueeze(0).unsqueeze(1)
        return torch.cat([resized_kspace.real, resized_kspace.imag], dim=0)

    def _process_ixi(self, filepath):
        matrix = np.load(filepath)
        tensor_data = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        
        # FIXED: Must normalize strictly to [-1.0, 1.0] for Gaussian Diffusion to work!
        min_val = tensor_data.min()
        max_val = tensor_data.max()
        tensor_data = ((tensor_data - min_val) / (max_val - min_val + 1e-8)) * 2.0 - 1.0
        return tensor_data

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        if self.dataset_type == 0:
            return self._process_fastmri(filepath)
        elif self.dataset_type == 1:
            return self._process_ixi(filepath)
        else:
            raise ValueError(f"Invalid dataset_type token: {self.dataset_type}")

# =============================================================================
# 3. ADADIFF SCHEDULER
# =============================================================================
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
# 4. GENERATOR ARCHITECTURE
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
        nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)

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
    def __init__(self, in_channels: int, input_size: int, patch_size: int = 2, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6):
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

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.final_linear.weight, 0.0)
        nn.init.constant_(self.final_linear.bias, 0.0)

    def forward(self, x: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_tokens = self.patchify(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp(get_timestep_embedding(t_tensor, embedding_dim=self.hidden_dim))
        for block in self.blocks: x_tokens = block(x_tokens, c)
        decoded = self.final_linear(self.final_ln(x_tokens))
        p = self.patch_size
        return decoded.view(B, H // p, W // p, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

# =============================================================================
# 5. DISCRIMINATOR ARCHITECTURE
# =============================================================================
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.time_project = nn.Linear(hidden_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.AvgPool2d(2, 2)) if in_channels != out_channels else nn.AvgPool2d(2, 2)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.conv1(x)) + self.time_project(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.downsample(self.act(self.conv2(out))) + identity

class AdaDiffDiscriminator(nn.Module):
    def __init__(self, dataset_type: int, virtual_coils: int = 5, hidden_dim: int = 256):
        super().__init__()
        base_channels = 2 if dataset_type == 1 else 2 * virtual_coils * 2
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim))
        ch = [base_channels, 64, 128, 256, 512, 512, 512]
        self.stages = nn.ModuleList([DiscriminatorBlock(ch[i], ch[i+1], hidden_dim) for i in range(6)])
        self.final_mlp = nn.Linear(512, 1)

    def forward(self, x_pair: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(get_timestep_embedding(t, embedding_dim=self.time_mlp[0].in_features))
        x = x_pair
        for stage in self.stages: x = stage(x, t_emb)
        return self.final_mlp(torch.mean(x, dim=[2, 3]))

# =============================================================================
# 6. PIPELINE INITIALIZATION & ENVIRONMENT CONFIGURATION
# =============================================================================
if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running accelerated pipeline on target device: {str(device).upper()}")

    # Training Configurations
    DATASET_TYPE = 1       # 1 = IXI
    VIRTUAL_COILS = 5      
    INPUT_SIZE = 200       # IXI image spatial dimension
    BATCH_SIZE = 4         # Stable server batch size
    NUM_EPOCHS = 50
    ACCUMULATION_STEPS = 2 
    PRINT_EVERY_N_BATCHES = 10
    SAVE_EVERY_N_EPOCHS = 1
    SAVE_EVERY_N_STEPS = 50

    DRIVE_IXI_TRAIN = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/train"
    CHECKPOINT_DIR = "/ugproj/sipl-prj10848/Idan/ixi_checkpoints_sinusoidal"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "adadiff_ixi_sinusoidal_latest.pt")

    in_channels_G = 2 * VIRTUAL_COILS if DATASET_TYPE == 0 else 1

    print("\nInitializing model parameters for high-speed IXI sinusoidal training...")
    dit_backbone = DiffusionTransformer(in_channels=in_channels_G, input_size=INPUT_SIZE, patch_size=2, hidden_dim=384, depth=12, num_heads=6).to(device)
    discriminator = AdaDiffDiscriminator(dataset_type=DATASET_TYPE, virtual_coils=VIRTUAL_COILS, hidden_dim=256).to(device)
    scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)

    # FIXED: TTUR Optimization
    optimizer_G = optim.AdamW(dit_backbone.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))

    # FIXED: Cosine Schedulers for smooth convergence
    scheduler_lr_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=NUM_EPOCHS, eta_min=1e-6)
    scheduler_lr_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=NUM_EPOCHS, eta_min=1e-6)

    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    start_epoch, start_step = 1, 0

    if os.path.exists(CHECKPOINT_FILE):
        print(f"📦 Found step checkpoint at: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        dit_backbone.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
        scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
        
        if 'scheduler_G_state_dict' in checkpoint:
            scheduler_lr_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_lr_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        start_epoch = checkpoint.get('epoch', 1)
        start_step = checkpoint.get('step', 0)
        print(f"🚀 Weights successfully restored! Resuming exactly from Epoch {start_epoch}, Step {start_step}.\n")
    else:
        print("Beginning sinusoidal training from scratch in clean directory...")

    train_dataset = UnifiedMRIDataset(data_dir=DRIVE_IXI_TRAIN, dataset_type=1, input_size=INPUT_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print(f"Loaded training setup containing {len(train_dataset)} scans.")
    print(f"Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS} | AMP Enabled: True")

    # =============================================================================
    # 9. AMP-ACCELERATED ADV_TRAINING LOOP
    # =============================================================================
    dit_backbone.train()
    discriminator.train()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_loss_G, epoch_loss_D, epoch_r1 = 0.0, 0.0, 0.0
        running_loss_G, running_loss_D, running_r1 = 0.0, 0.0, 0.0
        start_time = time.time()

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        for batch_idx, clean_x0 in enumerate(train_loader):
            current_step = (batch_idx + 1) // ACCUMULATION_STEPS
            if epoch == start_epoch and batch_idx < (start_step * ACCUMULATION_STEPS):
                continue

            stage_idx = torch.randint(0, scheduler.num_stages - 1, (1,)).item()
            clean_x0 = clean_x0.to(device, non_blocking=True)
            batch_size = clean_x0.shape[0]

            t_step_parent = scheduler.stages[stage_idx + 1]
            t_step_target = scheduler.stages[stage_idx]
            t_parent_tensor = torch.full((batch_size,), t_step_parent, dtype=torch.long, device=device)
            t_target_tensor = torch.full((batch_size,), t_step_target, dtype=torch.long, device=device)

            x_parent, _ = scheduler.add_noise(clean_x0, stage_idx + 1)
            x_real_target, _ = scheduler.add_noise(clean_x0, stage_idx)

            # ---------------------------------------------------------------------
            # CRITIC STREAM
            # ---------------------------------------------------------------------
            x_real_target.requires_grad_(True)
            with torch.cuda.amp.autocast():
                pred_real = discriminator(torch.cat([x_parent, x_real_target], dim=1), t_target_tensor)
                with torch.no_grad():
                    x_fake_target = dit_backbone(x_parent, t_parent_tensor) # Condition on parent!
                pred_fake = discriminator(torch.cat([x_parent, x_fake_target.detach()], dim=1), t_target_tensor)

                loss_D_real = F.softplus(-pred_real).mean()
                loss_D_fake = F.softplus(pred_fake).mean()

            grad_real = torch.autograd.grad(outputs=pred_real.sum(), inputs=x_real_target, create_graph=True, retain_graph=True, only_inputs=True)[0]
            r1_penalty = (grad_real.view(batch_size, -1).norm(2, dim=1) ** 2).mean()

            loss_D = (loss_D_real + loss_D_fake + (0.5 * 10.0 * r1_penalty)) / ACCUMULATION_STEPS
            scaler_D.scale(loss_D).backward()

            # ---------------------------------------------------------------------
            # GENERATOR STREAM
            # ---------------------------------------------------------------------
            with torch.cuda.amp.autocast():
                x_fake_train = dit_backbone(x_parent, t_parent_tensor) # Condition on parent!
                pred_fake_train = discriminator(torch.cat([x_parent, x_fake_train], dim=1), t_target_tensor)
                
                loss_G_adv = F.softplus(-pred_fake_train).mean() / ACCUMULATION_STEPS
                
                # FIXED: Added massive Spatial Content Loss to prevent hallucinations!
                loss_G_spatial = F.l1_loss(x_fake_train, x_real_target) / ACCUMULATION_STEPS
                
                loss_G_step = loss_G_adv + (20.0 * loss_G_spatial)

            scaler_G.scale(loss_G_step).backward()

            running_loss_G += loss_G_step.item() * ACCUMULATION_STEPS
            running_loss_D += (loss_D_real + loss_D_fake).item()
            running_r1 += r1_penalty.item()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler_G.unscale_(optimizer_G)
                scaler_D.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(dit_backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                scaler_G.step(optimizer_G); scaler_G.update()
                scaler_D.step(optimizer_D); scaler_D.update()
                optimizer_G.zero_grad(); optimizer_D.zero_grad()

                epoch_loss_G += running_loss_G / ACCUMULATION_STEPS
                epoch_loss_D += running_loss_D / ACCUMULATION_STEPS
                epoch_r1 += running_r1 / ACCUMULATION_STEPS

                if current_step % SAVE_EVERY_N_STEPS == 0:
                    step_path = os.path.join(CHECKPOINT_DIR, f"adadiff_ixi_sinusoidal_latest.pt")
                    torch.save({
                        'epoch': epoch, 'step': current_step,
                        'generator_state_dict': dit_backbone.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'scaler_G_state_dict': scaler_G.state_dict(),
                        'scaler_D_state_dict': scaler_D.state_dict(),
                        'scheduler_G_state_dict': scheduler_lr_G.state_dict(),
                        'scheduler_D_state_dict': scheduler_lr_D.state_dict(),
                    }, step_path)

                if current_step % PRINT_EVERY_N_BATCHES == 0:
                    curr_lr = optimizer_G.param_groups[0]['lr']
                    print(f" -> Batch {current_step} | LR: {curr_lr:.6f} | Loss_G: {running_loss_G/ACCUMULATION_STEPS:.4f} | Loss_D: {running_loss_D/ACCUMULATION_STEPS:.4f} | R1: {running_r1/ACCUMULATION_STEPS:.4f}")

                running_loss_G, running_loss_D, running_r1 = 0.0, 0.0, 0.0

        num_steps = len(train_loader) // ACCUMULATION_STEPS
        avg_G = epoch_loss_G / num_steps if num_steps > 0 else 0
        avg_D = epoch_loss_D / num_steps if num_steps > 0 else 0
        avg_r1 = epoch_r1 / num_steps if num_steps > 0 else 0
        elapsed = time.time() - start_time

        print(f"\n=========================================================")
        print(f"🏁 IXI EPOCH {epoch}/{NUM_EPOCHS} COMPLETED ({elapsed:.1f}s)")
        print(f"=========================================================")
        print(f" -> Average Generator Loss : {avg_G:.4f}")
        print(f" -> Average Discriminator  : {avg_D:.4f}")
        print(f" -> Average R1 Stability   : {avg_r1:.4f}")
        print(f"=========================================================\n")

        # FIXED: Smooth learning rate decay applied
        scheduler_lr_G.step()
        scheduler_lr_D.step()

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            epoch_path = os.path.join(CHECKPOINT_DIR, f"adadiff_ixi_sinusoidal_epoch_{epoch}.pt")
            save_dict = {
                'epoch': epoch, 'step': num_steps,
                'generator_state_dict': dit_backbone.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scaler_G_state_dict': scaler_G.state_dict(),
                'scaler_D_state_dict': scaler_D.state_dict(),
                'scheduler_G_state_dict': scheduler_lr_G.state_dict(),
                'scheduler_D_state_dict': scheduler_lr_D.state_dict(),
            }
            torch.save(save_dict, epoch_path)
            torch.save(save_dict, CHECKPOINT_FILE)
            print(f"💾 Saved permanent end-of-epoch IXI checkpoint to: {epoch_path}\n")

    print("⚡ Training run successfully completed!")
