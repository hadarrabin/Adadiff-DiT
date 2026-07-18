# train_ixi_gan.py
import os
import re
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
# 1. DATASET PROCESSING
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

        stride = acceleration
        for i in range(0, cols, stride):
            mask[i] = 1.0
        return mask

    def _process_fastmri(self, filepath):
        with h5py.File(filepath, 'r') as hf:
            kspace = np.array(hf['kspace'], dtype=np.complex64)

        mid_slice = kspace.shape[0] // 2
        slice_data = kspace[mid_slice]

        kspace_tensor = torch.from_numpy(slice_data)
        num_coils, h, w = kspace_tensor.shape

        # --- GEOMETRIC COIL COMPRESSION (GCC) ---
        flattened_kspace = kspace_tensor.view(num_coils, -1)
        U, S, Vh = torch.linalg.svd(flattened_kspace, full_matrices=False)
        compression_matrix = U[:, :self.virtual_coils].H

        compressed_flattened = torch.matmul(compression_matrix, flattened_kspace)
        compressed_kspace = compressed_flattened.view(self.virtual_coils, h, w)

        # --- RESIZE K-SPACE TO A FIXED INPUT_SIZE ---
        compressed_kspace_real = compressed_kspace.real.unsqueeze(0)
        compressed_kspace_imag = compressed_kspace.imag.unsqueeze(0)

        resized_kspace_real = F.interpolate(compressed_kspace_real, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        resized_kspace_imag = F.interpolate(compressed_kspace_imag, size=(input_size, input_size), mode='bilinear', align_corners=False)

        resized_kspace = torch.complex(resized_kspace_real.squeeze(0), resized_kspace_imag.squeeze(0))

        # --- ACCELERATION MASK APPLICATION ---
        mask = self._generate_mri_mask(self.input_size, self.acceleration)
        resized_kspace = resized_kspace * mask.unsqueeze(0).unsqueeze(1)

        tensor_data = torch.cat([resized_kspace.real, resized_kspace.imag], dim=0)
        return tensor_data

    def _process_ixi(self, filepath):
        """
        Loads IXI magnitude scans and scales values cleanly to [0, 1] 
        to ensure training stability.
        """
        matrix = np.load(filepath)
        tensor_data = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        
        # FIX: Added dynamic min-max normalization
        min_val = tensor_data.min()
        max_val = tensor_data.max()
        tensor_data = (tensor_data - min_val) / (max_val - min_val + 1e-8)
        
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
# 2. ADADIFF SCHEDULER
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
        alpha_t_bar = self.alpha_bar[stage_idx]
        noise = torch.randn_like(x_0)
        noisy_sample = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1.0 - alpha_t_bar) * noise
        return noisy_sample, noise

# =============================================================================
# 3. GENERATOR ARCHITECTURE (Diffusion Transformer)
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

        nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)

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

        nn.init.normal_(self.pos_embed, std=0.02)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.final_linear.weight, 0.0)
        nn.init.constant_(self.final_linear.bias, 0.0)

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

# =============================================================================
# 4. DISCRIMINATOR ARCHITECTURE
# =============================================================================
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.time_project = nn.Linear(hidden_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
        else:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.conv1(x))
        bias = self.time_project(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = out + bias
        out = self.act(self.conv2(out))
        out = self.downsample(out)
        return out + identity

class AdaDiffDiscriminator(nn.Module):
    def __init__(self, dataset_type: int, virtual_coils: int = 5, hidden_dim: int = 256):
        super().__init__()
        if dataset_type == 1:
            base_channels = 2
        else:
            base_channels = 2 * virtual_coils * 2

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        ch = [base_channels, 64, 128, 256, 512, 512, 512]
        self.stages = nn.ModuleList([
            DiscriminatorBlock(ch[i], ch[i+1], hidden_dim) for i in range(6)
        ])
        self.final_mlp = nn.Linear(512, 1)

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        half_dim = embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        args = timesteps.unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, x_pair: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_sin = self.get_timestep_embedding(t, embedding_dim=self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_sin)

        x = x_pair
        for stage in self.stages:
            x = stage(x, t_emb)

        x = torch.mean(x, dim=[2, 3])
        return self.final_mlp(x)

# =============================================================================
# 5. UNIFIED ADVERSARIAL STEP
# =============================================================================
def train_step_adadiff(dit_model, discriminator, scheduler, clean_x0, stage_idx,
                       optimizer_G, optimizer_D, r1_gamma=10.0, device="cuda"):
    dit_model.to(device)
    discriminator.to(device)
    clean_x0 = clean_x0.to(device)

    batch_size = clean_x0.shape[0]

    t_step_parent = scheduler.stages[stage_idx + 1] 
    t_step_target = scheduler.stages[stage_idx]     

    t_parent_tensor = torch.full((batch_size,), t_step_parent, dtype=torch.long, device=device)
    t_target_tensor = torch.full((batch_size,), t_step_target, dtype=torch.long, device=device)

    x_parent, _ = scheduler.add_noise(clean_x0, stage_idx + 1)
    x_real_target, _ = scheduler.add_noise(clean_x0, stage_idx)

    # -------------------------------------------------------------------------
    # PART A: Optimize Discriminator Network
    # -------------------------------------------------------------------------
    optimizer_D.zero_grad()
    x_real_target.requires_grad_(True)

    real_pair = torch.cat([x_parent, x_real_target], dim=1)
    pred_real = discriminator(real_pair, t_target_tensor)

    with torch.no_grad():
        t_emb = torch.randn(batch_size, dit_model.hidden_dim, device=device)
        x_fake_target = dit_model(x_parent, t_emb)

    fake_pair = torch.cat([x_parent, x_fake_target.detach()], dim=1)
    pred_fake = discriminator(fake_pair, t_target_tensor)

    loss_D_real = F.softplus(-pred_real).mean()
    loss_D_fake = F.softplus(pred_fake).mean()

    grad_real = torch.autograd.grad(
        outputs=pred_real.sum(), inputs=x_real_target,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    r1_penalty = (grad_real.view(batch_size, -1).norm(2, dim=1) ** 2).mean()

    loss_D = loss_D_real + loss_D_fake + (0.5 * r1_gamma * r1_penalty)
    loss_D.backward()
    optimizer_D.step()

    # -------------------------------------------------------------------------
    # PART B: Optimize Generator (Diffusion Transformer prior)
    # -------------------------------------------------------------------------
    optimizer_G.zero_grad()

    t_emb_train = torch.randn(batch_size, dit_model.hidden_dim, device=device)
    x_fake_train = dit_model(x_parent, t_emb_train)

    fake_pair_train = torch.cat([x_parent, x_fake_train], dim=1)
    pred_fake_train = discriminator(fake_pair_train, t_target_tensor)

    loss_G = F.softplus(-pred_fake_train).mean()
    loss_G.backward()
    optimizer_G.step()

    return loss_G.item(), loss_D.item(), r1_penalty.item()

# =============================================================================
# 6. PIPELINE INITIALIZATION & ENVIRONMENT CONFIGURATION
# =============================================================================
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running pipeline on target device: {str(device).upper()}")

DATASET_TYPE = 1       # 1 = IXI
VIRTUAL_COILS = 5      # (Unused for IXI but defined for network consistency)
INPUT_SIZE = 200       # IXI image spatial dimension
BATCH_SIZE = 4         # Stable server batch size (scaled up from 2 to stabilize GAN gradients)

if DATASET_TYPE == 0:
    in_channels_G = 2 * VIRTUAL_COILS
else:
    in_channels_G = 1  # 1-channel magnitude inputs for IXI

print("\nInitializing model parameters for IXI training...")

# Initialize Generator prior
dit_backbone = DiffusionTransformer(
    in_channels=in_channels_G,
    input_size=INPUT_SIZE,
    patch_size=2,             # FIX: Changed from 4 to 2 to eliminate spatial bottlenecks
    hidden_dim=384,
    depth=12,
    num_heads=6
).to(device)

# Initialize Discriminator
discriminator = AdaDiffDiscriminator(
    dataset_type=DATASET_TYPE,
    virtual_coils=VIRTUAL_COILS,
    hidden_dim=256
).to(device)

# Initialize large-step scheduler
scheduler = AdaDiffScheduler(step_size_k=125, total_steps_T=1000)

optimizer_G = optim.AdamW(dit_backbone.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# =============================================================================
# 7. DYNAMIC CHECKPOINT LOADER (AUTO-RESUME DETECTION)
# =============================================================================
start_epoch = 5
start_step = 0

CHECKPOINT_FILE = "/ugproj/sipl-prj10848/Idan/ixi_checkpoints/adadiff_ixi_epoch_5_step_100.pt"

if os.path.exists(CHECKPOINT_FILE):
    print(f"📦 Found step checkpoint at: {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

    dit_backbone.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'step' in checkpoint:
        start_step = checkpoint['step']
        
    print(f"🚀 Weights successfully restored! Resuming exactly from Epoch {start_epoch}, Step {start_step}.\n")
else:
    print(f"⚠️ Checkpoint file NOT found at: {CHECKPOINT_FILE}")
    print("Beginning training from scratch...")

# =============================================================================
# 8. TRAINING CONFIGURATIONS & DATALOADER
# =============================================================================
DRIVE_IXI_TRAIN = "/ugproj/sipl-prj10848/Idan/IXI_adadiff/train"
CHECKPOINT_DIR = "/ugproj/sipl-prj10848/Idan/ixi_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_EPOCHS = 20
ACCUMULATION_STEPS = 2         # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS = 8
PRINT_EVERY_N_BATCHES = 10
SAVE_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_STEPS = 50        

# Load the IXI training dataset
train_dataset = UnifiedMRIDataset(data_dir=DRIVE_IXI_TRAIN, dataset_type=1, input_size=INPUT_SIZE)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,     
    shuffle=True,
    num_workers=4,  
    drop_last=True
)

print(f"Loaded training setup containing {len(train_dataset)} scans.")
print(f"Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")

# =============================================================================
# 9. ADV_TRAINING LOOP (WITH STEP-SKIPPING LOGIC)
# =============================================================================
dit_backbone.train()
discriminator.train()

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_r1 = 0.0

    running_loss_G = 0.0
    running_loss_D = 0.0
    running_r1 = 0.0

    start_time = time.time()

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    for batch_idx, clean_x0 in enumerate(train_loader):
        current_step = (batch_idx + 1) // ACCUMULATION_STEPS
        
        # --- THE STEP-SKIPPING ENGINE ---
        if epoch == start_epoch and current_step <= start_step:
            continue

        stage_idx = torch.randint(0, scheduler.num_stages - 1, (1,)).item()

        clean_x0 = clean_x0.to(device)
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
        real_pair = torch.cat([x_parent, x_real_target], dim=1)
        pred_real = discriminator(real_pair, t_target_tensor)

        with torch.no_grad():
            t_emb = torch.randn(batch_size, dit_backbone.hidden_dim, device=device)
            x_fake_target = dit_backbone(x_parent, t_emb)

        fake_pair = torch.cat([x_parent, x_fake_target.detach()], dim=1)
        pred_fake = discriminator(fake_pair, t_target_tensor)

        loss_D_real = torch.nn.functional.softplus(-pred_real).mean()
        loss_D_fake = torch.nn.functional.softplus(pred_fake).mean()

        grad_real = torch.autograd.grad(
            outputs=pred_real.sum(), inputs=x_real_target,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        r1_penalty = (grad_real.view(batch_size, -1).norm(2, dim=1) ** 2).mean()

        loss_D = (loss_D_real + loss_D_fake + (0.5 * 10.0 * r1_penalty)) / ACCUMULATION_STEPS
        loss_D.backward()

        # ---------------------------------------------------------------------
        # GENERATOR STREAM
        # ---------------------------------------------------------------------
        t_emb_train = torch.randn(batch_size, dit_backbone.hidden_dim, device=device)
        x_fake_train = dit_backbone(x_parent, t_emb_train)

        fake_pair_train = torch.cat([x_parent, x_fake_train], dim=1)
        pred_fake_train = discriminator(fake_pair_train, t_target_tensor)

        loss_G_step = torch.nn.functional.softplus(-pred_fake_train).mean() / ACCUMULATION_STEPS
        loss_G_step.backward()

        running_loss_G += loss_G_step.item() * ACCUMULATION_STEPS
        running_loss_D += (loss_D_real + loss_D_fake).item()
        running_r1 += r1_penalty.item()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer_G.step()
            optimizer_D.step()

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            epoch_loss_G += running_loss_G / ACCUMULATION_STEPS
            epoch_loss_D += running_loss_D / ACCUMULATION_STEPS
            epoch_r1 += running_r1 / ACCUMULATION_STEPS

            if current_step % SAVE_EVERY_N_STEPS == 0:
                step_path = os.path.join(CHECKPOINT_DIR, f"adadiff_ixi_epoch_{epoch}_step_{current_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'generator_state_dict': dit_backbone.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, step_path)
                print(f"💾 [SAFETY SAVE] Saved checkpoint mid-epoch: {step_path}")

            if current_step % PRINT_EVERY_N_BATCHES == 0:
                print(f" -> Batch {current_step} | "
                      f"Loss_G: {running_loss_G/ACCUMULATION_STEPS:.4f} | "
                      f"Loss_D: {running_loss_D/ACCUMULATION_STEPS:.4f} | "
                      f"R1: {running_r1/ACCUMULATION_STEPS:.4f}")

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_r1 = 0.0

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

    if epoch % SAVE_EVERY_N_EPOCHS == 0:
        epoch_path = os.path.join(CHECKPOINT_DIR, f"adadiff_ixi_epoch_{epoch}_final.pt")
        torch.save({
            'epoch': epoch,
            'step': num_steps,
            'generator_state_dict': dit_backbone.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, epoch_path)
        print(f"💾 Saved permanent end-of-epoch IXI checkpoint to: {epoch_path}\n")

print("⚡ Training run successfully completed!")