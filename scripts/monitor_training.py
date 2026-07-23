import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # Prevent display errors on cluster/SSH environments
import matplotlib.pyplot as plt

# =============================================================================
# 1. IMPORT ARCHITECTURE
# =============================================================================
# ⚠️ UPDATE 'your_model_file' to the actual name of your .py file containing these classes
from your_model_file import DiffusionTransformer, AdaDiffScheduler

# =============================================================================
# 2. DATASET LOADER
# =============================================================================
class UnifiedMRIDataset(Dataset):
    def __init__(self, data_dir, virtual_coils=5, input_size=256):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
        self.virtual_coils = virtual_coils
        self.input_size = input_size

    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.data_dir, self.file_list[idx]), 'r') as hf:
            kspace = torch.from_numpy(np.array(hf['kspace'], dtype=np.complex64)[kspace_shape := hf['kspace'].shape[0] // 2])
        
        # Compression logic
        flattened = kspace.view(kspace.shape[0], -1)
        U, _, _ = torch.linalg.svd(flattened, full_matrices=False)
        compressed = torch.matmul(U[:, :self.virtual_coils].H, flattened).view(self.virtual_coils, kspace.shape[1], kspace.shape[2])
        
        # Resize & Spatial Transform
        res = F.interpolate(compressed.real.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear')
        img = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(res, res), dim=(-2, -1)), norm="ortho")
        return torch.cat([img.real, img.imag], dim=1).squeeze(0)

# =============================================================================
# 3. VISUALIZATION FUNCTION
# =============================================================================
def visualize_latest(checkpoint_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = DiffusionTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator_state_dict'])
    model.eval()
    
    ds = UnifiedMRIDataset(data_dir)
    sched = AdaDiffScheduler()
    
    clean = ds[0].unsqueeze(0).to(device)
    noisy, _ = sched.add_noise(clean, 2)
    t = torch.tensor([sched.stages[2]], device=device)
    
    with torch.no_grad():
        recon = model(noisy, t)
        
    def to_rss(t_tensor): 
        c = torch.complex(t_tensor[:, :5], t_tensor[:, 5:])
        return torch.sqrt(torch.sum(torch.abs(c)**2, dim=1))[0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, title in zip(ax, [to_rss(clean), to_rss(noisy), to_rss(recon)], ["Truth", "Noisy", "Recon"]):
        a.imshow(img, cmap='gray')
        a.set_title(title)
        a.axis('off')
        
    plt.tight_layout()
    plt.savefig("latest_snapshot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved 'latest_snapshot.png'")

if __name__ == "__main__":
    # !!! UPDATE THESE PATHS !!!
    visualize_latest(
        "/ugproj/sipl-prj10848/Idan/fastmri_checkpoints_sinusoidal/adadiff_fastmri_sinusoidal_latest.pt",
        "/ugproj/sipl-prj10848/Idan/fastMRI_adadiff/val"
    )