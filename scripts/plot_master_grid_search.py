import os
import matplotlib

matplotlib.use("Agg")  # Safe for cluster/SSH execution
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. EXPERIMENTAL CONVERGENCE DATA (EXTRACTED FROM GRID SEARCH LOGS)
# =============================================================================

# --- fastMRI Convergence Trajectories (p=4, N=4096 tokens) ---
# R=4x Optimal: LR=1e-4, ETA=1.0, Scope=all
fmri_r4_time = [0.35, 2.25, 5.02, 9.68, 18.89]
fmri_r4_psnr = [24.26, 27.21, 27.27, 27.37, 27.39]
fmri_r4_ssim = [58.41, 75.99, 76.03, 76.10, 75.54]

# R=8x Optimal: LR=1e-4, ETA=1.0, Scope=adaln_only (Notice the peak at J=25!)
fmri_r8_time = [0.35, 2.10, 4.65, 8.94, 17.44]
fmri_r8_psnr = [16.85, 24.40, 24.87, 24.72, 24.41]
fmri_r8_ssim = [25.30, 61.77, 64.23, 63.63, 61.78]

# --- IXI Convergence Trajectories (p=2, N=10000 tokens) ---
# R=4x Optimal: LR=1e-4, ETA=1.0, Scope=adaln_only
ixi_r4_time = [1.62, 10.07, 22.35, 42.82, 83.80]
ixi_r4_psnr = [26.23, 27.88, 28.78, 30.02, 30.16]
ixi_r4_ssim = [68.94, 75.78, 80.43, 83.16, 82.71]

# R=8x Optimal: LR=1e-4, ETA=0.8, Scope=adaln_only
ixi_r8_time = [1.62, 10.07, 22.35, 42.82, 83.77]
ixi_r8_psnr = [24.44, 25.23, 25.59, 25.95, 26.43]
ixi_r8_ssim = [61.46, 67.29, 71.17, 72.74, 73.50]


# =============================================================================
# 2. HELPER FUNCTION TO PLOT INDIVIDUAL ROWS
# =============================================================================
def plot_dataset_row(
    ax_psnr,
    ax_ssim,
    t_r4,
    p_r4,
    s_r4,
    t_r8,
    p_r8,
    s_r8,
    dataset_name,
    opt_r4_idx,
    opt_r8_idx,
    r4_label,
    r8_label,
):
  # --- PSNR Plot ---
  ax_psnr.plot(
      t_r4,
      p_r4,
      marker="o",
      linewidth=2.5,
      markersize=8,
      color="#1f77b4",
      label=f"R=4x ({r4_label})",
  )
  ax_psnr.plot(
      t_r8,
      p_r8,
      marker="s",
      linewidth=2.5,
      markersize=8,
      color="#ff7f0e",
      label=f"R=8x ({r8_label})",
  )

  # Annotate Winners
  ax_psnr.annotate(
      f"Selected\n({p_r4[opt_r4_idx]:.2f} dB)",
      xy=(t_r4[opt_r4_idx], p_r4[opt_r4_idx]),
      xytext=(t_r4[opt_r4_idx] - 3, p_r4[opt_r4_idx] - 1.2),
      arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
      fontsize=10,
      fontweight="bold",
  )
  ax_psnr.annotate(
      f"Selected\n({p_r8[opt_r8_idx]:.2f} dB)",
      xy=(t_r8[opt_r8_idx], p_r8[opt_r8_idx]),
      xytext=(t_r8[opt_r8_idx] + 1, p_r8[opt_r8_idx] - 2.0),
      arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
      fontsize=10,
      fontweight="bold",
  )

  ax_psnr.set_title(
      f"{dataset_name}: PSNR vs. Latency",
      fontsize=13,
      fontweight="bold",
      pad=10,
  )
  ax_psnr.set_xlabel("Inference Runtime per Slice (seconds)", fontsize=11)
  ax_psnr.set_ylabel("PSNR (dB)", fontsize=11)
  ax_psnr.grid(True, linestyle="--", alpha=0.5)
  ax_psnr.legend(fontsize=10, loc="lower right")

  # --- SSIM Plot ---
  ax_ssim.plot(
      t_r4,
      s_r4,
      marker="o",
      linewidth=2.5,
      markersize=8,
      color="#2ca02c",
      label=f"R=4x ({r4_label})",
  )
  ax_ssim.plot(
      t_r8,
      s_r8,
      marker="s",
      linewidth=2.5,
      markersize=8,
      color="#d62728",
      label=f"R=8x ({r8_label})",
  )

  # Annotate Winners
  ax_ssim.annotate(
      f"Selected\n({s_r4[opt_r4_idx]:.2f}%)",
      xy=(t_r4[opt_r4_idx], s_r4[opt_r4_idx]),
      xytext=(t_r4[opt_r4_idx] - 3, s_r4[opt_r4_idx] - 8.0),
      arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
      fontsize=10,
      fontweight="bold",
  )
  ax_ssim.annotate(
      f"Selected\n({s_r8[opt_r8_idx]:.2f}%)",
      xy=(t_r8[opt_r8_idx], s_r8[opt_r8_idx]),
      xytext=(t_r8[opt_r8_idx] + 1, s_r8[opt_r8_idx] - 12.0),
      arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
      fontsize=10,
      fontweight="bold",
  )

  ax_ssim.set_title(
      f"{dataset_name}: SSIM vs. Latency",
      fontsize=13,
      fontweight="bold",
      pad=10,
  )
  ax_ssim.set_xlabel("Inference Runtime per Slice (seconds)", fontsize=11)
  ax_ssim.set_ylabel("SSIM (%)", fontsize=11)
  ax_ssim.grid(True, linestyle="--", alpha=0.5)
  ax_ssim.legend(fontsize=10, loc="lower right")


# =============================================================================
# 3. EXPORT 1: MASTER 2x2 FIGURE (BOTH DATASETS)
# =============================================================================
fig_master, axes_master = plt.subplots(2, 2, figsize=(15, 11))

# Top Row: fastMRI (Idx 2 is J=25)
plot_dataset_row(
    axes_master[0, 0],
    axes_master[0, 1],
    fmri_r4_time,
    fmri_r4_psnr,
    fmri_r4_ssim,
    fmri_r8_time,
    fmri_r8_psnr,
    fmri_r8_ssim,
    dataset_name="fastMRI Multi-Coil ($p=4$, $N=4096$)",
    opt_r4_idx=2,
    opt_r8_idx=2,
    r4_label="J=25, $\eta=1.0$, all",
    r8_label="J=25, $\eta=1.0$, adaln",
)

# Bottom Row: IXI (Idx 3 is J=50, Idx 4 is J=100)
plot_dataset_row(
    axes_master[1, 0],
    axes_master[1, 1],
    ixi_r4_time,
    ixi_r4_psnr,
    ixi_r4_ssim,
    ixi_r8_time,
    ixi_r8_psnr,
    ixi_r8_ssim,
    dataset_name="IXI Single-Coil ($p=2$, $N=10000$)",
    opt_r4_idx=3,
    opt_r8_idx=4,
    r4_label="J=50, $\eta=1.0$, adaln",
    r8_label="J=100, $\eta=0.8$, adaln",
)

plt.suptitle(
    "Test-Time Adaptation Pareto Convergence Frontier across Modalities",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)
plt.tight_layout()
plt.savefig("pareto_master_2x2.png", dpi=300, bbox_inches="tight")
plt.close(fig_master)
print("📸 Saved Master 2x2 Figure: 'pareto_master_2x2.png'")

# =============================================================================
# 4. EXPORT 2 & 3: STANDALONE 1x2 FIGURES FOR INDIVIDUAL SECTIONS
# =============================================================================

# Standalone fastMRI Figure
fig_fmri, axes_fmri = plt.subplots(1, 2, figsize=(14, 5))
plot_dataset_row(
    axes_fmri[0],
    axes_fmri[1],
    fmri_r4_time,
    fmri_r4_psnr,
    fmri_r4_ssim,
    fmri_r8_time,
    fmri_r8_psnr,
    fmri_r8_ssim,
    dataset_name="fastMRI Multi-Coil",
    opt_r4_idx=2,
    opt_r8_idx=2,
    r4_label="J=25, $\eta=1.0$, all",
    r8_label="J=25, $\eta=1.0$, adaln",
)
plt.suptitle(
    "fastMRI Test-Time Adaptation Convergence Frontier",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("pareto_fastmri_1x2.png", dpi=300, bbox_inches="tight")
plt.close(fig_fmri)
print("📸 Saved fastMRI Standalone Figure: 'pareto_fastmri_1x2.png'")

# Standalone IXI Figure
fig_ixi, axes_ixi = plt.subplots(1, 2, figsize=(14, 5))
plot_dataset_row(
    axes_ixi[0],
    axes_ixi[1],
    ixi_r4_time,
    ixi_r4_psnr,
    ixi_r4_ssim,
    ixi_r8_time,
    ixi_r8_psnr,
    ixi_r8_ssim,
    dataset_name="IXI Single-Coil",
    opt_r4_idx=3,
    opt_r8_idx=4,
    r4_label="J=50, $\eta=1.0$, adaln",
    r8_label="J=100, $\eta=0.8$, adaln",
)
plt.suptitle(
    "IXI Test-Time Adaptation Convergence Frontier",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig("pareto_ixi_1x2.png", dpi=300, bbox_inches="tight")
plt.close(fig_ixi)
print("📸 Saved IXI Standalone Figure: 'pareto_ixi_1x2.png'")

print("\n🎉 All Pareto visualization figures generated successfully!")