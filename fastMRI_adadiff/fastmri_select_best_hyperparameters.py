import json
import os

def normalize(val, min_val, max_val, higher_is_better=True):
    """Min-Max normalization. Returns 1.0 for the best value, 0.0 for the worst."""
    if max_val == min_val:
        return 1.0
    if higher_is_better:
        return (val - min_val) / (max_val - min_val)
    else:
        return (max_val - val) / (max_val - min_val)

def process_grid_results(output_path="best_hyperparameters_fastmri.json"):
    # Automatically locate the correct grid summary file
    possible_files = ["fastmri_adadiff_grid_summary.json", "dit_adadiff_grid_summary.json"]
    json_path = None
    for fname in possible_files:
        if os.path.exists(fname):
            json_path = fname
            break

    if not json_path:
        print("❌ Error: Could not find any grid summary JSON file in the current directory.")
        print("Please make sure you are running this script inside the folder containing your grid search results.")
        return

    print(f"📂 Loading grid search data from: '{json_path}'")
    with open(json_path, "r") as f:
        grid_data = json.load(f)

    print(f"📊 Loaded {len(grid_data)} total configurations. Separating by acceleration rate...")

    # Separate data by acceleration rate
    r_groups = {4: {}, 8: {}}
    for key, metrics in grid_data.items():
        # Case-insensitive check for acceleration rate 'R'
        r_val = metrics.get("R", metrics.get("r"))
        if r_val in r_groups:
            r_groups[r_val][key] = metrics

    best_configs = {}

    for r_val, configs in r_groups.items():
        if not configs:
            print(f"⚠️ Warning: No configurations found for R={r_val}")
            continue
            
        print(f"\n========================================")
        print(f" Evaluating R={r_val} Configurations ({len(configs)} total)")
        print(f"========================================")

        psnr_vals = [c["PSNR"] for c in configs.values()]
        ssim_vals = [c["SSIM"] for c in configs.values()]
        lpips_vals = [c["LPIPS"] for c in configs.values()]
        time_vals = [c["Time"] for c in configs.values()]

        bounds = {
            "PSNR": {"min": min(psnr_vals), "max": max(psnr_vals)},
            "SSIM": {"min": min(ssim_vals), "max": max(ssim_vals)},
            "LPIPS": {"min": min(lpips_vals), "max": max(lpips_vals)},
            "Time": {"min": min(time_vals), "max": max(time_vals)}
        }

        best_score = -1.0
        best_key = None

        for key, metrics in configs.items():
            n_psnr = normalize(metrics["PSNR"], bounds["PSNR"]["min"], bounds["PSNR"]["max"], higher_is_better=True)
            n_ssim = normalize(metrics["SSIM"], bounds["SSIM"]["min"], bounds["SSIM"]["max"], higher_is_better=True)
            n_lpips = normalize(metrics["LPIPS"], bounds["LPIPS"]["min"], bounds["LPIPS"]["max"], higher_is_better=False)
            n_time = normalize(metrics["Time"], bounds["Time"]["min"], bounds["Time"]["max"], higher_is_better=False)

            # Apply 40/40/10/10 weights
            final_score = (0.40 * n_psnr) + (0.40 * n_ssim) + (0.10 * n_lpips) + (0.10 * n_time)
            configs[key]["Final_Score"] = final_score

            if final_score > best_score:
                best_score = final_score
                best_key = key

        # Case-insensitive key extraction with fallbacks to prevent KeyErrors
        winner = configs[best_key]
        best_configs[f"R_{r_val}"] = {
            "lr": winner.get("lr", winner.get("LR", 0.0005)),
            "J": winner.get("J", winner.get("j", 100)),
            "eta": winner.get("eta", winner.get("ETA", 1.0)),
            "scope": winner.get("scope", winner.get("Scope", "adaln_only"))
        }

        print(f"🏆 Winner: {best_key}")
        print(f"   Score: {best_score:.4f}")
        print(f"   PSNR:  {winner['PSNR']:.2f} dB")
        print(f"   SSIM:  {winner['SSIM']*100:.2f}%")
        print(f"   LPIPS: {winner['LPIPS']:.4f}")
        print(f"   Time:  {winner['Time']:.2f}s")

    # Write the final nested structure cleanly
    with open(output_path, "w") as f:
        json.dump(best_configs, f, indent=4)
    print(f"\n✅ Best hyperparameters successfully exported to '{output_path}'")

if __name__ == "__main__":
    process_grid_results()