import os
import shutil
import numpy as np

# Paths
val_dir = "./val"
t1_dir = os.path.join(val_dir, "T1")
t2_dir = os.path.join(val_dir, "T2")
pd_dir = os.path.join(val_dir, "PD")

for d in [t1_dir, t2_dir, pd_dir]:
    os.makedirs(d, exist_ok=True)

# Find all slices
slices = [f for f in os.listdir(val_dir) if f.startswith("ixi_slice_") and f.endswith(".npy")]
slices.sort()

# Collect features for every slice
features = []
print("🧠 Analyzing physical slice intensities...")
for slice_file in slices:
    src_path = os.path.join(val_dir, slice_file)
    try:
        img = np.load(src_path)
        
        # Calculate features
        mean_val = np.mean(img)
        # T2 has highly bright CSF pixels (intensity > 0.8)
        bright_ratio = np.sum(img > 0.8) / img.size
        
        features.append({
            "name": slice_file,
            "mean": mean_val,
            "bright_ratio": bright_ratio
        })
    except Exception:
        continue

# 1. Identify T2: The slices with the highest ratio of hyper-bright pixels (glowing ventricles)
features.sort(key=lambda x: x["bright_ratio"], reverse=True)
t2_selections = features[:10]

# Remove chosen T2s from the candidate pool
remaining = [f for f in features if f not in t2_selections]

# 2. Of the remaining slices, sort by mean intensity:
# - T1 has the darkest brain tissue (lowest mean)
# - PD has the brightest overall brain tissue (highest mean)
remaining.sort(key=lambda x: x["mean"])

t1_selections = remaining[:10]         # Darkest 10
pd_selections = remaining[-10:]        # Brightest 10

# Move the selected files to their folders
moved = {"T1": 0, "T2": 0, "PD": 0}

for item in t1_selections:
    shutil.move(os.path.join(val_dir, item["name"]), os.path.join(t1_dir, item["name"]))
    moved["T1"] += 1

for item in t2_selections:
    shutil.move(os.path.join(val_dir, item["name"]), os.path.join(t2_dir, item["name"]))
    moved["T2"] += 1

for item in pd_selections:
    shutil.move(os.path.join(val_dir, item["name"]), os.path.join(pd_dir, item["name"]))
    moved["PD"] += 1

print("\n🎉 Auto-Sorting Complete!")
print(f" -> Successfully moved {moved['T1']} files to ./val/T1")
print(f" -> Successfully moved {moved['T2']} files to ./val/T2")
print(f" -> Successfully moved {moved['PD']} files to ./val/PD")
