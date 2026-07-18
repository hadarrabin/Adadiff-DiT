import os
import shutil
import numpy as np
from tqdm import tqdm

# Paths
test_dir = "./test"
t1_dir = os.path.join(test_dir, "T1")
t2_dir = os.path.join(test_dir, "T2")
pd_dir = os.path.join(test_dir, "PD")

# Create target directories
for d in [t1_dir, t2_dir, pd_dir]:
    os.makedirs(d, exist_ok=True)

# Find all slice files
slices = [f for f in os.listdir(test_dir) if f.startswith("ixi_slice_") and f.endswith(".npy")]
slices.sort()

if not slices:
    print(f"❌ No unsorted slices starting with 'ixi_slice_' found in '{test_dir}'!")
    print("They may already be inside the T1, T2, or PD subfolders.")
    exit()

features = []
print(f"🧠 Analyzing physical slice intensities for {len(slices)} test files...")
for slice_file in tqdm(slices, desc="Processing"):
    src_path = os.path.join(test_dir, slice_file)
    try:
        img = np.load(src_path)
        mean_val = np.mean(img)
        # Ratio of hyper-intense pixels (for glowing T2 ventricles)
        bright_ratio = np.sum(img > 0.8) / img.size
        
        features.append({
            "name": slice_file,
            "mean": mean_val,
            "bright_ratio": bright_ratio
        })
    except Exception as e:
        print(f"Error reading {slice_file}: {e}")
        continue

total_samples = len(features)
print(f"\nSorting {total_samples} total slices into 3 equal contrast categories...")

# 1. Identify T2: The slices with the highest ratio of hyper-bright pixels (approx. 1/3 of the dataset)
features.sort(key=lambda x: x["bright_ratio"], reverse=True)
split_size = total_samples // 3
t2_selections = features[:split_size]

# Remove selected T2s from the remaining candidate pool
remaining = [f for f in features if f not in t2_selections]

# 2. Sort the remaining by mean tissue intensity:
# - Darkest half of remaining becomes T1
# - Brightest half of remaining becomes PD
remaining.sort(key=lambda x: x["mean"])
half_size = len(remaining) // 2
t1_selections = remaining[:half_size]
pd_selections = remaining[half_size:]

# Move the files to their respective contrast folders
moved = {"T1": 0, "T2": 0, "PD": 0}

for item in t1_selections:
    shutil.move(os.path.join(test_dir, item["name"]), os.path.join(t1_dir, item["name"]))
    moved["T1"] += 1

for item in t2_selections:
    shutil.move(os.path.join(test_dir, item["name"]), os.path.join(t2_dir, item["name"]))
    moved["T2"] += 1

for item in pd_selections:
    shutil.move(os.path.join(test_dir, item["name"]), os.path.join(pd_dir, item["name"]))
    moved["PD"] += 1

print("\n🎉 Test Dataset Sorting Complete!")
print(f" -> Successfully moved {moved['T1']} files to ./test/T1")
print(f" -> Successfully moved {moved['T2']} files to ./test/T2")
print(f" -> Successfully moved {moved['PD']} files to ./test/PD")
