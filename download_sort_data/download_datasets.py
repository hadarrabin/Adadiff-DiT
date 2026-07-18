import os
import sys
import time
import json
import math
import shutil
import tarfile
import requests
import h5py
import numpy as np
import subprocess
from tqdm import tqdm

# =============================================================================
# STAGE 1: ENVIRONMENT SETUP & DEPENDENCY INSTALLATION
# =============================================================================
def setup_environment():
    print("====================================================")
    print("📦 STAGE 1: Cloning Repositories & Installing Packages")
    print("====================================================")
    
    # Clone AdaDiff and DiT repositories into the workspace
    if not os.path.exists('AdaDiff'):
        print("Cloning AdaDiff repository...")
        subprocess.run(["git", "clone", "https://github.com/icon-lab/AdaDiff.git"], check=True)
    
    if not os.path.exists('DiT'):
        print("Cloning DiT repository...")
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/DiT.git"], check=True)

    # Install required engineering dependencies
    print("Installing packages: timm, diffusers, h5py, datasets...")
    subprocess.run([sys.executable, "-m", "pip", "install", "timm", "diffusers", "h5py", "datasets", "-q"], check=True)
    
    # Append workspaces directly to system path for subsequent runtime visibility
    sys.path.append(os.path.abspath('AdaDiff'))
    sys.path.append(os.path.abspath('DiT'))
    
    print("Current workspace directories:", os.listdir('.'))

# =============================================================================
# STAGE 2: FASTMRI STREAMING DOWNLOAD & FILTERING PIPELINE
# =============================================================================
def download_fastmri():
    print("\n====================================================")
    print("🧠 STAGE 2: Processing fastMRI Pipeline (Batch 0 & 1)")
    print("====================================================")

    # Define absolute target directory structures 
    DRIVE_TRAIN_DIR = './fastMRI_adadiff/train'
    DRIVE_VAL_DIR = './fastMRI_adadiff/val'
    DRIVE_TEST_DIR = './fastMRI_adadiff/test'
    LOCAL_TEMP = './scratch_extract'

    for folder in [DRIVE_TRAIN_DIR, DRIVE_VAL_DIR, DRIVE_TEST_DIR, LOCAL_TEMP]:
        os.makedirs(folder, exist_ok=True)

    # Active download endpoints from your fastMRI portal session
    urls = {
        "TRAIN": {
            "url": "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ht5Ter%2FPB2BdpT06faUgF89nCEI%3D&Expires=1790494021",
            "target_dir": DRIVE_TRAIN_DIR,
            "required": 240
        },
        "VAL": {
            "url": "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WaTaRcVGIOn3XqVq2FpAmTEBPVs%3D&Expires=1790494021",
            "target_dir": DRIVE_VAL_DIR,
            "required": 60
        },
        "TEST": {
            "url": "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ryfh%2FUphVopHISHmE200G2rJIVM%3D&Expires=1790494021",
            "target_dir": DRIVE_TEST_DIR,
            "required": 120
        }
    }

    # Process Batch 0 segments
    for stage, config in urls.items():
        print(f"\n--- Processing Stage: {stage} ---")
        existing_files = [f for f in os.listdir(config['target_dir']) if f.endswith('.h5')]
        existing_count = len(existing_files)

        if existing_count >= config['required']:
            print(f"Directory already contains {existing_count}/{config['required']} subjects. Skipping stage.")
            continue

        needed_count = config['required'] - existing_count
        curated_count = 0
        file_scan_counter = 0

        print(f"Streaming remaining {needed_count} subjects directly using session tokens...")
        response = requests.get(config['url'], stream=True)
        pbar = tqdm(total=needed_count, desc=f"Extracting {stage}")

        with tarfile.open(fileobj=response.raw, mode='r|xz') as tar:
            for member in tar:
                if curated_count >= needed_count:
                    break

                if member.isfile() and member.name.endswith('.h5'):
                    filename = os.path.basename(member.name)
                    file_scan_counter += 1

                    if file_scan_counter % 5 == 0:
                        print(f" -> Active stream progress: Processing file #{file_scan_counter}: {filename}")

                    if filename in existing_files:
                        continue

                    tar.extract(member, path=LOCAL_TEMP)
                    local_fpath = os.path.join(LOCAL_TEMP, member.name)

                    try:
                        with h5py.File(local_fpath, 'r') as hf:
                            kspace_shape = hf['kspace'].shape

                        # Physical selection criteria filter check (>= 10 slices, >= 20 coils)
                        if kspace_shape[0] >= 10 and kspace_shape[1] >= 20:
                            shutil.move(local_fpath, os.path.join(config['target_dir'], filename))
                            curated_count += 1
                            pbar.update(1)
                        else:
                            os.remove(local_fpath)
                    except Exception:
                        if os.path.exists(local_fpath):
                            os.remove(local_fpath)
                        continue
        pbar.close()

    # --- TOPPING OFF TRAINING FOLDER VIA BATCH 1 STREAM ---
    print("\n--- Topping off Training Data from Batch 1 ---")
    batch_1_url = "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=RRHgR6uMp57gfbNqpZmZjuyQCHo%3D&Expires=1790174218"
    
    existing_files = [f for f in os.listdir(DRIVE_TRAIN_DIR) if f.endswith('.h5')]
    curated_count = len(existing_files)
    needed_count = 240 - curated_count

    if needed_count <= 0:
        print(f"Your training folder already has {curated_count} subjects. No top-off needed!")
    else:
        print(f"Current count: {curated_count}. Fetching exactly {needed_count} more subjects from Batch 1...")
        response = requests.get(batch_1_url, stream=True)
        pbar = tqdm(total=needed_count, desc="Fetching final training subjects")

        with tarfile.open(fileobj=response.raw, mode='r|xz') as tar:
            for member in tar:
                if curated_count >= 240:
                    break 

                if member.isfile() and member.name.endswith('.h5'):
                    filename = os.path.basename(member.name)
                    if filename in existing_files:
                        continue

                    tar.extract(member, path=LOCAL_TEMP)
                    local_fpath = os.path.join(LOCAL_TEMP, member.name)

                    try:
                        with h5py.File(local_fpath, 'r') as hf:
                            kspace_shape = hf['kspace'].shape

                        if kspace_shape[0] >= 10 and kspace_shape[1] >= 20:
                            shutil.move(local_fpath, os.path.join(DRIVE_TRAIN_DIR, filename))
                            curated_count += 1
                            pbar.update(1)
                        else:
                            os.remove(local_fpath)
                    except Exception:
                        if os.path.exists(local_fpath):
                            os.remove(local_fpath)
                        continue
        pbar.close()

    final_total = len([f for f in os.listdir(DRIVE_TRAIN_DIR) if f.endswith('.h5')])
    print(f"\nCompleted! Training folder total is now exactly: {final_total} subjects.")
    
    # Wipe down local temp files safely
    shutil.rmtree(LOCAL_TEMP)

# =============================================================================
# STAGE 3: IXI HUGGING FACE RESUMABLE DATASET STREAMER
# =============================================================================
def download_ixi():
    print("\n====================================================")
    print("📁 STAGE 3: Streaming IXI 2D Brain Slice Dataset")
    print("====================================================")

    from datasets import load_dataset
    
    DRIVE_IXI_TRAIN = './IXI_adadiff/train'
    DRIVE_IXI_VAL = './IXI_adadiff/val'
    DRIVE_IXI_TEST = './IXI_adadiff/test'

    for folder in [DRIVE_IXI_TRAIN, DRIVE_IXI_VAL, DRIVE_IXI_TEST]:
        os.makedirs(folder, exist_ok=True)

    print("Connecting to IXI 2D Brain Slice Dataset stream from repository...")
    base_dataset = load_dataset("iamkzntsv/IXI2D", split="train", streaming=True)

    train_slice_limit = 2268
    val_slice_limit = 1620
    test_slice_limit = 3240

    def save_slices_resumable(target_folder, max_slices, global_start_offset):
        existing_files = [f for f in os.listdir(target_folder) if f.endswith('.npy')]
        existing_count = len(existing_files)

        if existing_count >= max_slices:
            print(f"Partition {target_folder.split('/')[-1]} is already complete ({existing_count}/{max_slices}).")
            return global_start_offset + max_slices

        total_skip_offset = global_start_offset + existing_count
        print(f"Folder has {existing_count}/{max_slices} slices. Advancing internet stream past {total_skip_offset} samples...")

        resumed_dataset = base_dataset.skip(total_skip_offset)
        dataset_iter = iter(resumed_dataset)

        saved_count = existing_count
        pbar = tqdm(total=max_slices, initial=existing_count, desc=f"Populating {target_folder.split('/')[-1]}")

        while saved_count < max_slices:
            try:
                sample = next(dataset_iter)
                img_gray = sample['image'].convert('L')
                matrix_2d = np.array(img_gray, dtype=np.float32)

                # Absolute dynamic max normalization layer
                if np.max(matrix_2d) > 0:
                    matrix_2d = matrix_2d / np.max(matrix_2d)

                file_name = f"ixi_slice_{saved_count:05d}.npy"
                np.save(os.path.join(target_folder, file_name), matrix_2d)

                saved_count += 1
                pbar.update(1)
            except StopIteration:
                print("\nReached the end of the streaming dataset partition.")
                break

        pbar.close()
        return global_start_offset + max_slices

    current_global_offset = 0
    current_global_offset = save_slices_resumable(DRIVE_IXI_TRAIN, max_slices=train_slice_limit, global_start_offset=current_global_offset)
    current_global_offset = save_slices_resumable(DRIVE_IXI_VAL, max_slices=val_slice_limit, global_start_offset=current_global_offset)
    current_global_offset = save_slices_resumable(DRIVE_IXI_TEST, max_slices=test_slice_limit, global_start_offset=current_global_offset)

    print("\nAll IXI dataset splits successfully synced and stored local space!")

# =============================================================================
# MAIN ORCHESTRATION PIPELINE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    t_start = time.time()
    
    setup_environment()
    download_fastmri()
    download_ixi()
    
    print(f"\n🏁 Complete dataset synchronization pipeline finished in {time.time() - t_start:.2f} seconds.")
