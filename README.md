
# AdaDiff-DiT: Accelerated MRI Reconstruction via Fine-Grained Patching and Diffusion Transformer Prior
Idan Nissany & Hadar Rabin

This repository contains the official implementation for the test-time adaptation (TTA) MRI reconstruction framework using Diffusion Transformers, evaluating both within-domain and cross-domain performance on the FastMRI and IXI datasets.

**Note:** Database files and model checkpoints are omitted from this repository due to size constraints. Please follow the instructions below to configure the data paths before running the code.

---

## 1. Database Information
This project utilizes two primary open-source medical imaging datasets:

*   **fastMRI Dataset (Multi-Coil)**
    *   **Description:** Used for training and within-domain evaluation. Contains multi-coil brain MRI acquisitions. Contains ($T_1$, $T_2$, $\text{FLAIR}$) contrasts.
    *   **Source:** Provided by NYU Langone Health and Meta AI. 
    *   **Download:** [https://fastmri.med.nyu.edu/]
*   **IXI Dataset (Single-Coil Magnitude)**
    *   **Description:** Used for training and within-domain evaluation and also for zero-shot cross-domain generalization and input adapter evaluation. Contains ($T_1$, $T_2$, $\text{PD}$) contrasts.
    *   **Source:** Information eXtraction from Images (IXI) database, collected by multiple London hospitals.
    *   **Download:** [https://brain-development.org/ixi-dataset/]

## 2. File Types
The data loaders in this repository are designed to read the following file formats:
*   **`.h5` / `.hdf5` (Primary):** The standard format for fastMRI and IXI data. The scripts automatically look for dataset keys such as `kspace`, `reconstruction_rss`, or `data`.
*   **`.npy` / `.npz` / `.pt`:** Supported for fallback evaluation of pre-processed tensor slices.

## 3. Data Path Configuration
Before running any training, evaluation or visualization scripts, you must update the hardcoded paths to point to your local copies of the datasets and `.pt` model checkpoints. 

Update the following variables in the respective scripts:

**In `IXI_adadiff/train_ixi_sinusoidal_fast.py` (train the ixi GAN):**
*   `DRIVE_IXI_TRAIN`: Path to the ixi train set directory.
*   `CHECKPOINT_DIR`: Path to the directory where your checkpoints files will be saved (e.g., `adadiff_ixi_sinusoidal_epoch_1.pt`)
*   `CHECKPOINT_FILE` : Make sure it matches the last checkpoint created (important when resuming training that stopped)

**In `IXI_adadiff/run_ixi_grid_search_sinusoidal.py` (Running and evaluating all hyperparameters configurations):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_ixi_sinusoidal_latest.pt`)
*   `val_base_dir`: Path to the fastMRI or IXI val directory containing the `.h5` files.
*    Make sure the script is in the same directory with `best_hyperparameters_ixi.json`.

**In `IXI_adadiff/ixi_test_within_domain.py` (Evaluating within-domain IXI performance and creates images to display):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_ixi_sinusoidal_latest.pt`)
*   `TEST_BASE_DIR`: Path to the fastMRI or IXI test directory containing the `.h5` files.
*   Make sure the script is in the same directory with `best_hyperparameters_ixi.json`.

**In `IXI_adadiff/ixi_literature_baseline.py` (Evaluating within-domain IXI baseline (j=1000) performance and creates images to display):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_ixi_sinusoidal_latest.pt`)
*   `TEST_BASE_DIR`: Path to the fastMRI or IXI test directory containing the `.h5` files.

**In `fastMRI_adadiff/train_fast_mri_sinusoidal_fast.py` (train the fastmri GAN):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_fastmri_sinusoidal_latest.pt`)
*   `CHECKPOINT_DIR` : Path to the trained generator checkpoints directory.
*   `DRIVE_FASTMRI_TRAIN`: Path to the fastMRI or IXI test directory containing the `.h5` files.

**In `fastMRI_adadiff/run_fastmri_grid_search_sinusoidal.py` (Running and evaluating all hyperparameters configurations):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_fastmri_sinusoidal_latest.pt`)
*   `val_dir`: Path to the fastMRI or IXI val directory containing the `.h5` files.

**In `fastMRI_adadiff/fastmri_test_within_domain.py` (Evaluating within-domain fastmri performance and creates images to display):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_fastmri_sinusoidal_latest.pt`)
*   `test_base_dir`: Path to the fastMRI or IXI test directory containing the `.h5` files.

**In `fastMRI_adadiff/run_cross_domain_fastmri_to_ixi_vx.py` (Evaluating cross-domain fastmri pipeline performance on IXI data and creates images to display):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_fastmri_sinusoidal_latest.pt`)
*   `TEST_BASE_DIR`: Path to the *IXI* test directory containing the `.h5` files.
  
**In `fastMRI_adadiff/generate_paper_figures_fastmri.py` (Main Evaluation & Figures):**
*   `CHECKPOINT_FILE`: Path to the trained generator (e.g., `adadiff_fastmri_sinusoidal_latest.pt`)
*   `test_base_dir`: Path to the fastMRI or IXI test split directory containing the `.h5` files.

**In `tools/monitor_training.py`:**
*   Update the paths at the very bottom of the file under the `if __name__ == "__main__":` block:
    *   `checkpoint_path`: Pointing to your latest `.pt` model.
    *   `data_dir`: Pointing to your validation dataset folder.
 

## 4. Execution Instructions and Relevant Details
### Prerequisites
Ensure your environment has the following dependencies installed:

```bash
pip install torch numpy h5py matplotlib scikit-image lpips
```

### Running the code
You can run a python script in 1 of the 2 following ways:

* Make sure you are in the correct folder where the script is located at and run:
```bash
python name_of_the_wanted_script.py
```

For monitoring:
```bash
tail -f name_of_log.log
```

* By slurm - look at the script directory and search/create a sbatch file that contains the python script you want to run.

Make sure you are in the script's directory and run:
```bash
sbatch nameofsbatchfile.sbatch
```
for monitioring:
```bash
tail -f slurm_logs/slurm-<YOUR_SLURM_ID>.out
