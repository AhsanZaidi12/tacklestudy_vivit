# ViViT-Based Risky Tackle Detection in American Football Practice Videos
 
Code for the ICPR 2026 paper:
 
> **ViTs for Action Classification in Videos: An Approach to Risky Tackle Detection in American Football Practice Videos**
> Syed Ahsan Masud Zaidi, William Hsu, Scott Dietrich
> Kansas State University & Albright College
> [arXiv:2604.01318](https://arxiv.org/abs/2604.01318)
 
Fine-tunes `google/vivit-b-16x2-kinetics400` with focal loss and a Taguchi L18
augmentation design to classify tackle clips as **risky** or **safe**.
Headline result: **risky recall = 0.67, risky F1 = 0.59** (5-fold CV, Run 15).
 
---
 
## Quick Links
 
| Resource | Link |
|---|---|
| Paper (arXiv) | https://arxiv.org/abs/2604.01318 |
| Sample dataset (public) | https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet-sample |
| Usage demo notebook | https://www.kaggle.com/code/ahsanzaidi786/tacklenet-usage-sample |
 
---
 
## Repository Structure
 
```
tacklestudy_vivit/
├── vivit_train_taguchi.py      # MAIN script — produces all paper results
├── vivit_train.py              # Earlier single-run version (not used for paper)
├── Taguchi_datasets.py         # Builds augmented fold datasets (Taguchi L18)
├── consolidate_taguchi.py      # Publication stats: FDR, effect sizes, CIs
├── consolidate_metrics.py      # Aggregates per-fold metrics_summary.csv files
├── run_vivit_train_taguchi.sh  # SLURM: multi-job Taguchi ablation submission
├── run_Access.sh               # SLURM: single-run submission
├── fix_vidExten.py             # Video extension normalization utility
├── framechck.py                # Frame count verification utility
├── requirements.txt            # Pinned pip dependencies
├── environment.yml             # Conda environment (vivit_pyt)
└── I3D/                        # Prior C3D/I3D baseline assets and annotations
```
 
---
 
## Hardware Used
 
| Component | Spec |
|---|---|
| GPU | 1× NVIDIA H100 80 GB |
| RAM | 128 GB |
| CPU | AMD EPYC 7742 64-core |
| HPC system | PSC Bridges-2 (ACCESS allocation) |
| OS | Linux (Ubuntu 20.04) |
 
**Minimum to reproduce one fold:** any CUDA GPU with ≥ 24 GB VRAM.
Reduce `BATCH_SIZE` in `vivit_train_taguchi.py` if GPU memory is limited.
 
---
 
## Installation
 
**Prerequisites:** Python 3.10, CUDA 12.1, Conda (recommended)
 
### Option A — Conda (recommended)
 
```bash
git clone https://github.com/AhsanZaidi12/tacklestudy_vivit.git
cd tacklestudy_vivit
 
conda env create -f environment.yml
conda activate vivit_pyt
```
 
### Option B — pip
 
```bash
git clone https://github.com/AhsanZaidi12/tacklestudy_vivit.git
cd tacklestudy_vivit
 
pip install -r requirements.txt
```
 
> **Note:** Model weights (`google/vivit-b-16x2-kinetics400`) are downloaded
> automatically from Hugging Face on first run. Internet access is required.
 
---
 
## Data
 
### Public sample dataset
A representative public sample of TackleNet videos is available on Kaggle:
 
**https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet-sample**
 
This sample is sufficient to verify that the pipeline runs end-to-end.
 
### Full dataset (IRB-constrained)
The complete 733-clip dataset was collected under IRB/consent constraints
and **cannot be freely redistributed**. To request access under a
data-use agreement, contact: **ahsanzaidi@ksu.edu**
 
### Expected directory layout
 
After obtaining data, organize as follows:
 
```
taguchi_runs/
└── run_15/
    ├── train/
    │   └── videos/            # .mp4 clips
    ├── train_labels.csv        # columns: fname,label
    ├── val/
    │   └── videos/
    └── val_labels.csv
```
 
Label CSV format (header required):
 
```
fname,label
tackle_001.mp4,0
tackle_002.mp4,1
```
 
`0` = safe, `1` = risky.
 
---
 
## Reproducing Paper Results
 
### Step 1 — Build Taguchi fold datasets
 
```bash
python Taguchi_datasets.py \
    --video_root /path/to/raw/videos \
    --label_csv  /path/to/labels.csv \
    --output_root ./taguchi_runs \
    --seed 42
```
 
Creates 20 run directories (18 Taguchi configurations + Run0 + Run_orig),
each with augmented training folds and unaugmented validation folds.
 
### Step 2 — Train a single fold
 
```bash
python vivit_train_taguchi.py \
    --runs run_15 \
    --fold 0
```
 
> **Before running:** update `BASE_RUNS_DIR` and `RESULTS_BASE_DIR` at
> lines 87–88 of `vivit_train_taguchi.py` to your local paths.
 
Full configuration used for paper results:
 
| Parameter | Value |
|---|---|
| Backbone | `google/vivit-b-16x2-kinetics400` |
| Input | 32 frames × 224 × 224 px |
| Batch size | 2 |
| Gradient accumulation | 8 steps (effective batch = 16) |
| Epochs | 50 |
| Focal loss α / γ | 0.6 / 1.6 |
| Threshold strategy | `macro_f1` |
| Seed | 42 |
| Precision | fp16 |
 
### Step 3 — Run all 100 experiments on HPC (SLURM)
 
```bash
bash run_vivit_train_taguchi.sh h100-80:1 GPU-shared asc180003p
#                               ^gpu type  ^partition  ^account
```
 
Edit the account/partition to match your HPC allocation.
Each job runs 3 configurations × 5 folds. Total: 20 configs × 5 folds = 100 runs.
 
### Step 4 — Aggregate results
 
```bash
# Full publication statistics (FDR correction, Cohen's d, 95% CI)
python consolidate_taguchi.py --results_dir ./taguchi_runs_GRADCAM_RESULTS
 
# Simple per-fold metric aggregation
python consolidate_metrics.py
```
 
### Expected headline numbers (Run 15, mean over 5 folds)
 
| Metric | Paper value |
|---|---|
| Risky recall | 0.67 |
| Risky F1 | 0.59 |
| Overall accuracy | 0.67 |
 
---
 
## Reproducibility Notes
 
- Global seed `SEED = 42` is set for `random`, `numpy`, `torch`, `cudnn`, and `PYTHONHASHSEED` at startup in `vivit_train_taguchi.py`.
- `DETERMINISTIC_SAMPLER = False` (default) gives better training but allows minor run-to-run variance. Set to `True` for fully deterministic
  behaviour at a small cost to performance.
- Expected run-to-run variance: ±2–3 pp on risky recall. Results in the paper are reported as mean ± SD over 5 folds.
- Pin `transformers==4.51.3` exactly — newer versions may change the `VivitForVideoClassification` API.
---
 
## Citation
 
```bibtex
@article{zaidi2026vivit,
  title   = {ViTs for Action Classification in Videos: An Approach to Risky Tackle Detection in American Football Practice Videos},
  author  = {Zaidi, Syed Ahsan Masud and Hsu, William and Dietrich, Scott},
  journal = {arXiv preprint arXiv:2604.01318},
  year    = {2026}
}
```
 
---
 
## License
 
MIT — see [LICENSE](LICENSE)
