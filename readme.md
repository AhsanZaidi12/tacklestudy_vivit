# ViTs-Based Risky Tackle Detection in American Football Practice Videos

Code and reproducibility materials for the ICPR 2026 paper:

> **ViTs for Action Classification in Videos: An Approach to Risky Tackle Detection in American Football Practice Videos**  
> Syed Ahsan Masud Zaidi, William Hsu, and Scott Dietrich  
> Kansas State University and Albright College  
> [arXiv:2604.01318](https://arxiv.org/abs/2604.01318)

The project fine-tunes `google/vivit-b-16x2-kinetics400` with focal loss and a Taguchi L18 augmentation design to classify practice tackle clips as risky or safe.

Reported headline result: **risky recall = 0.67** and **risky F1 = 0.59** for Run 15, averaged over five folds.

## Quick links

| Resource | Link |
|---|---|
| Paper | https://arxiv.org/abs/2604.01318 |
| Full public TackleNet release | https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet |
| Exact 733-clip paper manifest | [paper_733_labels.csv](paper_733_labels.csv) |
| Smaller public sample | https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet-sample |
| Usage demonstration | https://www.kaggle.com/code/ahsanzaidi786/tacklenet-usage-sample |

## Repository structure

```text
tacklestudy_vivit/
├── vivit_train_taguchi.py
│   └── Main ViViT training, evaluation, Grad-CAM, and threshold analysis
├── vivit_train.py
│   └── Earlier single-run implementation; not used for headline results
├── Taguchi_datasets_legacy_190.py
│   └── Legacy preliminary 190-clip data builder; not the final paper pipeline
├── paper_733_labels.csv
│   └── Exact clip membership and binary labels used in the paper
├── consolidate_taguchi.py
│   └── Statistical consolidation, confidence intervals, and confusion matrices
├── consolidate_metrics.py
│   └── Aggregates fold-level metrics_summary.csv files
├── run_vivit_train_taguchi.sh
│   └── PSC Bridges-2 SLURM training launcher
├── run_Access.sh
│   └── Earlier ACCESS/SLURM launcher
├── fix_vidExten.py
│   └── Video-extension normalization utility
├── framechck.py
│   └── Verifies that prepared model-input clips contain 32 frames
├── requirements.txt
├── environment.yml
└── I3D/
    └── Earlier C3D/I3D baseline materials
```

## Current reproducibility status

| Artifact | Status |
|---|---|
| Model training and evaluation code | Available |
| ViViT configuration and focal loss | Available |
| Taguchi L18 configuration encoding | Available |
| Grad-CAM implementation | Available |
| Statistical consolidation code | Available |
| Exact 733-clip paper membership and labels | Available |
| Full deidentified public video release | Available on Kaggle |
| Exact five-fold assignments used in the paper | Being prepared for release |
| Final 733-clip fold-construction pipeline | Being prepared for release |
| FPOC-to-32-frame preprocessing script | Being prepared for release |
| Fold-level historical metrics and logs | Being organized for release |
| Historical model checkpoints | Not currently included |

The repository previously described `Taguchi_datasets.py` as the final paper data-preparation entry point. That file was actually a preliminary utility for an earlier approximately 190-clip dataset. It has been renamed `Taguchi_datasets_legacy_190.py` and retained only for provenance.

The final 733-clip five-fold datasets were prepared using a separate experiment-preparation pipeline. The exact fold artifacts and final preprocessing code will be added separately.

## Dataset

### Exact dataset used in the paper

The paper experiments used 733 deidentified tackle video clips:

| Class | Count |
|---|---:|
| Safe (`0`) | 474 |
| Risky (`1`) | 259 |
| Total | 733 |

The authoritative paper manifest is:

```text
paper_733_labels.csv
```

It contains two columns:

```text
fname,label
001.mp4,1
002.mp4,0
```

The experimental unit is a tackle video clip, not a distinct participant or subject. The released manifest uses deidentified clip filenames and does not provide participant identities.

### Relationship to the current Kaggle release

A later curated public release containing 737 clips is available at:

https://www.kaggle.com/datasets/ahsanzaidi786/tacklenet

The 737-clip release is not a strict superset of the exact paper dataset. It overlaps with 732 of the 733 paper clips.

The current Kaggle release contains five clips that were not in the paper manifest:

```text
061.mp4
063.mp4
064.mp4
066.mp4
633.mp4
```

Paper clip `662.mp4` is not present in the current curated Kaggle release.

Three shared clips received revised binary labels during later dataset curation:

| Clip | Paper label | Current Kaggle label |
|---|---:|---:|
| `628.mp4` | Risky (`1`) | Safe (`0`) |
| `631.mp4` | Safe (`0`) | Risky (`1`) |
| `632.mp4` | Risky (`1`) | Safe (`0`) |

For reproducing the paper experiments, use the labels in `paper_733_labels.csv`. The frozen train/validation/test split distributed with the later Kaggle dataset is a subsequent benchmark split and is not the five-fold assignment used in the paper.

### First point of contact

The public TackleNet annotations include manually identified first-point-of-contact frame indices. The paper used a fixed 32-frame window consisting of 15 frames before contact, the contact frame, and 16 frames after contact.

The final script used to convert the FPOC annotations into the paper’s prepared 32-frame inputs is being prepared for release. `framechck.py` only validates already prepared 32-frame clips; it does not perform FPOC localization or cropping.

## Experimental design

The final study contained 20 ViViT configurations:

- 18 Taguchi L18 augmentation configurations;
- one original unbalanced baseline; and
- one duplication-only/oversampled baseline.

Each configuration was evaluated across five stratified folds, yielding 100 fold-level training experiments.

The intended final run naming is:

```text
run_0_original              Original unbalanced baseline
run_0                       Duplication-only baseline
run_01 through run_18       Taguchi L18 configurations
```

The legacy preliminary builder generates only 19 configurations and 86 additional clips because it belongs to the earlier approximately 190-clip study. Those values do not describe the final 733-clip paper experiment.

## Expected prepared-data structure

`vivit_train_taguchi.py` expects prepared fold directories in the following structure:

```text
taguchi_runs/
├── run_0_original/
│   ├── fold_0/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── run_0/
│   └── fold_0 ... fold_4/
├── run_01/
│   └── fold_0 ... fold_4/
└── run_18/
    └── fold_0 ... fold_4/
```

Each fold directory must contain:

```text
fold_0/
├── train/
│   └── videos/
├── train_labels.csv
├── val/
│   └── videos/
└── val_labels.csv
```

The training and validation CSV files use:

```text
fname,label
001.mp4,1
002.mp4,0
```

The exact paper fold directories cannot yet be recreated solely from the public repository because the fold-preparation artifact is still being prepared for release.

## Hardware used

| Component | Specification |
|---|---|
| GPU | 1× NVIDIA H100 80 GB |
| CPU | AMD EPYC 7742, 64 cores |
| RAM | 128 GB |
| HPC system | PSC Bridges-2 |
| Operating system | Linux/Ubuntu 20.04 |
| PyTorch CUDA build | CUDA 12.1 |

A CUDA-capable GPU with at least 24 GB of VRAM is recommended for one ViViT fold. Batch size may need to be reduced on smaller GPUs.

## Installation

### Conda

```bash
git clone https://github.com/AhsanZaidi12/tacklestudy_vivit.git
cd tacklestudy_vivit

conda env create -f environment.yml
conda activate vivit_pyt
```

### Pip

```bash
git clone https://github.com/AhsanZaidi12/tacklestudy_vivit.git
cd tacklestudy_vivit

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The PyTorch packages are installed through pip using the official CUDA 12.1 wheel index. They are not installed from the conda-forge PyTorch build.

Verify the environment with:

```bash
nvidia-smi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The NVIDIA driver must support the CUDA 12.1 runtime. Model weights for `google/vivit-b-16x2-kinetics400` are downloaded automatically from Hugging Face on first use.

There is no runtime dependency on a file named `HELPERS.md` in the released scripts.

## Configuring paths

Set the prepared-data and results directories through environment variables:

```bash
export TACKLE_RUNS_DIR=/path/to/taguchi_runs
export TACKLE_RESULTS_DIR=/path/to/taguchi_runs_GRADCAM_RESULTS
```

If the variables are not defined, the defaults are:

```text
./taguchi_runs
./taguchi_runs_GRADCAM_RESULTS
```

## Training one prepared fold

After the prepared fold directories are available:

```bash
python vivit_train_taguchi.py \
    --runs run_15 \
    --fold 0
```

The repository does not currently claim that this command alone reconstructs the exact paper folds from raw video. It trains and evaluates an already prepared fold.

## Training configuration

| Parameter | Value |
|---|---|
| Backbone | `google/vivit-b-16x2-kinetics400` |
| Input shape | 32 frames × 224 × 224 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Maximum epochs | 50 |
| Learning rate | `5e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Scheduler | Cosine |
| Maximum gradient norm | `1.0` |
| Hidden dropout | `0.2` |
| Attention dropout | `0.2` |
| Focal-loss alpha | `0.6` |
| Focal-loss gamma | `1.6` |
| Early-stopping metric | Macro F1 |
| Early-stopping patience | 10 epochs |
| Threshold strategy | Macro-F1 optimization |
| Random seed | 42 |

## Mixed precision

Precision is selected automatically by `vivit_train_taguchi.py`:

- `bf16` on supported GPUs such as the NVIDIA H100;
- `fp16` on CUDA GPUs without `bf16` support; and
- `fp32` when CUDA is unavailable.

Therefore, the paper experiments run on the H100 used `bf16`. Earlier README documentation that listed only `fp16` was incomplete.

## Running the SLURM launcher

`run_vivit_train_taguchi.sh` is a PSC Bridges-2-oriented launcher. It loops over fold IDs 0–4 but assumes that the corresponding prepared fold directories already exist.

Site-specific values such as the allocation, partition, project directory, and storage path must be changed before use on another system.

## Consolidating results

After fold-level `metrics_summary.csv` files have been produced:

```bash
python consolidate_metrics.py
```

For statistical analysis:

```bash
python consolidate_taguchi.py \
    --results_dir "$TACKLE_RESULTS_DIR"
```

`statsmodels` is now included in the pinned environment. The consolidation script also includes a fallback Benjamini-Hochberg FDR implementation if `statsmodels` is unavailable.

## Reported headline results

For Run 15, averaged across five folds:

| Metric | Reported value |
|---|---:|
| Risky recall | 0.67 |
| Risky F1 | 0.59 |
| Overall accuracy | 0.67 |

The fold-level historical result artifacts supporting these aggregates are being organized for release.

## Reproducibility notes

- `SEED = 42` is applied to Python, NumPy, PyTorch, CUDA, cuDNN, and `PYTHONHASHSEED`.
- `DETERMINISTIC_SAMPLER = False` is the historical default. Consequently, minor run-to-run numerical variation is expected.
- Set `--deterministic_sampler` for deterministic weighted-sampler ordering.
- `transformers==4.51.3` is pinned because later releases may change the ViViT API.
- The exact five-fold and FPOC-preparation artifacts remain necessary for complete raw-data-to-result reproduction and will be released separately.

## Citation

```bibtex
@article{zaidi2026vits,
  title   = {ViTs for Action Classification in Videos: An Approach to Risky Tackle Detection in American Football Practice Videos},
  author  = {Zaidi, Syed Ahsan Masud and Hsu, William and Dietrich, Scott},
  journal = {arXiv preprint arXiv:2604.01318},
  year    = {2026}
}
```

## License

The code is released under the MIT License. Dataset use is governed by the license and terms shown on its Kaggle page.
