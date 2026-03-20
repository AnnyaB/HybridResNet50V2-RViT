# An Experimental Approach to mitigate Shortcut Learning - Pathology-Guided Hybrid CNN-Transformer Models for Explainable Brain Tumour MRI Classification

**University of Hertfordshire — Department of Computer Science**  
**Programme:** Modular BSc (Hons) Computer Science (Artificial Intelligence)  
**Module:** 6COM2017 – Artificial Intelligence Project  

**Project Title:** Mitigating Shortcut Learning in Brain Tumour Detection and Classification  

**Author:** Riya Basak (SRN: 22089065)  
**Supervisor:** Dr Kheng Lee Koay  

**THIS IS AN ONGOING PROJECT; THE FYP REPORT IS NOT YET COMPLETE.**

---

## Problem statement

Brain tumour MRI classifiers can appear accurate while relying on non-tumour shortcuts such as artefacts, skull boundaries, or acquisition bias. This project develops hybrid CNN–Transformer models with experimental guidance modules and explainability tools to encourage tumour-centred reasoning and more interpretable outputs.

This repository contains:

- **offline preprocessing** with leakage-safe SHA1 deduplication, tight-crop preprocessing, and split generation
- **four training variants**
  - Hybrid A (**PFD-A and GSTE-A**)
  - Hybrid B (**PFD-B and GSTE-B**)
  - Ablation without PFD-A / GSTE-A
  - Ablation without PFD-B / GSTE-B
- **post-hoc explainability** outputs using Grad-CAM++ and attention rollout
- a **local Flask demo web app** for qualitative inspection on a single uploaded image

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/brain-tumor-mri-dataset/         # downloaded Kaggle dataset
│   ├── processed/tightcrop/                 # generated 224×224 tight-cropped images
│   └── splits/tightcrop/                    # generated train.csv / val.csv / test.csv
├── docs/
│   ├── dataset_prep.md                      # preprocessing notes
│   └── images/
│       ├── hybrid-a-architecture.png
│       ├── hybrid-b-architecture.png
│       └── demo-app.png
├── results/                                 # preprocessing audit outputs + CSV summaries + plots
├── scripts/
│   ├── data.py                              # BrainMRICSV + build_transforms
│   ├── dataset_prep.py                      # offline preprocessing + leakage-safe dedup + split generation
│   ├── dataset_plots.py                     # dataset plots used by preprocessing
│   └── Confusion_metrics_plot_generator.py  # helper plotting utilities
├── Hybrid-model-with-pfdA-gsteA/
│   ├── models/hybrid_model.py
│   ├── train-A.py
│   └── Xai-A.py
├── Hybrid-model-with-pfdB-gsteB/
│   ├── models/hybrid_model.py
│   ├── train-B.py
│   └── Xai-B.py
├── Hybrid-model-without-pfdA-gsteA/
│   ├── models/hybrid_model.py
│   ├── train-without-A.py
│   └── Xai-without-A.py
├── Hybrid-model-without-pfdB-gsteB/
│   ├── models/hybrid_model.py
│   ├── train-without-B.py
│   └── Xai-without-B.py
└── webapp/
    ├── app.py
    ├── models_registry.json
    ├── requirements.txt
    └── templates/index.html
```

---

## Proposed contributions

This project introduces two experimental guidance modules inside a hybrid CNN–Transformer pipeline:

- **PFD (Pathology-Focused Disentanglement):** a learnable soft spatial mask over the CNN feature map
- **GSTE (Guided Semantic Token Evolution):** reuses the same mask to guide transformer tokens

Two guidance strengths are implemented for matched comparison:

- **Hybrid A:** PFD-A gates only the transformer token pathway; GSTE-A reweights 49 CNN tokens
- **Hybrid B:** PFD-B influences both the CNN descriptor and transformer guidance pathway; GSTE-B weights 196 patch tokens and can optionally shrink them toward highlighted regions

---

## Source code access

This repository is currently **private** and is shared for **academic assessment and supervision**.

If you are an authorised marker or reviewer and would like access:

1. create a GitHub account if you do not already have one
2. email your **GitHub username** to **rb23ack@herts.ac.uk**
3. request access to the private repository
4. accept the GitHub invitation and then clone the repository normally

---

## Reproducibility overview

There are two ways to use this repository:

1. **Reproduce the full project from raw data**, by running preprocessing, training, and XAI scripts
2. **Run the demo web app locally**, using trained checkpoints stored with Git LFS

The demo web app is a **supporting qualitative inspection tool**. It is **not** the main training or evaluation pipeline.

Run the commands below from the main project folder (the top-level repository folder) unless a step tells you to change into another folder such as `webapp/`.

---

### Reproduction order

1. download and extract the Kaggle dataset into `data/raw/brain-tumor-mri-dataset/`
2. run `python scripts/dataset_prep.py`
3. run one of the training scripts for Hybrid A, Hybrid B, Ablation A, or Ablation B
4. optionally run the corresponding XAI script
5. optionally run the demo app from `webapp/`


## Dataset

**Dataset used:** Masoud Nickparvar (2021), *Brain Tumor MRI Dataset*  
Classes: glioma, meningioma, pituitary, and no tumor  
The code uses the label **`notumor`** for the **no tumor** class.

**Reference:**  
Masoud Nickparvar. (2021). *Brain Tumor MRI Dataset* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886

**Dataset note:**  
The benchmark combines multiple public sources, including figshare, SARTAJ, and Br35H. In this project’s curation notes, SARTAJ glioma-class issues were handled by using figshare images instead.

### Dataset setup for reproduction

The raw dataset is **not redistributed** in this repository. To reproduce the work from scratch:

1. download the Kaggle dataset
2. extract it into:

```text
data/raw/brain-tumor-mri-dataset/
```

The preprocessing script expects the raw dataset to be present in that location before it is run.

---

## Preprocessing and split generation

Preprocessing is performed by:

```bash
python scripts/dataset_prep.py
```

### Goals

1. remove duplicates using **SHA1-based leakage-safe deduplication**
2. create a clean **224×224 RGB** dataset using **tight-crop preprocessing**
3. keep Kaggle **Testing** as the held-out test set and create **train/val** from Kaggle Training

### What the preprocessing pipeline does

- scans **7023** raw images
- performs SHA1 deduplication
  - **unique after dedup:** **6726**
  - **duplicates removed:** **297**
- audits image geometry and intensity statistics
  - **total suspect images:** **0**
- applies:
  - EXIF transpose where needed
  - tight crop with **threshold = 5** and **margin = 10**
  - RGB conversion
  - resize to **224×224**

### Files written by preprocessing

Processed images:
```text
data/processed/tightcrop/{train,val,test}/{class}/
```

CSV split files:
```text
data/splits/tightcrop/train.csv
data/splits/tightcrop/val.csv
data/splits/tightcrop/test.csv
```

Audit and summary outputs:
```text
results/
```

### Split sizes after deduplication

- **Train:** 4353
- **Val:** 1089
- **Test:** 1284

### Per-class counts

- **Train:** pituitary 1152, notumor 1080, meningioma 1064, glioma 1057
- **Val:** pituitary 288, notumor 270, meningioma 267, glioma 264
- **Test:** notumor 381, meningioma 304, pituitary 300, glioma 299

---

## Architecture overview

### Hybrid A (PFD-A and GSTE-A)

- input: RGB 224×224
- ResNet50V2 backbone produces `feat` with shape `(B, 2048, 7, 7)`
- CNN descriptor is computed from **ungated** features
- PFD-A gates features only for transformer token formation
- the 7×7 grid becomes **49 tokens**
- GSTE-A reweights those tokens while keeping token count fixed
- four internal rotations are used: **0°, 90°, 180°, 270°**
- a rotation-aware transformer encoder produces the token descriptor
- CNN and transformer descriptors are fused for 4-class classification

![Hybrid A architecture](docs/images/hybrid-a-architecture.png)

### Hybrid B (PFD-B and GSTE-B)

- uses the same backbone and learned pathology mask
- CNN descriptor is computed from **gated** features
- the transformer branch uses raw-image patch tokens (**14×14 = 196 tokens**)
- the guidance mask is upsampled and pooled to the patch grid
- optional token-grid shrinking can reduce computation toward highlighted regions
- the fusion and classification pattern remains the same

![Hybrid B architecture](docs/images/hybrid-b-architecture.png)

---

## Environment and dependencies

### Environment used

- **Python:** 3.12.2
- **Training platform:** Kaggle (Tesla P100)
- **Local development:** macOS
- **Platform string:** `macOS-26.2-arm64-arm-64bit`

### Main stack (preprocessing, training, evaluation, XAI)

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pillow`

### Demo web app stack

- `flask`
- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pillow`
- `matplotlib`

### Install dependencies

For preprocessing, training, and XAI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the demo web app, use the separate environment in `webapp/requirements.txt`.

---

## Training protocol

### Data input

Training scripts do **not** call preprocessing automatically. They read only:

- `data/splits/tightcrop/train.csv`
- `data/splits/tightcrop/val.csv`
- `data/splits/tightcrop/test.csv`

using the shared loader and transforms from:

- `scripts/data.py` → `BrainMRICSV`, `build_transforms`

### Defaults

- epochs: **100**
- batch size: **32**
- seed: **42**
- device: `"cuda"` if available, else `"cpu"`
- dataloader: `num_workers=2`, `pin_memory=True`
- Hybrid B-style runs use `drop_last=True` on the training loader

### Augmentation (train only)

- `RandomRotation(±15°)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomAffine(translate=0.05)`
- optional Gaussian noise `(std=0.02, p=0.5)`
- normalisation with mean = std = `(0.5, 0.5, 0.5)`

Validation and test use tensor conversion and normalisation only.

### Optimisation and scheduling

- optimiser: **AdamW**
- differential learning rates:
  - CNN LR: `1e-4`
  - transformer / fusion LR: `5e-4`
- selective weight decay:
  - `weight_decay = 0.01`
  - no decay for bias, norm, or 1D parameters
- training loss: `CrossEntropy(label_smoothing=0.05)`
- evaluation loss: plain `CrossEntropy`
- scheduler: `CosineAnnealingLR` with `eta_min = 1e-6`
- warmup: freeze CNN for **5 epochs**, then unfreeze and rebuild optimiser and scheduler
- gradient clipping: `max_norm = 1.0`

### Optional flags

- `--amp` for mixed precision
- `--freeze_cnn_bn` to freeze CNN BatchNorm statistics

### Early stopping and checkpointing

- monitored metric: **validation macro-F1**
- patience: **10**
- best checkpoint saved as: `best_model.pt`

Each checkpoint includes weights, class names, normalisation values, training arguments, and model configuration.

### Training artefacts

Each run saves:

- `best_model.pt`
- `history.csv`
- `loss_curves.png`
- `acc_curves.png`
- `confusion_matrix.png`
- `metrics.json`

---

## Train the models

### Hybrid A
```bash
python Hybrid-model-with-pfdA-gsteA/train-A.py
```

### Hybrid B
```bash
python Hybrid-model-with-pfdB-gsteB/train-B.py
```

### Ablation without PFD-A / GSTE-A
```bash
python Hybrid-model-without-pfdA-gsteA/train-without-A.py
```

### Ablation without PFD-B / GSTE-B
```bash
python Hybrid-model-without-pfdB-gsteB/train-without-B.py
```

### Example with optional flag
```bash
python Hybrid-model-with-pfdA-gsteA/train-A.py --amp
```

---

## Run explainability (XAI)

Each variant has a corresponding XAI script for qualitative inspection of model focus.

### Hybrid A
```bash
python Hybrid-model-with-pfdA-gsteA/Xai-A.py
```

### Hybrid B
```bash
python Hybrid-model-with-pfdB-gsteB/Xai-B.py
```

### Ablation A
```bash
python Hybrid-model-without-pfdA-gsteA/Xai-without-A.py
```

### Ablation B
```bash
python Hybrid-model-without-pfdB-gsteB/Xai-without-B.py
```

These scripts generate qualitative outputs such as **Grad-CAM++** and **attention rollout** visualisations.

---

## Reproducibility notes

To support fair comparison across variants:

- preprocessing is performed **once offline**
- all training variants use the same **leakage-aware CSV splits**
- the default random seed is **42**
- the default training length is **100 epochs**
- the best checkpoint is selected using **validation macro-F1**
- test performance is reported on the held-out test split
- Hybrid B-style runs use `drop_last=True`; Hybrid A-style runs do not

Because package versions, CUDA availability, drivers, and hardware can differ across systems, exact runtime behaviour may vary even when the same code and seed are used.

### Expected folder state before training

```text
data/
├── raw/brain-tumor-mri-dataset/
├── processed/tightcrop/
└── splits/tightcrop/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

If these outputs do not exist, the training scripts will not run correctly.

---

## Results

The table below summarises the final test-set performance recorded in each run’s `metrics.json`.

| Model | Test Acc | Macro F1 (test) | Cohen’s Kappa | MCC | Macro Specificity | Best Epoch (val macro-F1) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid A (PFD-A + GSTE-A) | 0.9875 | 0.9875 | 0.9833 | 0.9833 | 0.9959 | 43 |
| Hybrid B (PFD-B + GSTE-B) | 0.9852 | 0.9849 | 0.9802 | 0.9803 | 0.9952 | 14 |
| Without A (ablation) | 0.9875 | 0.9873 | 0.9833 | 0.9834 | 0.9959 | 30 |
| Without B (ablation) | 0.9922 | 0.9920 | 0.9896 | 0.9896 | 0.9975 | 42 |

---

## Explainability and uncertainty

The project supports post-hoc explainability and uncertainty analysis:

- **Grad-CAM++** on the CNN branch
- **attention rollout** on the transformer branch
- **MC Dropout** at inference to estimate predictive mean and variance

These tools are used for qualitative inspection of tumour-centred evidence rather than as replacements for quantitative evaluation.

---

## Getting started for the demo web app

Trained checkpoints are stored using **Git LFS**. Do **not** use GitHub “Download ZIP”, because ZIP downloads may contain pointer files instead of the real `.pt` checkpoints.

If checkpoint files are missing or unexpectedly small after cloning, run:

```bash
git lfs pull
```

### Python note

The demo app was tested on Python **3.12.2** on macOS. On some Windows systems, Python **3.11.x** may be more reliable depending on available PyTorch wheels.

### PyTorch note

Torch installation varies by OS, CPU/GPU, and Python version. If installing `torch` or `torchvision` fails, use the official PyTorch install command for your platform first, then install the remaining packages from `webapp/requirements.txt`.

---

## Run the demo web app (macOS / Linux / Windows)

### macOS

#### 1) Install and enable Git LFS
```bash
brew install git-lfs
git lfs install
```

#### 2) Clone the repository
```bash
cd ~/Downloads
git clone https://github.com/AnnyaB/HybridResNet50V2-RViT.git
cd HybridResNet50V2-RViT
```

#### 3) Check a checkpoint file
```bash
ls -lh Hybrid-model-with-pfdA-gsteA/best_model.pt
```

#### 4) Start the web app
```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Linux (Ubuntu/Debian)

#### 1) Install and enable Git LFS
```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
```

#### 2) Clone the repository
```bash
cd ~/Downloads
git clone https://github.com/AnnyaB/HybridResNet50V2-RViT.git
cd HybridResNet50V2-RViT
```

#### 3) Check a checkpoint file
```bash
ls -lh Hybrid-model-with-pfdA-gsteA/best_model.pt
```

#### 4) Start the web app
```bash
cd webapp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Windows (PowerShell)

#### 1) Install Python 3.11
```powershell
winget install --id Python.Python.3.11 -e
winget upgrade --id Python.Python.3.11
```

#### 2) Install Git
```powershell
winget install --id Git.Git -e
```

#### 3) Install and enable Git LFS
```powershell
winget install --id GitHub.GitLFS -e
git lfs install
```

#### 4) Check that Git and Python are available
```powershell
git --version
git lfs version
py -3.11 --version
```

#### 5) Clone the repository
```powershell
mkdir $env:USERPROFILE\ai_project
cd $env:USERPROFILE\ai_project
git clone https://github.com/AnnyaB/HybridResNet50V2-RViT.git
cd HybridResNet50V2-RViT
```

#### 6) Check a checkpoint file
```powershell
dir Hybrid-model-with-pfdA-gsteA\best_model.pt
```

#### 7) Start the web app
```powershell
cd webapp
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Open the app

After startup, open:

```text
http://127.0.0.1:5000
```

### Demo note

`app.py` starts the local Flask demo web application for model inference and visualisation. It does **not** train or test the models.

### Optional test images

A small test set of 16 images can be used for quick demo checking if it is included alongside the web application materials.

### Demo screenshot

![Demo Web App](docs/images/demo-app.png)

### timm warning note

If you see a timm warning about a deprecated model-name mapping, this is usually **not an error**. It means a model alias name was remapped internally and the model can still load normally.

---

## References

Bolya, D., Fu, C., Dai, X., Zhang, P., Feichtenhofer, C. and Hoffman, J. (2022) *Token merging: Your ViT but faster*. arXiv preprint. https://doi.org/10.48550/arXiv.2210.09461

He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Identity mappings in deep residual networks’, *European Conference on Computer Vision*, pp. 630–645. https://doi.org/10.1007/978-3-319-46493-038

Hugging Face (2019) *timm/resnetv2_50x1_bit.goog_in21k_ft_in1k* [Pretrained model weights]. Available at: https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k_ft_in1k (Accessed: 14 February 2026).

Kleinberg, J. and Tardos, E. (2006) *Algorithm design*. 1st edn. Boston, MA: Pearson Education / Addison-Wesley.

Kolesnikov, A. et al. (2020) ‘Big Transfer (BiT): General visual representation learning’, *European Conference on Computer Vision*. https://doi.org/10.48550/arXiv.1912.11370

Krishnan, P.T., Krishnadoss, P., Khandelwal, M., Gupta, D., Nihaal, A. and Kumar, T.S. (2024) ‘Enhancing brain tumor detection in MRI with a rotation invariant Vision Transformer’, *Frontiers in Neuroinformatics*, 18, 1414925. https://doi.org/10.3389/fninf.2024.1414925

Rao, Y. et al. (2021) ‘DynamicViT: Efficient vision transformers with dynamic token sparsification’, *Advances in Neural Information Processing Systems*. https://doi.org/10.48550/arXiv.2106.02034

Sarada, B., Reddy, K.N., Muktisingh, R., Babu, R. and Babu, B.S.S.V.R. (2025) ‘Brain tumor classification using modified ResNet50V2 deep learning model’, *International Journal of Computing and Digital Systems*, 17(1), pp. 1–11. https://doi.org/10.12785/ijcds/1571021750

Xia, T., Chartsias, A. and Tsaftaris, S.A. (2020) ‘Pseudo-healthy synthesis with pathology disentanglement and adversarial learning’, *Medical Image Analysis*, 64, 101719. https://doi.org/10.1016/j.media.2020.101719

---

## Licensing and medical disclaimer

### Private academic license (All Rights Reserved)

This repository is provided under a **Private Academic / All Rights Reserved** arrangement for **assessment and supervision only**. Access is granted only to approved viewers. No part of this repository may be redistributed, copied, or used outside the scope of academic evaluation without explicit permission from the owner. Do **not** make unauthorised changes to the repository.

### Medical disclaimer

This software is for **research and educational use only**. It is **not** a certified medical device and must **not** be used for clinical diagnosis, patient management, or treatment decisions. Any outputs produced by this code are experimental and may be incorrect.

### Contact

For access requests or repository questions, contact:

- **rb23ack@herts.ac.uk**
- **riyabasak639@gmail.com**