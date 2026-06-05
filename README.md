# An Experimental Approach to Mitigate Shortcut Learning Using Pathology-Guided Hybrid CNN-Transformer Models

**University of Hertfordshire — Department of Computer Science**  
**Programme:** Modular BSc (Hons) Computer Science (Artificial Intelligence)  
**Module:** 6COM2017 – Artificial Intelligence Project  

**Project Title:** Mitigating Shortcut Learning in Brain Tumour MRI Classification  

**Author:** Riya Basak; **Supervised by:** Dr Kheng Lee Koay  

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
 - a reusable **PFD-GSTE guidance library** in `pfd_gste/` for importing the guidance modules into other CNN, Transformer, or hybrid medical image classifiers
---

## Repository structure

```text

.
├── README.md
├── requirements.txt
├── .gitattributes                         # Git LFS tracking rules for model checkpoints
├── .gitignore                             # ignored files and folders
├── CITATION.bib                           # BibTeX citation 
├── CITATION.cff                           # machine-readable citation metadata
├── CODE_OF_CONDUCT.md                     # contributor and responsible-use guidelines                        
├── LICENSE                  
├── Research_Note.pdf                      # research note
├── additional.txt                         # additional submitted project text
├── data/
│   ├── raw/brain-tumor-mri-dataset/       # downloaded Kaggle dataset
│   ├── processed/tightcrop/               # generated 224x224 tight-cropped images
│   └── splits/tightcrop/                  # generated train.csv / val.csv / test.csv
├── docs/
│   ├── dataset_prep.md                    # preprocessing notes
│   └── images/
│       ├── hybrid-a-architecture.png
│       ├── hybrid-b-architecture.png
│       └── demo-app.png
├── pfd_gste/                               # reusable PFD-GSTE guidance modules
│   ├── __init__.py                         # public package interface
│   └── guidance.py                         # PFD, GSTE-A, GSTE-B, patch guidance, and MC-dropout helpers
├── results/                               # preprocessing audit outputs, CSV summaries, plots, and evaluation outputs
├── Misclassified-results/                 # saved misclassification examples and related analysis outputs
├── scripts/
│   ├── data.py                            # BrainMRICSV and build_transforms
│   ├── dataset_prep.py                    # offline preprocessing, leakage-safe deduplication, and split generation
│   ├── dataset_plots.py                   # dataset plots used by preprocessing
│   └── Confusion_metrics_plot_generator.py # helper plotting utilities
├── Hybrid-model-with-pfdA-gsteA/
│   ├── models/
│   │   └── hybrid_model.py
│   ├── train-A.py
│   ├── Xai-A.py
│   └── best_model.pt                      # trained checkpoint stored through Git LFS, if pulled correctly
├── Hybrid-model-with-pfdB-gsteB/
│   ├── models/
│   │   └── hybrid_model.py
│   ├── train-B.py
│   ├── Xai-B.py
│   └── best_model.pt                      # trained checkpoint stored through Git LFS, if pulled correctly
├── Hybrid-model-without-pfdA-gsteA/
│   ├── models/
│   │   └── hybrid_model.py
│   ├── train-without-A.py
│   ├── Xai-without-A.py
│   └── best_model.pt                      # trained checkpoint stored through Git LFS, if pulled correctly
├── Hybrid-model-without-pfdB-gsteB/
│   ├── models/
│   │   └── hybrid_model.py
│   ├── train-without-B.py
│   ├── Xai-without-B.py
│   └── best_model.pt                      # trained checkpoint stored through Git LFS, if pulled correctly
└── webapp/
    ├── app.py                             # local Flask demo app
    ├── models_registry.json               # checkpoint/model registry used by the demo
    ├── requirements.txt                   # separate dependencies for the web app
    └── templates/
        └── index.html                     # demo web interface


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

## Sample OOD Dataset

 For the out-of-distribution ***(OOD)*** demo, a small external sample was taken from Fernando Feltrin’s Brain Tumor MRI Images 44 Classes Kaggle dataset. The dataset is described as a collection of T1, contrast-enhanced T1, and T2 brain MRI images grouped by tumour type, and the class list includes meningioma, together with many other specific tumour categories.

Only five randomly selected T1ce meningioma images were used for the demo. This was done because meningioma is explicitly provided as a named class, whereas the project’s other four-class categories do not map cleanly to this dataset: pituitary and no tumour are not listed as classes on the dataset page, and glioma is not presented as one single class but is split across multiple, more specific tumour labels. 


Therefore, this dataset was used only for a small qualitative OOD demonstration, not for a formal benchmark evaluation.

## References

Bolya, D., Fu, C., Dai, X., Zhang, P., Feichtenhofer, C. and Hoffman, J. (2022) *Token merging: Your ViT but faster*. arXiv preprint. https://doi.org/10.48550/arXiv.2210.09461

da Costa-Luis, C.O. (2019) tqdm: a fast, extensible progress meter for Python and CLI, Journal of Open Source Software, 4(37), 1277. https://doi.org/10.21105/joss.01277

Feltrin, F. (2023) Brain Tumor MRI Images 44 Classes [dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c 
(Accessed: 15 April 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M.H., Brett, M., Haldane, A., del Río, J.F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C. and Oliphant, T.E. (2020) Array programming with NumPy, Nature, 585, pp. 357-362. https://doi.org/10.1038/s41586-020-2649-2

He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Identity mappings in deep residual networks’, *European Conference on Computer Vision*, pp. 630–645. https://doi.org/10.1007/978-3-319-46493-038

Hugging Face (2019) *timm/resnetv2_50x1_bit.goog_in21k_ft_in1k* [Pretrained model weights]. Available at: https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k_ft_in1k (Accessed: 14 February 2026).

Hunter, J.D. (2007) Matplotlib: a 2D graphics environment, Computing in Science & Engineering, 9(3), pp. 90-95. https://doi.org/10.1109/MCSE.2007.55

Kleinberg, J. and Tardos, E. (2006) *Algorithm design*. 1st edn. Boston, MA: Pearson Education / Addison-Wesley.

Kolesnikov, A. et al. (2020) ‘Big Transfer (BiT): General visual representation learning’, *European Conference on Computer Vision*. https://doi.org/10.48550/arXiv.1912.11370

Krishnan, P.T., Krishnadoss, P., Khandelwal, M., Gupta, D., Nihaal, A. and Kumar, T.S. (2024) ‘Enhancing brain tumor detection in MRI with a rotation invariant Vision Transformer’, *Frontiers in Neuroinformatics*, 18, 1414925. https://doi.org/10.3389/fninf.2024.1414925

McKinney, W. (2010) Data structures for statistical computing in Python, in van der Walt, S. and Millman, J. (eds.) Proceedings of the 9th Python in Science Conference, pp. 56-61. https://doi.org/10.25080/Majora-92bf1922-00a

Pallets (2024) Flask documentation. Available at: https://flask.palletsprojects.com/
(Accessed: 12 February 2026). 

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019) PyTorch: an imperative style, high-performance deep learning library, in Wallach, H., Larochelle, H., Beygelzimer, A., d’Alché-Buc, F., Fox, E. and Garnett, R. (eds.) Advances in Neural Information Processing Systems, 32, pp. 8024-8035. https://doi.org/10.48550/arXiv.1912.01703

PyTorch (2024) torchvision documentation. Available at: https://docs.pytorch.org/vision/main/index.html
 (Accessed: 22 October 2025).

Rao, Y. et al. (2021) ‘DynamicViT: Efficient vision transformers with dynamic token sparsification’, *Advances in Neural Information Processing Systems*. https://doi.org/10.48550/arXiv.2106.02034

Sarada, B., Reddy, K.N., Muktisingh, R., Babu, R. and Babu, B.S.S.V.R. (2025) ‘Brain tumor classification using modified ResNet50V2 deep learning model’, *International Journal of Computing and Digital Systems*, 17(1), pp. 1–11. https://doi.org/10.12785/ijcds/1571021750

Xia, T., Chartsias, A. and Tsaftaris, S.A. (2020) ‘Pseudo-healthy synthesis with pathology disentanglement and adversarial learning’, *Medical Image Analysis*, 64, 101719. https://doi.org/10.1016/j.media.2020.101719


---

## Reusable PFD-GSTE guidance library

The folder `pfd_gste/` contains reusable PyTorch modules for pathology-focused feature gating and guided token reweighting. These modules are not tied to the original ResNet50V2-RViT model. They can be imported into other CNN, Transformer, or hybrid classifiers for related medical image classification tasks, such as brain, breast, lung, retinal, or other tumour-classification problems.

* `PFDGSTEVariantA`: feature-token guidance for models where transformer tokens are produced from CNN feature maps.
* `PFDGSTEVariantB`: patch-token guidance for models where raw-image patch tokens are guided by a CNN-derived pathology mask.
* `PathologyFocusedGate`: standalone soft spatial feature gating.
* `mc_dropout_predict`: helper for MC-dropout uncertainty estimation.

### Import

```python
from pfd_gste import PFDGSTEVariantA, PFDGSTEVariantB
```

### Minimal Variant A example

Use Variant A when a CNN backbone returns a spatial feature map and the transformer operates on feature tokens.

```python
import torch.nn as nn

from pfd_gste import PFDGSTEVariantA


class GuidedFeatureClassifier(nn.Module):
    def __init__(self, backbone, feature_channels, num_classes, embed_dim=128):
        super().__init__()

        self.backbone = backbone
        self.guidance = PFDGSTEVariantA(feature_channels, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        features = self.backbone(images)
        tokens, mask, alpha = self.guidance(features)

        tokens = self.encoder(tokens)
        logits = self.classifier(tokens.mean(dim=1))

        return logits
```

### Minimal Variant B example

Use Variant B when PFD should guide the CNN feature pathway and GSTE should guide raw-image patch tokens.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from pfd_gste import PFDGSTEVariantB


class GuidedPatchClassifier(nn.Module):
    def __init__(self, backbone, feature_channels, num_classes, embed_dim=128):
        super().__init__()

        self.backbone = backbone
        self.guidance = PFDGSTEVariantB(
            in_channels=feature_channels,
            embed_dim=embed_dim,
            patch_size=16,
            min_side=7,
            max_shrink=0.50,
        )

        self.feature_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(feature_channels, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, images):
        features = self.backbone(images)

        gated_features, tokens, mask, alpha, token_hw = self.guidance(
            images,
            features,
            shrink=True,
        )

        cnn_vec = self.feature_pool(gated_features).flatten(1)
        cnn_vec = F.relu(self.feature_proj(cnn_vec), inplace=True)

        tokens = self.encoder(tokens)
        token_vec = tokens.mean(dim=1)

        logits = self.classifier(torch.cat([cnn_vec, token_vec], dim=1))

        return logits
```

### Minimal training step

The modules can be trained normally as part of a PyTorch model.

```python
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimiser, device):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count
```

### Import check

From the repository root:

```bash
python - <<'PY'
from pfd_gste import PFDGSTEVariantA, PFDGSTEVariantB
print("PFD-GSTE library imports correctly.")
PY
```

The library is included as a local project package. It can be *reused* inside this repository without separate installation, as long as scripts are run from the repository root or the repository root is on `PYTHONPATH`.


## License and citation

### License

This project is released under the MIT License. This means the code may be used, copied, modified, merged, published, distributed, sublicensed, and reused in future research or software projects, provided that the original copyright notice and MIT License text are included.

### Citation

If this repository, code, trained models, or PFD-GSTE guidance modules are useful in your work, please cite:

Basak, R. (2026) *Mitigating Shortcut Learning in Brain Tumour MRI Classification*. BSc Artificial Intelligence Project, University of Hertfordshire. Available at: https://github.com/AnnyaB/HybridResNet50V2-RViT

### Medical disclaimer

This software is for research and educational use only. It is **not** a certified medical device and must not be used for clinical diagnosis, patient management, or treatment decisions. Any outputs produced by this code are experimental and *may be* incorrect.

## Contact and contributions

For questions, reproducibility issues, or suggested improvements, please open a GitHub issue.
