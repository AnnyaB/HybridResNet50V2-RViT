# PFD–GSTE

Reusable PyTorch modules for pathology-focused feature gating and guided token reweighting in CNN, Transformer and hybrid image classifiers.

PFD–GSTE was developed as part of the research project **Mitigating Shortcut Learning in Brain Tumour MRI Classification**. This package isolates the reusable guidance components from the complete experimental repository.

## Included modules

* `PathologyFocusedGate` learns a soft spatial guidance mask from a CNN feature map.
* `FeatureTokenGuidance` applies a spatial mask to CNN-derived transformer tokens.
* `PatchEmbed2d` provides lightweight two-dimensional patch embedding.
* `PatchTokenGuidance` guides image patch tokens and can optionally reduce the token grid.
* `PFDGSTEVariantA` combines pathology-focused gating with feature-token guidance.
* `PFDGSTEVariantB` combines pathology-focused gating with image patch-token guidance.
* `enable_mc_dropout` enables dropout layers during inference.
* `mc_dropout_predict` performs repeated stochastic inference for MC-dropout estimation.

## Installation

After publication to PyPI:

```bash
pip install pfd-gste
```

## Basic import

```python
from pfd_gste import (
    PathologyFocusedGate,
    PFDGSTEVariantA,
    PFDGSTEVariantB,
    mc_dropout_predict,
)
```

## Variant A

Variant A is intended for models whose transformer tokens are produced from a CNN feature map.

```python
import torch

from pfd_gste import PFDGSTEVariantA

guidance = PFDGSTEVariantA(
    in_channels=2048,
    embed_dim=128,
)

features = torch.randn(2, 2048, 7, 7)
tokens, mask, alpha = guidance(features)

print(tokens.shape)
print(mask.shape)
print(alpha.shape)
```

## Variant B

Variant B combines a CNN-derived pathology mask with image patch tokens.

```python
import torch

from pfd_gste import PFDGSTEVariantB

guidance = PFDGSTEVariantB(
    in_channels=2048,
    embed_dim=128,
    image_channels=3,
    patch_size=16,
    min_side=7,
    max_shrink=0.50,
)

images = torch.randn(2, 3, 224, 224)
features = torch.randn(2, 2048, 7, 7)

gated_features, tokens, mask, alpha, token_hw = guidance(
    images,
    features,
    shrink=True,
)

print(gated_features.shape)
print(tokens.shape)
print(mask.shape)
print(alpha.shape)
print(token_hw)
```

## Complete research repository

The complete repository contains the preprocessing pipeline, four matched model variants, training and held-out evaluation workflows, explainability scripts, recorded results, trained checkpoints and local Flask prototype:

`https://github.com/AnnyaB/HybridResNet50V2-RViT`

The Python package contains only the reusable PFD–GSTE guidance components. It does not contain datasets, trained checkpoints, complete classifiers, experimental results or clinical software.

## Research-use notice

This package is provided for research and educational use only. It is not a certified medical device and must not be used for clinical diagnosis, patient management or treatment decisions.

## Licence

MIT License.
