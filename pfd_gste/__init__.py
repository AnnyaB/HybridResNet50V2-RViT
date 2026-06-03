

"""
Public interface for the reusable PFD-GSTE guidance package.

The package exposes lightweight PyTorch modules for pathology-focused feature
gating, token guidance, optional patch-token reduction, and MC-dropout inference.

"""




from .guidance import (
    FeatureTokenGuidance,
    PatchEmbed2d,
    PatchTokenGuidance,
    PathologyFocusedGate,
    PFDGSTEVariantA,
    PFDGSTEVariantB,
    enable_mc_dropout,
    mc_dropout_predict,
)

__all__ = [
    "FeatureTokenGuidance",
    "PatchEmbed2d",
    "PatchTokenGuidance",
    "PathologyFocusedGate",
    "PFDGSTEVariantA",
    "PFDGSTEVariantB",
    "enable_mc_dropout",
    "mc_dropout_predict",
]
