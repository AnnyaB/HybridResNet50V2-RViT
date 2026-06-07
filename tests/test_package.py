import torch
import torch.nn as nn

from pfd_gste import (
    FeatureTokenGuidance,
    PatchEmbed2d,
    PatchTokenGuidance,
    PathologyFocusedGate,
    PFDGSTEVariantA,
    PFDGSTEVariantB,
    enable_mc_dropout,
    mc_dropout_predict,
)


def test_public_api_is_importable():
    public_objects = [
        FeatureTokenGuidance,
        PatchEmbed2d,
        PatchTokenGuidance,
        PathologyFocusedGate,
        PFDGSTEVariantA,
        PFDGSTEVariantB,
        enable_mc_dropout,
        mc_dropout_predict,
    ]

    assert all(callable(obj) for obj in public_objects)


def test_pathology_focused_gate_shapes_and_range():
    module = PathologyFocusedGate(in_channels=8)
    features = torch.randn(2, 8, 7, 7)

    gated_features, mask = module(features)

    assert gated_features.shape == features.shape
    assert mask.shape == (2, 1, 7, 7)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)


def test_feature_token_guidance_shapes():
    module = FeatureTokenGuidance()
    tokens = torch.randn(2, 49, 16)
    mask = torch.rand(2, 1, 7, 7)

    guided_tokens, alpha = module(tokens, mask)

    assert guided_tokens.shape == tokens.shape
    assert alpha.shape == (2, 49, 1)


def test_patch_embedding_shapes():
    module = PatchEmbed2d(
        in_channels=3,
        embed_dim=16,
        patch_size=4,
    )
    images = torch.randn(2, 3, 32, 32)

    tokens, token_h, token_w = module(images)

    assert tokens.shape == (2, 64, 16)
    assert (token_h, token_w) == (8, 8)


def test_patch_token_guidance_without_shrinking():
    module = PatchTokenGuidance(min_side=2)
    tokens = torch.randn(2, 64, 16)
    mask = torch.rand(2, 1, 8, 8)

    guided_tokens, alpha, output_hw = module(
        tokens,
        mask,
        token_hw=(8, 8),
        shrink=False,
    )

    assert guided_tokens.shape == tokens.shape
    assert alpha.shape == (2, 1, 8, 8)
    assert output_hw == (8, 8)


def test_variant_a_forward_shapes():
    module = PFDGSTEVariantA(
        in_channels=8,
        embed_dim=16,
    )
    features = torch.randn(2, 8, 4, 4)

    tokens, mask, alpha = module(features)

    assert tokens.shape == (2, 16, 16)
    assert mask.shape == (2, 1, 4, 4)
    assert alpha.shape == (2, 16, 1)


def test_variant_b_forward_shapes_without_shrinking():
    module = PFDGSTEVariantB(
        in_channels=8,
        embed_dim=16,
        image_channels=3,
        patch_size=4,
        min_side=2,
    )

    images = torch.randn(2, 3, 32, 32)
    features = torch.randn(2, 8, 4, 4)

    gated_features, tokens, mask, alpha, output_hw = module(
        images,
        features,
        shrink=False,
    )

    assert gated_features.shape == features.shape
    assert tokens.shape == (2, 64, 16)
    assert mask.shape == (2, 1, 32, 32)
    assert alpha.shape == (2, 1, 8, 8)
    assert output_hw == (8, 8)


def test_enable_mc_dropout_changes_only_dropout_layers():
    model = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(4, 2),
    )
    model.eval()

    enable_mc_dropout(model)

    assert model[0].training is True
    assert model[1].training is False


def test_mc_dropout_predict_output_shapes_and_probabilities():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(4, 3),
    )
    images = torch.randn(2, 1, 2, 2)

    mean, variance = mc_dropout_predict(
        model,
        images,
        samples=5,
    )

    assert mean.shape == (2, 3)
    assert variance.shape == (2, 3)
    assert torch.all(variance >= 0.0)
    assert torch.allclose(
        mean.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
