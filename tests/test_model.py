"""
Tests for GeolocationCLIP adaptation strategies.
Run with: uv run pytest tests/ -v

open_clip is patched with a lightweight stub so the full ViT-L/14 checkpoint
(~878 MB) is never downloaded.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from PIL import Image
from torchvision import transforms

CLASSES = ["France", "Germany", "Japan", "Brazil", "Australia", "India", "Canada"]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 2
_EMBED_DIM = 768  # must match GeolocationCLIP._EMBED_DIM


# ---------------------------------------------------------------------------
# open_clip stub
# ---------------------------------------------------------------------------

class _StubCLIP(nn.Module):
    """Minimal CLIP stand-in with the correct interface but no downloaded weights."""

    def __init__(self):
        super().__init__()
        # Real parameters so named_parameters() / requires_grad toggling works.
        self.proj = nn.Linear(_EMBED_DIM, _EMBED_DIM, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) → (B, _EMBED_DIM): per-sample mean expanded through proj
        B = images.shape[0]
        feat = images.flatten(1).mean(dim=1, keepdim=True).expand(B, _EMBED_DIM)
        return self.proj(feat.contiguous())

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # (C, 77) → (C, _EMBED_DIM): seeded random gives distinct per-class embeddings
        C = tokens.shape[0]
        g = torch.Generator()
        g.manual_seed(0)
        return torch.randn(C, _EMBED_DIM, generator=g)


_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _stub_tokenizer(texts):
    if isinstance(texts, str):
        texts = [texts]
    return torch.zeros(len(texts), 77, dtype=torch.long)


def _stub_create(model_name, pretrained=None, **kwargs):
    return _StubCLIP(), None, _PREPROCESS


def _stub_get_tokenizer(model_name):
    return _stub_tokenizer


@pytest.fixture(autouse=True, scope="session")
def patch_open_clip():
    with patch("open_clip.create_model_and_transforms", _stub_create), \
         patch("open_clip.get_tokenizer", _stub_get_tokenizer):
        yield


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_batch(patch_open_clip):
    from src.models import GeolocationCLIP
    model = GeolocationCLIP(NUM_CLASSES, CLASSES, mode="zero_shot")
    img = Image.new("RGB", (224, 224))
    tensor = model.preprocess(img).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
    return tensor  # (B, 3, 224, 224) on CPU


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

class TestZeroShot:
    @pytest.fixture(scope="class")
    def model(self, patch_open_clip):
        from src.models import GeolocationCLIP
        return GeolocationCLIP(NUM_CLASSES, CLASSES, mode="zero_shot")

    def test_output_shape(self, model, dummy_batch):
        # Model produces valid (B, C) logits
        with torch.no_grad():
            logits = model(dummy_batch)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_no_trainable_parameters(self, model):
        # Zero-shot is pure inference, nothing to train
        counts = model.parameter_counts()
        assert counts["trainable"] == 0, (
            f"zero_shot should have 0 trainable params, got {counts['trainable']:,}"
        )

    def test_text_embeddings_built(self, model, dummy_batch):
        # Text encoder produces one embedding per class
        with torch.no_grad():
            model(dummy_batch)
        assert hasattr(model, "text_embeddings")
        assert model.text_embeddings.shape == (NUM_CLASSES, _EMBED_DIM)

    def test_text_embeddings_normalised(self, model, dummy_batch):
        # Embeddings are unit-norm, so dot product == cosine similarity
        with torch.no_grad():
            model(dummy_batch)
        norms = model.text_embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(NUM_CLASSES), atol=1e-5)

    def test_different_classes_differ(self, model, dummy_batch):
        # Model distinguishes between countries in embedding space
        with torch.no_grad():
            model(dummy_batch)
        # If all embeddings were identical, the pairwise cosine sim matrix
        # would be all-ones off-diagonal.
        embs = model.text_embeddings  # (C, d)
        sim = embs @ embs.T
        off_diag = sim[~torch.eye(NUM_CLASSES, dtype=torch.bool)]
        assert (off_diag < 0.999).all(), "text embeddings are not distinct per class"


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class TestLinearProbe:
    @pytest.fixture(scope="class")
    def model(self, patch_open_clip):
        from src.models import GeolocationCLIP
        return GeolocationCLIP(NUM_CLASSES, CLASSES, mode="linear_probe")

    def test_output_shape(self, model, dummy_batch):
        # Model produces valid (B, C) logits
        with torch.no_grad():
            logits = model(dummy_batch)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_only_head_is_trainable(self, model):
        # Encoder is frozen - only the linear head trains
        for name, param in model.clip.named_parameters():
            assert not param.requires_grad, f"clip.{name} should be frozen"
        for name, param in model.head.named_parameters():
            assert param.requires_grad, f"head.{name} should be trainable"

    def test_trainable_param_count(self, model):
        # Ensure head has <1% of total params to confirm parameter efficiency
        counts = model.parameter_counts()
        head_params = sum(p.numel() for p in model.head.parameters())
        assert counts["trainable"] == head_params
        ratio = counts["trainable"] / counts["total"]
        assert ratio < 0.01, f"expected <1% trainable, got {ratio:.2%}"

    def test_head_gradient_flows(self, model, dummy_batch):
        # Backprop reaches the head but stops at the frozen encoder
        logits = model(dummy_batch)
        loss = logits.sum()
        loss.backward()
        for name, param in model.head.named_parameters():
            assert param.grad is not None, f"head.{name} has no gradient"
        for name, param in model.clip.named_parameters():
            assert param.grad is None, f"clip.{name} should have no gradient"

    def test_head_dimensions(self, model):
        # Head input/output dims match embed dim and number of classes
        assert model.head.in_features == _EMBED_DIM
        assert model.head.out_features == NUM_CLASSES


# ---------------------------------------------------------------------------
# Full fine-tuning
# ---------------------------------------------------------------------------

class TestFullFinetune:
    @pytest.fixture(scope="class")
    def model(self, patch_open_clip):
        from src.models import GeolocationCLIP
        return GeolocationCLIP(NUM_CLASSES, CLASSES, mode="full_finetune")

    def test_encoder_and_head_are_trainable(self, model):
        for name, param in model.clip.named_parameters():
            assert param.requires_grad, f"clip.{name} should be trainable"
        for name, param in model.head.named_parameters():
            assert param.requires_grad, f"head.{name} should be trainable"

    def test_optimizer_param_groups_split_lrs_and_weight_decay(self, model):
        groups = model.optimizer_param_groups(
            encoder_lr=1e-6,
            head_lr=1e-4,
            weight_decay=1e-2,
        )
        assert len(groups) == 4
        assert groups[0]["lr"] == pytest.approx(1e-6)
        assert groups[1]["lr"] == pytest.approx(1e-6)
        assert groups[2]["lr"] == pytest.approx(1e-4)
        assert groups[3]["lr"] == pytest.approx(1e-4)
        assert groups[0]["weight_decay"] == pytest.approx(1e-2)
        assert groups[1]["weight_decay"] == pytest.approx(0.0)
        assert groups[2]["weight_decay"] == pytest.approx(1e-2)
        assert groups[3]["weight_decay"] == pytest.approx(0.0)

        all_grouped = set()
        for group in groups:
            for param in group["params"]:
                assert id(param) not in all_grouped, "parameter duplicated across groups"
                all_grouped.add(id(param))

        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert all_grouped == expected


# ---------------------------------------------------------------------------
# Mode switching
# ---------------------------------------------------------------------------

class TestModes:
    def test_set_mode_freezes_encoder(self, patch_open_clip):
        # set_mode correctly toggles encoder grad requirements
        from src.models import GeolocationCLIP
        model = GeolocationCLIP(NUM_CLASSES, CLASSES, mode="full_finetune")
        assert any(p.requires_grad for p in model.clip.parameters())
        model.set_mode("linear_probe")
        assert not any(p.requires_grad for p in model.clip.parameters())

    def test_optimizer_groups_reject_non_full_finetune_mode(self, patch_open_clip):
        from src.models import GeolocationCLIP
        model = GeolocationCLIP(NUM_CLASSES, CLASSES, mode="linear_probe")
        with pytest.raises(ValueError):
            model.optimizer_param_groups(1e-6, 1e-4, 1e-2)

    def test_invalid_mode_raises(self):
        # Unknown modes are caught at init time
        with pytest.raises(AssertionError):
            from src.models import GeolocationCLIP
            GeolocationCLIP(NUM_CLASSES, CLASSES, mode="invalid")
