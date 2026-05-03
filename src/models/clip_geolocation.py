"""
Geolocation classifier built on OpenCLIP ViT-L/14 (LAION-2B checkpoint).

Supported adaptation modes
--------------------------
zero_shot     : Image and text (country name) are independently encoded; class is predicted by
                cosine similarity between the image embedding and each text class embedding,
                scaled by the learned CLIP logit scale. No parameters are trained.

linear_probe  : Encoder is frozen. A single linear layer is trained on top of the frozen image
                embeddings to predict geographic class.

full_finetune : All encoder parameters and the classification head are unfrozen and trained
                end-to-end on the geolocation dataset.

lora          : Low-rank adaptation (LoRA) modules are inserted into the transformer layers of
                the frozen encoder. Only the LoRA parameters and the classification head are
                trained. LoRA adapters are injected externally via peft after calling
                set_mode("lora").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class GeolocationCLIP(nn.Module):
    # ViT-L/14, LAION-2B — pretrained key used by open_clip
    _MODEL_NAME = "ViT-L-14"
    _PRETRAINED = "laion2b_s32b_b82k"
    _EMBED_DIM = 768  # projected visual/text embedding dimension for this checkpoint

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        mode: str = "linear_probe",
        prompt_template: str = "a photo taken in {}",
    ):
        """
        Parameters
        ----------
        num_classes:      Number of geographic classes.
        class_names:      Ordered list of class name strings matching label indices.
        mode:             Adaptation strategy: zero_shot | linear_probe | full_finetune | lora.
        prompt_template:  Format string used to build per-class text prompts for zero-shot.
        """
        super().__init__()
        assert mode in ("zero_shot", "linear_probe", "full_finetune", "lora"), \
            f"Unknown mode: {mode}"

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            self._MODEL_NAME, pretrained=self._PRETRAINED
        )
        self.clip = clip_model
        # Expose preprocess so callers can pass it as the dataset transform.
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self._MODEL_NAME)

        self.num_classes = num_classes
        self.class_names = class_names
        self.prompt_template = prompt_template

        # Single linear head shared by linear_probe / full_finetune / lora.
        # Not used in zero_shot, but kept so checkpoints are compatible across modes.
        self.head = nn.Linear(self._EMBED_DIM, num_classes)

        self._configure_mode(mode)

        if mode == "zero_shot":
            # Text embeddings are computed lazily on first forward call so that
            # the model can be moved to the right device first.
            self._text_embeddings_built = False

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def _configure_mode(self, mode: str):
        self.mode = mode
        encoder_grad = mode in ("full_finetune", "lora")
        for param in self.clip.parameters():
            param.requires_grad = encoder_grad
        # Head is only trained in non-zero-shot modes
        head_grad = mode != "zero_shot"
        for param in self.head.parameters():
            param.requires_grad = head_grad

    def set_mode(self, mode: str):
        """Switch adaptation strategy at runtime."""
        self._configure_mode(mode)
        if mode == "zero_shot" and not hasattr(self, "_text_embeddings_built"):
            self._text_embeddings_built = False

    # TODO (lora): LoRA adapter injection not yet implemented.
    #   - Add peft to pyproject.toml dependencies
    #   - After set_mode("lora"), wrap self.clip:
    #       from peft import get_peft_model, LoraConfig
    #       config = LoraConfig(target_modules=["attn.in_proj", "attn.out_proj",
    #                                           "mlp.c_fc", "mlp.c_proj"],
    #                           r=16, lora_alpha=32, lora_dropout=0.1)
    #       model.clip = get_peft_model(model.clip, config)
    #   - Only LoRA params + self.head will have requires_grad=True
    #   - Verify with model.parameter_counts() that trainable params are << total

    # ------------------------------------------------------------------
    # Zero-shot text embeddings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_text_embeddings(self):
        """Compute and register normalized text embeddings for zero-shot inference."""
        device = next(self.clip.parameters()).device
        prompts = [self.prompt_template.format(c) for c in self.class_names]
        tokens = self.tokenizer(prompts).to(device)
        text_embs = self.clip.encode_text(tokens)
        text_embs = F.normalize(text_embs, dim=-1)
        self.register_buffer("text_embeddings", text_embs)  # (C, d)
        self._text_embeddings_built = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw (unnormalized) image embeddings."""
        return self.clip.encode_image(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns logits of shape (B, num_classes).

        For zero_shot: scaled cosine similarities to text class embeddings.
        For all other modes: linear projection of image embeddings.
        """
        if self.mode == "zero_shot":
            if not self._text_embeddings_built:
                self.build_text_embeddings()
            img_embs = F.normalize(self.encode_image(images), dim=-1)
            logit_scale = self.clip.logit_scale.exp()
            return logit_scale * (img_embs @ self.text_embeddings.T)

        # linear_probe / full_finetune / lora
        img_embs = self.encode_image(images)
        return self.head(img_embs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only parameters that require grad (for optimizer construction)."""
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def optimizer_param_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float,
    ) -> list[dict]:
        """Build AdamW-style parameter groups for full fine-tuning.

        Biases and 1D parameters (for example LayerNorm scales) are excluded from
        weight decay. The encoder and classification head receive separate learning
        rates so the pretrained backbone can be updated more conservatively.
        """
        if self.mode != "full_finetune":
            raise ValueError(
                "optimizer_param_groups() is intended for full_finetune mode; "
                f"current mode is {self.mode!r}"
            )

        def split_decay(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
            decay_params: list[nn.Parameter] = []
            no_decay_params: list[nn.Parameter] = []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith("bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            return decay_params, no_decay_params

        encoder_decay, encoder_no_decay = split_decay(self.clip)
        head_decay, head_no_decay = split_decay(self.head)

        return [
            {"params": encoder_decay, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]
