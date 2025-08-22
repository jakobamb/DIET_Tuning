"""MAE (Masked Autoencoder) model implementation for DIET finetuning."""

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from config.models import MODEL_SPECS


def get_mae_model(device, model_size="base"):
    """Create MAE model."""

    if "mae" not in MODEL_SPECS:
        raise ValueError("MAE not configured in MODEL_SPECS")

    spec = MODEL_SPECS["mae"]

    if model_size not in spec["sizes"]:
        raise ValueError(f"Model size {model_size} not supported for MAE")

    # Map model size to Hugging Face model ID
    model_map = {
        "base": "facebook/vit-mae-base",
        "large": "facebook/vit-mae-large",
        "huge": "facebook/vit-mae-huge",
    }

    model_id = model_map[model_size]

    processor = AutoImageProcessor.from_pretrained(model_id)
    base_model = ViTMAEForPreTraining.from_pretrained(model_id)

    # Unfreeze all parameters
    for param in base_model.parameters():
        param.requires_grad = True

    class MAEWrapper(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor = processor

        def forward(self, x):
            if x.shape[-1] != 224:
                x = F.interpolate(
                    x, size=(224, 224), mode="bilinear", align_corners=False
                )

            # For feature extraction, use the encoder part of MAE
            features = self.model.vit(x).last_hidden_state[:, 0]  # CLS token
            return features

    model = MAEWrapper(base_model, processor).to(device)
    embedding_dim = spec["embedding_dims"][model_size]

    return model, embedding_dim
