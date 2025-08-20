"""DINOv2/v3 model implementation for DIET finetuning."""

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def get_dinov2_model(device, model_size="small"):
    """Create DINOv2 model."""
    return get_dino_model(device, model_size, version="v2")


def get_dinov3_model(device, model_size="small"):
    """Create DINOv3 model."""
    return get_dino_model(device, model_size, version="v3")


def get_dino_model(device, model_size, version="v2"):
    """Create DINOv2 or DINOv3 model."""

    # Model configurations
    configs = {
        "v2": {
            "sizes": {"small": 384, "base": 768, "large": 1024, "giant": 1536},
            "model_template": "facebook/dinov2-{size}",
            "feature_key": "cls_token",
        },
        "v3": {
            "sizes": {
                "small": 384,
                "base": 768,
                "large": 1024,
                "s16": 384,
                "b16": 768,
                "l16": 1024,
                "s16plus": 384,
                "h16plus": 1280,
                "7b16": 4096,
            },
            "aliases": {"small": "s16", "base": "b16", "large": "l16"},
            "model_template": "facebook/dinov3-vit{size}-pretrain-lvd1689m",
            "feature_key": "pooler_output",
        },
    }

    if version not in configs:
        raise ValueError(f"Version {version} not supported")

    config = configs[version]

    # Handle model size and aliases
    if version == "v3" and model_size in config.get("aliases", {}):
        actual_size = config["aliases"][model_size]
    else:
        actual_size = model_size

    if model_size not in config["sizes"]:
        raise ValueError(f"Model size {model_size} not supported for DINO{version}")

    # Load model
    if version == "v2":
        model_name = config["model_template"].format(size=actual_size)
    else:
        model_name = config["model_template"].format(size=actual_size)

    base_model = AutoModel.from_pretrained(model_name)

    # Unfreeze all parameters
    for param in base_model.parameters():
        param.requires_grad = True

    class DINOWrapper(nn.Module):
        def __init__(self, model, feature_key, im_size=224):
            super().__init__()
            self.model = model
            self.feature_key = feature_key
            self.im_size = im_size

        def forward(self, x):
            if x.shape[-1] != self.im_size:
                x = F.interpolate(
                    x,
                    size=(self.im_size, self.im_size),
                    mode="bilinear",
                    align_corners=False,
                )

            outputs = self.model(x)

            if self.feature_key == "pooler_output":
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0]

    model = DINOWrapper(base_model, config["feature_key"]).to(device)
    embedding_dim = config["sizes"][model_size]

    return model, embedding_dim
