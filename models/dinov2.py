"""DINOv2/v3 model implementation for DIET finetuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


def get_dinov2_model(device, model_size="small"):
    """Create DINOv2 model with memory optimization"""
    return get_dino_model(device, model_size, version="v2")


def get_dinov3_model(device, model_size="s16"):
    """Create DINOv3 model with memory optimization"""
    return get_dino_model(device, model_size, version="v3")


def get_dino_model(device, model_size, version="v2"):
    """
    Unified function to create DINOv2 or DINOv3 models

    Args:
        device: The device to put the model on
        model_size: Model size identifier
        version: Either "v2" or "v3"
    """
    print(f"Loading DINO{version}-{model_size} model...")

    if version == "v2":
        # DINOv2 model size to embedding dimension mapping
        dim_map = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
        model_name = f"facebook/dinov2-{model_size}"

    elif version == "v3":
        # DINOv3 model size to embedding dimension mapping
        dim_map = {
            "s16": 384,  # ViT-S/16
            "s16plus": 384,  # ViT-S+/16
            "b16": 768,  # ViT-B/16
            "l16": 1024,  # ViT-L/16
            "h16plus": 1280,  # ViT-H+/16
            "7b16": 4096,  # ViT-7B/16
            # Convenient aliases
            "small": 384,  # alias for s16
            "base": 768,  # alias for b16
            "large": 1024,  # alias for l16
            # ConvNeXt variants
            "convnext-tiny": 768,
            "convnext-small": 768,
            "convnext-base": 1024,
            "convnext-large": 1536,
        }

        # Map model size to full model name for DINOv3
        # Handle aliases first
        model_size_mapping = {"small": "s16", "base": "b16", "large": "l16"}
        actual_model_size = model_size_mapping.get(model_size, model_size)

        if actual_model_size.startswith("convnext"):
            model_name = f"facebook/dinov3-{actual_model_size}-pretrain-lvd1689m"
        else:
            model_name = f"facebook/dinov3-vit{actual_model_size}-pretrain-lvd1689m"
    else:
        raise ValueError(f"Unsupported version: {version}. Choose 'v2' or 'v3'")

    if model_size not in dim_map:
        raise ValueError(
            f"Model size {model_size} not supported for DINO{version}. Choose from {list(dim_map.keys())}"
        )

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # UNFREEZE ALL PARAMETERS
    print(f"Unfreezing all DINO{version} parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in DINO{version} backbone")

    class DINOWrapper(nn.Module):
        def __init__(self, model, version):
            super().__init__()
            self.model = model
            self.processor = processor
            self.version = version

        def forward(self, x):
            # Make x require gradients to force gradient flow
            x = x.detach().requires_grad_(True)

            # Process smaller batches if needed
            batch_size = x.shape[0]
            if batch_size > 16 and x.device.type == "cuda":
                # Process in chunks to save memory
                outputs_list = []
                for i in range(0, batch_size, 16):
                    # Get batch chunk
                    x_chunk = x[i : i + 16]
                    # Resize to expected input (224x224)
                    if x_chunk.shape[-1] != 224:
                        x_chunk = F.interpolate(
                            x_chunk,
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )
                    # Process chunk WITHOUT autocast and with gradient tracking
                    chunk_output = self.model(x_chunk)

                    # Extract features based on version
                    if self.version == "v3":
                        # DINOv3 uses pooler_output
                        features = chunk_output.pooler_output
                    else:
                        # DINOv2 uses class token (first token)
                        features = chunk_output.last_hidden_state[:, 0]

                    outputs_list.append(features)
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                if x.shape[-1] != 224:
                    x = F.interpolate(
                        x, size=(224, 224), mode="bilinear", align_corners=False
                    )
                # Process WITHOUT autocast and with gradient tracking
                outputs = self.model(x)

                # Extract features based on version
                if self.version == "v3":
                    # DINOv3 uses pooler_output
                    return outputs.pooler_output
                else:
                    # DINOv2 uses class token (first token)
                    return outputs.last_hidden_state[:, 0]

    model = DINOWrapper(base_model, version).to(device)
    embedding_dim = dim_map[model_size]

    print(f"DINO{version}-{model_size} loaded. Embedding dimension: {embedding_dim}")
    return model, embedding_dim


def apply_strategic_freezing(model):
    """Apply strategic freezing to DINOv2 model - only freeze embeddings"""
    print("Applying minimal freezing to DINOv2 (allowing more gradients)...")
    frozen_params = 0
    total_params = 0

    for name, param in model.model.named_parameters():
        total_params += 1
        # Only freeze embeddings, unfreeze all transformer layers
        if "embeddings" in name:  # Only freeze embeddings, not encoder layers
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True  # Explicitly set other layers to trainable

    print(f"Frozen {frozen_params} out of {total_params} parameters")

    # Add diagnostic to check which layers are trainable
    print("\nTrainable layers in DINOv2:")
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    return model
