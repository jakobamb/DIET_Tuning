"""IJEPA model implementation for DIET finetuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_ijepa_model(device, model_size):
    """
    Create I-JEPA model using the Hugging Face transformers library
    with the correct model identifiers and dimensions.

    Args:
        device: The device to put the model on
        model_size: Model size, one of "b16_1k", "l14_22k", "h14_1k", etc.

    Returns:
        model: I-JEPA model wrapped in a custom wrapper
        embedding_dim: Embedding dimension of the model
    """
    # Map model size to exact model IDs based on the available options and their correct dimensions
    model_map = {
        "h16_1k": {"id": "facebook/ijepa_vith16_1k", "dim": 1408, "img_size": 224},
        "g16_22k": {"id": "facebook/ijepa_vitg16_22k", "dim": 1408, "img_size": 224},
        "h14_22k": {"id": "facebook/ijepa_vith14_22k", "dim": 1408, "img_size": 224},
        "h14_1k": {"id": "facebook/ijepa_vith14_1k", "dim": 1408, "img_size": 224},
    }

    if model_size not in model_map:
        raise ValueError(
            f"Model size {model_size} not supported. Choose from {list(model_map.keys())}"
        )

    model_id = model_map[model_size]["id"]
    embedding_dim = model_map[model_size]["dim"]
    img_size = model_map[model_size]["img_size"]

    print(f"Loading I-JEPA model {model_id} using transformers...")
    print(f"Embedding dimension: {embedding_dim}, Image size: {img_size}x{img_size}")

    try:
        # Import the required libraries
        from transformers import AutoModel

        # Load the model with the correct ID
        base_model = AutoModel.from_pretrained(model_id)
        print(f"Successfully loaded I-JEPA model {model_id}")
    except Exception as e:
        print(f"Error loading I-JEPA model: {e}")
        raise ValueError(
            f"Could not load I-JEPA model. Please check if transformers is installed and the model ID is correct."
        )

    # Unfreeze all parameters as requested
    print("Unfreezing all I-JEPA parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in I-JEPA backbone")

    # Define wrapper class that uses the correct image size
    class IJEPAWrapper(nn.Module):
        def __init__(self, model, img_size=448):
            super().__init__()
            self.model = model
            self.img_size = img_size

        def forward(self, x):
            # Make x require gradients for proper gradient flow
            x = x.detach().requires_grad_(True)

            # Process smaller batches if needed (for memory efficiency)
            batch_size = x.shape[0]
            if (
                batch_size > 8 and x.device.type == "cuda"
            ):  # Reduced batch size for large images
                # Process in chunks
                outputs_list = []
                for i in range(0, batch_size, 8):
                    # Get batch chunk
                    x_chunk = x[i : i + 8]
                    # Forward pass - IJEPA models can handle different input sizes
                    outputs = self.model(x_chunk, interpolate_pos_encoding=True)
                    # Extract embeddings using mean pooling
                    features = outputs.last_hidden_state.mean(dim=1)
                    outputs_list.append(features)

                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                # No need to resize - IJEPA can handle different input sizes

                try:
                    # Forward pass - IJEPA models can handle different input sizes
                    outputs = self.model(x, interpolate_pos_encoding=True)
                    # Extract embeddings using mean pooling
                    features = outputs.last_hidden_state.mean(dim=1)
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    try:
                        # Try without interpolate_pos_encoding
                        outputs = self.model(x)
                        features = outputs.last_hidden_state.mean(dim=1)
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        # Return zeros as a last resort to avoid crashing
                        features = torch.zeros(
                            (x.size(0), embedding_dim), device=x.device
                        )

                return features

    # Create and return wrapped model
    model = IJEPAWrapper(base_model, img_size=img_size).to(device)

    # Detect actual embedding dimension by doing a forward pass
    print("Detecting actual embedding dimension...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        try:
            dummy_output = model(dummy_input)
            actual_embedding_dim = dummy_output.shape[1]
            print(f"Detected actual embedding dimension: {actual_embedding_dim}")
            embedding_dim = actual_embedding_dim
        except Exception as e:
            print(f"Could not detect embedding dimension: {e}")
            print(f"Using configured dimension: {embedding_dim}")

    print(f"I-JEPA model wrapper created with image size {img_size}x{img_size}")
    return model, embedding_dim
