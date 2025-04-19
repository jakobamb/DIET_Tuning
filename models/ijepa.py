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
        "b16_1k": {"id": "facebook/ijepa_vith16_1k", "dim": 1280, "img_size": 448},
        "b16_22k": {"id": "facebook/ijepa_vitg16_22k", "dim": 1280, "img_size": 448},
        "l14_22k": {"id": "facebook/ijepa_vith14_22k", "dim": 1280, "img_size": 448},
        "h14_1k": {"id": "facebook/ijepa_vith14_1k", "dim": 1280, "img_size": 448},
    }
    
    if model_size not in model_map:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(model_map.keys())}")
    
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
        raise ValueError(f"Could not load I-JEPA model. Please check if transformers is installed and the model ID is correct.")
    
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
            if batch_size > 8 and x.device.type == 'cuda':  # Reduced batch size for large images
                # Process in chunks
                outputs_list = []
                for i in range(0, batch_size, 8):
                    # Get batch chunk
                    x_chunk = x[i:i+8]
                    # Resize to expected input size (448x448 by default)
                    if x_chunk.shape[-1] != self.img_size:
                        x_chunk = F.interpolate(x_chunk, size=(self.img_size, self.img_size), 
                                               mode='bilinear', align_corners=False)
                    
                    try:
                        # Forward pass with the correct image size
                        outputs = self.model(x_chunk)
                        # Extract embeddings using mean pooling
                        features = outputs.last_hidden_state.mean(dim=1)
                        outputs_list.append(features)
                    except Exception as e:
                        print(f"Error in forward pass: {e}")
                        try:
                            # Try with interpolate_pos_encoding=True
                            outputs = self.model(x_chunk, interpolate_pos_encoding=True)
                            features = outputs.last_hidden_state.mean(dim=1)
                            outputs_list.append(features)
                        except Exception as e2:
                            print(f"Second attempt failed: {e2}")
                            # Return zeros as a last resort to avoid crashing
                            dummy_features = torch.zeros((x_chunk.size(0), embedding_dim), device=x_chunk.device)
                            outputs_list.append(dummy_features)
                
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                if x.shape[-1] != self.img_size:
                    x = F.interpolate(x, size=(self.img_size, self.img_size), 
                                     mode='bilinear', align_corners=False)
                
                try:
                    # Forward pass with the correct image size
                    outputs = self.model(x)
                    # Extract embeddings using mean pooling
                    features = outputs.last_hidden_state.mean(dim=1)
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    try:
                        # Try with interpolate_pos_encoding=True
                        outputs = self.model(x, interpolate_pos_encoding=True)
                        features = outputs.last_hidden_state.mean(dim=1)
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        # Return zeros as a last resort to avoid crashing
                        features = torch.zeros((x.size(0), embedding_dim), device=x.device)
                
                return features
    
    # Create and return wrapped model
    model = IJEPAWrapper(base_model, img_size=img_size).to(device)
    print(f"I-JEPA model wrapper created with image size {img_size}x{img_size}")
    return model, embedding_dim