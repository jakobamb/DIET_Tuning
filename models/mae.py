"""MAE (Masked Autoencoder) model implementation for DIET finetuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ViTMAEForPreTraining

def get_mae_model(device, model_size="base"):
    """Create MAE model with memory optimization"""
    print(f"Loading MAE-{model_size} model...")
    
    # Model size to embedding dimension mapping
    dim_map = {
        "base": 768,      # ViT-Base dimension
        "large": 1024,    # ViT-Large dimension
        "huge": 1280      # ViT-Huge dimension
    }
    
    # Map model size to Hugging Face model ID
    model_map = {
        "base": "facebook/vit-mae-base",
        "large": "facebook/vit-mae-large",  # This may need verification
        "huge": "facebook/vit-mae-huge"     # This may need verification
    }
    
    model_id = model_map[model_size]
    
    try:
        # Import the required classes
        
        # Load processor and model
        processor = AutoImageProcessor.from_pretrained(model_id)
        base_model = ViTMAEForPreTraining.from_pretrained(model_id)
        print(f"Successfully loaded MAE model from {model_id}")
    except Exception as e:
        print(f"Error loading MAE model: {e}")
        raise ValueError(f"Could not load MAE model. Please check if transformers is installed.")
    
    # UNFREEZE ALL PARAMETERS - exactly like your DINOv2 function
    print("Unfreezing all MAE parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in MAE backbone")
    
    # Define wrapper with same structure as DINOv2Wrapper
    class MAEWrapper(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor = processor
            
        def forward(self, x):
            # Make x require gradients to force gradient flow
            x = x.detach().requires_grad_(True)
            
            # Process smaller batches if needed
            batch_size = x.shape[0]
            if batch_size > 16 and x.device.type == 'cuda':
                # Process in chunks to save memory
                outputs_list = []
                for i in range(0, batch_size, 16):
                    # Get batch chunk
                    x_chunk = x[i:i+16]
                    # Resize to expected input (224x224)
                    if x_chunk.shape[-1] != 224:
                        x_chunk = F.interpolate(x_chunk, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    # For feature extraction, we use the encoder part of MAE
                    features = self.model.vit(x_chunk).last_hidden_state[:, 0]  # CLS token
                    outputs_list.append(features)
                
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                if x.shape[-1] != 224:
                    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
                # For feature extraction, we use the encoder part of MAE
                features = self.model.vit(x).last_hidden_state[:, 0]  # CLS token
                return features
    
    model = MAEWrapper(base_model, processor).to(device)
    embedding_dim = dim_map[model_size]
    
    print(f"MAE-{model_size} loaded. Embedding dimension: {embedding_dim}")
    return model, embedding_dim