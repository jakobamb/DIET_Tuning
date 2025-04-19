"""DINOv2 model implementation for DIET finetuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

def get_dinov2_model(device, model_size="small"):
    """Create DINOv2 model with memory optimization"""
    print(f"Loading DINOv2-{model_size} model...")
    
    # Model size to embedding dimension mapping
    dim_map = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536
    }
    
    # Load model and processor
    model_name = f"facebook/dinov2-{model_size}"
    processor = AutoImageProcessor.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    # UNFREEZE ALL PARAMETERS
    print("Unfreezing all DINOv2 parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in DINOv2 backbone")
    
    class DINOv2Wrapper(nn.Module):
        def __init__(self, model):
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
                    # Process chunk WITHOUT autocast and with gradient tracking
                    chunk_output = self.model(x_chunk)
                    outputs_list.append(chunk_output.last_hidden_state[:, 0])
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                if x.shape[-1] != 224:
                    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                # Process WITHOUT autocast and with gradient tracking
                outputs = self.model(x)
                return outputs.last_hidden_state[:, 0]
    
    model = DINOv2Wrapper(base_model).to(device)
    embedding_dim = dim_map[model_size]
    
    print(f"DINOv2-{model_size} loaded. Embedding dimension: {embedding_dim}")
    return model, embedding_dim

def apply_strategic_freezing(model):
    """Apply strategic freezing to DINOv2 model - only freeze embeddings"""
    print("Applying minimal freezing to DINOv2 (allowing more gradients)...")
    frozen_params = 0
    total_params = 0
    
    for name, param in model.model.named_parameters():
        total_params += 1
        # Only freeze embeddings, unfreeze all transformer layers
        if 'embeddings' in name:  # Only freeze embeddings, not encoder layers
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