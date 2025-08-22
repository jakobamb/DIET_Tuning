"""MambaVision model implementation for DIET finetuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mambavision_model(device, model_variant="T"):
    """Create MambaVision model using direct feature extraction approach
    
    Args:
        device: The device to put the model on
        model_variant: Model variant (T, T2, S, B, L, L2, etc.) or full name
        
    Returns:
        model: Wrapped MambaVision model for feature extraction
        embedding_dim: Embedding dimension of the model
    """
    # Map model variants to their configurations based on the documentation
    model_configs = {
        # ImageNet-1K models
        "T": {"id": "nvidia/MambaVision-T-1K", "dim": 512, "res": 224, "params": 31.8},
        "T2": {"id": "nvidia/MambaVision-T2-1K", "dim": 512, "res": 224, "params": 35.1},
        "S": {"id": "nvidia/MambaVision-S-1K", "dim": 768, "res": 224, "params": 50.1},
        "B": {"id": "nvidia/MambaVision-B-1K", "dim": 1024, "res": 224, "params": 97.7},
        "L": {"id": "nvidia/MambaVision-L-1K", "dim": 1280, "res": 224, "params": 227.9},
        "L2": {"id": "nvidia/MambaVision-L2-1K", "dim": 1408, "res": 224, "params": 241.5},
        
        # ImageNet-21K models
        "B-21K": {"id": "nvidia/MambaVision-B-21K", "dim": 1024, "res": 224, "params": 97.7},
        "L-21K": {"id": "nvidia/MambaVision-L-21K", "dim": 1280, "res": 224, "params": 227.9},
        "L2-512-21K": {"id": "nvidia/MambaVision-L2-512-21K", "dim": 1408, "res": 512, "params": 241.5},
        "L3-256-21K": {"id": "nvidia/MambaVision-L3-256-21K", "dim": 1568, "res": 256, "params": 739.6},
        "L3-512-21K": {"id": "nvidia/MambaVision-L3-512-21K", "dim": 1568, "res": 512, "params": 739.6},
    }
    
    # Handle full model names too (e.g., "MambaVision-T" or just "T")
    if model_variant.startswith("MambaVision-"):
        model_variant = model_variant[12:]  # Remove "MambaVision-" prefix
    
    if model_variant not in model_configs:
        raise ValueError(f"Model variant {model_variant} not supported. Choose from {list(model_configs.keys())}")
    
    config = model_configs[model_variant]
    model_id = config["id"]
    embedding_dim = config["dim"]
    input_res = config["res"]
    
    print(f"Loading MambaVision-{model_variant} model...")
    print(f"Model ID: {model_id}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Input resolution: {input_res}x{input_res}")
    print(f"Parameter count: {config['params']} million")
    
    try:
        from transformers import AutoModel
        
        # Use AutoModel for feature extraction
        base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        print(f"Successfully loaded MambaVision model")
        
        # Unfreeze all parameters
        print("Unfreezing all MambaVision parameters...")
        unfrozen_params = 0
        for param in base_model.parameters():
            param.requires_grad = True
            unfrozen_params += 1
        print(f"Unfrozen {unfrozen_params} parameters in MambaVision backbone")
        
        # Create a wrapper class for feature extraction
        class MambaVisionWrapper(nn.Module):
            def __init__(self, model, input_res, emb_dim):
                super().__init__()
                self.model = model
                self.input_res = input_res
                self.emb_dim = emb_dim
                
            def forward(self, x):
                # Make x require gradients to force gradient flow
                x = x.detach().requires_grad_(True)
                
                # Process in small batches to save memory
                batch_size = x.shape[0]
                if batch_size > 4 and x.device.type == 'cuda':  # Use very small batch size for large models
                    outputs_list = []
                    for i in range(0, batch_size, 4):
                        # Get batch chunk
                        x_chunk = x[i:i+4]
                        
                        # Resize to expected input size
                        if x_chunk.shape[-1] != self.input_res:
                            x_chunk = F.interpolate(x_chunk, size=(self.input_res, self.input_res), 
                                                   mode='bilinear', align_corners=False)
                        
                        # Extract features from model
                        try:
                            # MambaVision AutoModel returns (avg_pool, features)
                            avg_pool, _ = self.model(x_chunk)
                            outputs_list.append(avg_pool)
                        except Exception as e:
                            print(f"Error in forward pass: {e}")
                            # Return zeros if there's an error
                            dummy = torch.zeros((x_chunk.size(0), self.emb_dim), device=x_chunk.device)
                            outputs_list.append(dummy)
                    
                    return torch.cat(outputs_list, dim=0)
                else:
                    # Process as a single batch
                    if x.shape[-1] != self.input_res:
                        x = F.interpolate(x, size=(self.input_res, self.input_res), 
                                         mode='bilinear', align_corners=False)
                    
                    try:
                        # MambaVision AutoModel returns (avg_pool, features)
                        avg_pool, _ = self.model(x)
                        return avg_pool
                    except Exception as e:
                        print(f"Error in forward pass: {e}")
                        return torch.zeros((x.size(0), self.emb_dim), device=x.device)
        
        # Create and return wrapped model
        model = MambaVisionWrapper(base_model, input_res, embedding_dim).to(device)
        return model, embedding_dim
    
    except Exception as e:
        print(f"Error setting up MambaVision: {e}")
        raise ValueError(f"Failed to load MambaVision. Consider using DINOv2 or MAE instead. Error: {e}")