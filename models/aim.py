"""AIM (Apple Image Model) implementation for DIET finetuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_aim_model(device, model_size="600M"):
    """Create AIM model using the properly installed AIM package
    
    Args:
        device: The device to load the model on
        model_size: Size of the model - "600M", "1B", "3B", or "7B"
        
    Returns:
        model: The wrapped AIM model
        embedding_dim: The ACTUAL embedding dimension from the loaded model
    """
    print(f"Loading AIM-{model_size} model...")
    
    # These are just reference values, we'll detect the actual dimension
    dim_map = {
        "600M": 768,    # Reference value, will be overridden by actual dimension
        "1B": 1024,
        "3B": 1536,
        "7B": 2048
    }
    
    # Map model size to model ID
    model_map = {
        "600M": "apple/aim-600M",
        "1B": "apple/aim-1B",
        "3B": "apple/aim-3B",
        "7B": "apple/aim-7B"
    }
    
    model_id = model_map[model_size]
    
    try:
        # Use the proper AIM imports
        from aim.v1.torch.models import AIMForImageClassification
        from aim.v1.torch.data import val_transforms
        
        # Load the model and transforms
        print(f"Loading AIM model: {model_id}")
        base_model = AIMForImageClassification.from_pretrained(model_id)
        transform = val_transforms()
        print(f"Successfully loaded AIM model: {model_id}")
        
    except Exception as e:
        print(f"Error loading AIM model: {e}")
        raise ValueError(f"Failed to load AIM model: {e}")
    
    # UNFREEZE ALL PARAMETERS
    print("Unfreezing all AIM parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in AIM backbone")
    
    # Define wrapper
    class AIMWrapper(nn.Module):
        def __init__(self, model, transform):
            super().__init__()
            self.model = model
            self.transform = transform
            self._feature_dim_detected = False
            self._feature_dim = None
            
        def forward(self, x):
            # Make x require gradients to force gradient flow
            x = x.detach().requires_grad_(True)
            
            # Process smaller batches if needed
            batch_size = x.shape[0]
            if batch_size > 8 and x.device.type == 'cuda':
                # Process in chunks to save memory
                outputs_list = []
                chunk_size = 4 if model_size in ["3B", "7B"] else 8
                
                for i in range(0, batch_size, chunk_size):
                    # Get batch chunk
                    x_chunk = x[i:i+chunk_size]
                    
                    # Resize to expected input (224x224)
                    if x_chunk.shape[-1] != 224:
                        x_chunk = F.interpolate(x_chunk, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    # Extract features
                    features = self._extract_features(x_chunk)
                    outputs_list.append(features)
                
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                if x.shape[-1] != 224:
                    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
                # Extract features
                features = self._extract_features(x)
                return features
                
        def _extract_features(self, x):
            """Extract features from the AIM model"""
            # Get model output
            output = self.model(x)
            
            # Check output on first run only
            if not self._feature_dim_detected:
                print(f"AIM model output type: {type(output)}")
                if isinstance(output, tuple):
                    print(f"Output tuple length: {len(output)}")
                    print(f"Features shape: {output[1].shape}")
                    self._feature_dim = output[1].shape[1]
                else:
                    print(f"Output shape: {output.shape}")
                    self._feature_dim = output.shape[1]
                
                print(f"Detected feature dimension: {self._feature_dim}")
                self._feature_dim_detected = True
            
            # Extract features
            if isinstance(output, tuple) and len(output) >= 2:
                return output[1]  # Features
            else:
                # Fallback
                return output
            
        @property
        def feature_dim(self):
            """Get the detected feature dimension"""
            if self._feature_dim is None:
                # This will force feature dimension detection with a dummy input
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(next(self.parameters()).device)
                    _ = self._extract_features(dummy_input)
            return self._feature_dim
    
    model = AIMWrapper(base_model, transform).to(device)
    
    # Detect the actual embedding dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model(dummy_input)
    
    embedding_dim = model.feature_dim
    print(f"AIM-{model_size} loaded. Detected embedding dimension: {embedding_dim}")
    
    return model, embedding_dim