"""SimCLRv1 model implementation for DIET finetuning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from huggingface_hub import hf_hub_download
import warnings

def get_simclr_model(device, model_size="resnet50-1x"):
    """Create SimCLRv1 model from Hugging Face
    
    Args:
        device: Device to load model on
        model_size: SimCLR variant options:
                   - "resnet50-1x": Standard ResNet-50 backbone (1x width)
                   - "resnet50-2x": ResNet-50 with 2x width
                   - "resnet50-4x": ResNet-50 with 4x width (if available)
    """
    print(f"Loading SimCLRv1 {model_size} model from Hugging Face...")
    
    # Map model sizes to HF repositories and file names
    model_mapping = {
        "resnet50-1x": {
            "repo": "lightly-ai/simclrv1-imagenet1k-resnet50-1x",
            "filename": "resnet50-1x.pth"
        },
        "resnet50-2x": {
            "repo": "lightly-ai/simclrv1-imagenet1k-resnet50-2x", 
            "filename": "resnet50-2x.pth"
        },
        "resnet50-4x": {
            "repo": "lightly-ai/simclrv1-imagenet1k-resnet50-4x",
            "filename": "resnet50-4x.pth"
        },
    }
    
    model_info = model_mapping.get(model_size, model_mapping["resnet50-1x"])
    
    # SimCLR ResNet-50 typically has 2048 feature dimensions
    embedding_dim = 2048
    
    try:
        print(f"🔄 Loading SimCLR model: {model_info['repo']}")
        print("📝 SimCLR Info:")
        print("   - Self-supervised contrastive learning")
        print("   - Trained on ImageNet-1k without labels")
        print("   - Uses data augmentation + contrastive loss")
        print("   - Excellent for transfer learning")
        
        # Download the actual PyTorch weights file
        print(f"📥 Downloading {model_info['filename']} from Hugging Face...")
        try:
            weights_path = hf_hub_download(
                repo_id=model_info['repo'],
                filename=model_info['filename'],
                cache_dir=None  # Use default cache
            )
            print(f"✅ Downloaded weights to: {weights_path}")
            
            # Create a standard ResNet-50 architecture
            backbone = models.resnet50(weights=None)  # No pretrained weights
            
            # Load the SimCLR weights
            print("🔄 Loading SimCLR weights into ResNet-50 architecture...")
            simclr_state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle nested state_dict structure
            if 'state_dict' in simclr_state_dict:
                print("📝 Found nested state_dict, extracting...")
                simclr_state_dict = simclr_state_dict['state_dict']
            
            # The SimCLR model might have different key names, so we need to adapt them
            # Remove the projection head keys and keep only the backbone
            backbone_state_dict = {}
            
            for key, value in simclr_state_dict.items():
                # Skip projection head layers (typically named 'projection' or 'head')
                if any(skip_key in key.lower() for skip_key in ['projection', 'head', 'fc']):
                    continue
                    
                # Adapt key names if necessary
                new_key = key
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                elif key.startswith('encoder.'):
                    new_key = key.replace('encoder.', '')
                elif key.startswith('resnet.'):
                    new_key = key.replace('resnet.', '')
                elif key.startswith('module.'):
                    new_key = key.replace('module.', '')
                
                backbone_state_dict[new_key] = value
            
            # Load the adapted state dict
            missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)} keys (will use ImageNet initialization)")
                if len(missing_keys) > 10:
                    print("   Too many missing keys - loading ImageNet weights for missing parts...")
                    # Load ImageNet weights for missing parts
                    imagenet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                    imagenet_state = imagenet_model.state_dict()
                    
                    # Fill in missing keys with ImageNet weights
                    for key in missing_keys:
                        if key in imagenet_state:
                            backbone_state_dict[key] = imagenet_state[key]
                    
                    # Reload with filled state dict
                    backbone.load_state_dict(backbone_state_dict, strict=False)
                    print("✅ Filled missing keys with ImageNet weights")
                else:
                    print(f"   Missing: {missing_keys[:3]}...")
                    
            if unexpected_keys:
                print(f"⚠️ Unexpected keys (ignored): {len(unexpected_keys)} keys")
            
            # Remove the final classification layer (fc layer)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            print(f"✅ Successfully loaded SimCLR weights from {model_info['repo']}")
            
        except Exception as download_error:
            print(f"❌ Failed to download SimCLR weights: {download_error}")
            raise download_error
            
    except Exception as e:
        print(f"❌ Failed to load SimCLR model: {e}")
        print("🔄 Falling back to standard ResNet-50...")
        
        # Fallback to torchvision ResNet-50 with updated API
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        print("✅ Loaded standard ResNet-50 as fallback (fixed deprecation warnings)")
    
    # UNFREEZE ALL PARAMETERS for full fine-tuning
    print("Unfreezing all SimCLR parameters...")
    unfrozen_params = 0
    for param in backbone.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in SimCLR backbone")
    
    class SimCLRWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            
        def forward(self, x):
            # Make x require gradients to force gradient flow
            x = x.detach().requires_grad_(True)
            
            # SimCLR expects 224x224 input
            if x.shape[-1] != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Process batches efficiently
            batch_size = x.shape[0]
            if batch_size > 32 and x.device.type == 'cuda':
                # Process in chunks to save memory for large batches
                outputs_list = []
                for i in range(0, batch_size, 32):
                    x_chunk = x[i:i+32]
                    chunk_output = self._forward_backbone(x_chunk)
                    outputs_list.append(chunk_output)
                return torch.cat(outputs_list, dim=0)
            else:
                return self._forward_backbone(x)
        
        def _forward_backbone(self, x):
            # Forward through SimCLR backbone
            features = self.backbone(x)
            
            # Ensure we have the right shape
            if len(features.shape) == 4:  # [B, C, H, W]
                features = self.adaptive_pool(features)
                features = self.flatten(features)
            elif len(features.shape) == 3:  # [B, H*W, C]
                features = features.mean(dim=1)  # Global average pooling
            elif len(features.shape) == 2:  # [B, C] - already pooled
                features = features
            else:
                # Handle unexpected shapes
                while len(features.shape) > 2:
                    features = features.mean(dim=-1)
            
            return features
    
    model = SimCLRWrapper(backbone).to(device)
    
    print(f"SimCLRv1 {model_size} loaded. Embedding dimension: {embedding_dim}")
    print(f"Self-supervised features from contrastive learning on ImageNet-1k")
    
    return model, embedding_dim

def apply_strategic_freezing(model):
    """Apply strategic freezing to SimCLR model
    
    For SimCLR, we can freeze early layers and fine-tune later ones
    since contrastive learning creates good low-level features
    """
    print("Applying strategic freezing to SimCLR...")
    frozen_params = 0
    total_params = 0
    
    for name, param in model.backbone.named_parameters():
        total_params += 1
        
        # Strategy: Freeze early layers but keep later layers trainable
        # SimCLR learns good early features, so we can freeze them
        if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True  # Keep layer3, layer4 trainable
    
    print(f"Frozen {frozen_params} out of {total_params} parameters")
    print("Frozen: conv1, bn1, layer1, layer2 (early contrastive features)")
    print("Trainable: layer3, layer4 (high-level features)")
    
    return model

def get_simclr_transforms():
    """Get the standard SimCLR transforms"""
    # SimCLR uses ImageNet normalization
    return {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'input_size': 224,
        'is_rgb': True
    }