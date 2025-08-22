"""ResNet-50 model implementation for DIET finetuning using timm."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def get_resnet50_model(device, model_size="a1_in1k"):
    """Create ResNet-50 model with timm
    
    Args:
        device: Device to load model on
        model_size: ResNet-50 variant options:
                   - "a1_in1k": Improved training recipe (standard)
                   - "tv2_in1k": TorchVision v2 recipe
                   - "gluon_in1k": GluonCV implementation
                   - "clip.cc12m": CLIP-trained ResNet-50 on CC12M dataset
                   - "clip.openai": Original OpenAI CLIP ResNet-50
                   - "fb_swsl_ig1b_ft_in1k": Facebook semi-supervised
    """
    print(f"Loading ResNet-50 {model_size} model from timm...")
    
    # ResNet-50 has a fixed embedding dimension of 2048 (before final classifier)
    embedding_dim = 2048
    
    # Handle special naming for CLIP models
    if model_size == "clip.cc12m":
        model_name = "resnet50_clip.cc12m"
        print("🔄 Loading CLIP-trained ResNet-50 (trained on image-text pairs)")
        print("Note: This model was trained with contrastive learning on CC12M dataset")
    elif model_size == "clip.openai":
        model_name = "resnet50_clip.openai"
        print("🔄 Loading OpenAI CLIP ResNet-50 (original CLIP model)")
        print("Note: This is the original OpenAI CLIP model trained on 400M image-text pairs")
    else:
        model_name = f"resnet50.{model_size}"
    
    try:
        # Load pretrained ResNet-50 model
        base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head to get feature embeddings
        )
        
        print(f"✅ Successfully loaded {model_name} from timm")
        
        # Get model info safely
        try:
            # For CLIP models, show additional info
            if "clip" in model_name:
                print("📝 CLIP Model Info:")
                if "openai" in model_name:
                    print("   - Original OpenAI CLIP model")
                    print("   - Trained on 400M image-text pairs from the web")
                    print("   - State-of-the-art zero-shot classification")
                    print("   - Best performance for cross-modal tasks")
                elif "cc12m" in model_name:
                    print("   - Trained on 12M image-text pairs (CC12M dataset)")
                    print("   - Uses QuickGELU activation instead of ReLU")
                    print("   - Optimized for cross-modal understanding")
                print("   - Can be used for zero-shot classification")
            
            model_stats = timm.model_info(model_name)
            print(f"Model parameters: {model_stats.get('params', 'N/A')}")
        except:
            print("Model loaded successfully (stats unavailable)")
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print("🔄 Falling back to standard resnet50...")
        # Fallback to basic ResNet-50 if specific variant not available
        base_model = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=0,  # Remove classifier head
        )
        print("✅ Loaded standard ResNet-50 as fallback")
    
    # Get model-specific data configuration
    data_config = timm.data.resolve_model_data_config(base_model)
    print(f"Model expects input size: {data_config['input_size']}")
    print(f"Model normalization: mean={data_config['mean']}, std={data_config['std']}")
    
    # UNFREEZE ALL PARAMETERS for full fine-tuning
    print("Unfreezing all ResNet-50 parameters...")
    unfrozen_params = 0
    for param in base_model.parameters():
        param.requires_grad = True
        unfrozen_params += 1
    print(f"Unfrozen {unfrozen_params} parameters in ResNet-50 backbone")
    
    class ResNet50Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.data_config = data_config
            
        def forward(self, x):
            # Make x require gradients to force gradient flow
            x = x.detach().requires_grad_(True)
            
            # ResNet-50 expects 224x224 input
            expected_size = self.data_config['input_size'][-1]  # Should be 224
            if x.shape[-1] != expected_size:
                x = F.interpolate(x, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            
            # Process batches efficiently
            batch_size = x.shape[0]
            if batch_size > 32 and x.device.type == 'cuda':
                # Process in chunks to save memory for large batches
                outputs_list = []
                for i in range(0, batch_size, 32):
                    x_chunk = x[i:i+32]
                    # Forward through ResNet-50 to get features (before classifier)
                    chunk_output = self.model.forward_features(x_chunk)
                    # Global average pooling to get final features
                    chunk_output = self.model.forward_head(chunk_output, pre_logits=True)
                    outputs_list.append(chunk_output)
                return torch.cat(outputs_list, dim=0)
            else:
                # Standard processing for smaller batches
                # Forward through ResNet-50 to get features
                features = self.model.forward_features(x)
                # Global average pooling to get final embeddings
                embeddings = self.model.forward_head(features, pre_logits=True)
                return embeddings
    
    model = ResNet50Wrapper(base_model).to(device)
    
    print(f"ResNet-50 {model_size} loaded. Embedding dimension: {embedding_dim}")
    print(f"Model architecture: ResNet-B with ReLU activations")
    print(f"Features: 7x7 conv, 1x1 shortcut downsample, trained on ImageNet-1k")
    
    return model, embedding_dim

def apply_strategic_freezing(model):
    """Apply strategic freezing to ResNet-50 model
    
    Options:
    1. Freeze early layers (conv1, bn1, layer1)
    2. Freeze backbone and only train later layers
    3. Minimal freezing (only first conv layer)
    """
    print("Applying strategic freezing to ResNet-50...")
    frozen_params = 0
    total_params = 0
    
    for name, param in model.model.named_parameters():
        total_params += 1
        
        # Strategy: Freeze early convolutional layers but keep later layers trainable
        if any(layer in name for layer in ['conv1', 'bn1', 'layer1']):
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True  # Keep layer2, layer3, layer4 trainable
    
    print(f"Frozen {frozen_params} out of {total_params} parameters")
    print("Frozen: conv1, bn1, layer1 (early features)")
    print("Trainable: layer2, layer3, layer4 (high-level features)")
    
    # Diagnostic output
    print("\nTrainable layers in ResNet-50:")
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}")
        elif frozen_params < 10:  # Only show first few frozen layers
            print(f"  ✗ {name}")
    
    return model

def get_resnet50_transforms():
    """Get the standard ResNet-50 transforms that match the training recipe"""
    # These are the standard ImageNet transforms used for ResNet-50
    return {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'input_size': 224,
        'is_rgb': True
    }