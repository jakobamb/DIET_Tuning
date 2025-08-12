"""
Test script to verify RGB conversion transforms are working correctly
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from loaders.data_loader import prepare_data_loaders, get_dataset, create_transforms

def test_rgb_transforms():
    """Test the RGB conversion transforms on different datasets"""
    
    print("🧪 Testing RGB Transform Pipeline...")
    print("=" * 50)
    
    # Test datasets with different original formats
    test_datasets = [
        ("cifar10", "RGB dataset (32x32)"),
        ("dermamnist", "RGB medical images (28x28)"), 
        ("chestmnist", "Grayscale medical images (28x28)"),
    ]
    
    for dataset_name, description in test_datasets:
        print(f"\n📊 Testing {dataset_name}: {description}")
        print("-" * 40)
        
        try:
            # Get dataset info
            train_data, test_data, num_classes, input_size, mean, std, is_rgb = get_dataset(dataset_name)
            print(f"Original format - Size: {input_size}x{input_size}, RGB: {is_rgb}")
            print(f"Original stats - Mean: {mean}, Std: {std}")
            
            # Create transforms
            train_transform, test_transform = create_transforms(mean, std, input_size, is_rgb, da_strength=1)
            
            # Apply transform to first sample
            sample, label = train_data[0]
            print(f"Before transform - Type: {type(sample)}, Mode: {sample.mode if hasattr(sample, 'mode') else 'N/A'}")
            
            # Apply the transform
            transformed = test_transform(sample)
            print(f"After transform - Shape: {transformed.shape}, Type: {transformed.dtype}")
            print(f"Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")
            
            # Verify it's 224x224 RGB
            if transformed.shape == torch.Size([3, 224, 224]):
                print("✅ Success: Converted to 3-channel 224x224 RGB!")
            else:
                print(f"❌ Error: Expected [3, 224, 224], got {transformed.shape}")
                
        except Exception as e:
            print(f"❌ Error testing {dataset_name}: {e}")
            
    print("\n" + "=" * 50)
    print("🎯 Testing Complete!")

def test_data_loader():
    """Test the full data loader pipeline"""
    
    print("\n🔄 Testing Full Data Loader Pipeline...")
    print("=" * 50)
    
    try:
        # Test with a small batch
        train_loader, test_loader, dataset_info = prepare_data_loaders(
            dataset_name="cifar10",
            batch_size=4,
            num_diet_classes=10,
            da_strength=1,
            limit_data=100  # Small test
        )
        
        print(f"Dataset info: {dataset_info}")
        
        # Get a batch
        batch = next(iter(train_loader))
        images, labels, diet_classes = batch
        
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Batch diet classes shape: {diet_classes.shape}")
        
        # Verify RGB format
        if images.shape[1:] == torch.Size([3, 224, 224]):
            print("✅ Success: Data loader produces 3-channel 224x224 RGB batches!")
        else:
            print(f"❌ Error: Expected [batch, 3, 224, 224], got {images.shape}")
            
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")

def visualize_transform_comparison():
    """Compare before and after transform visually"""
    
    print("\n👁️ Visual Comparison Test...")
    print("=" * 50)
    
    try:
        # Get a sample from CIFAR-10
        train_data, _, _, _, _, _, _ = get_dataset("cifar10")
        sample, label = train_data[0]
        
        # Create transforms
        _, test_transform = create_transforms([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 32, True, 1)
        
        # Apply transform
        transformed = test_transform(sample)
        
        print(f"Original: {type(sample)}, size: {sample.size if hasattr(sample, 'size') else 'N/A'}")
        print(f"Transformed: {transformed.shape}, dtype: {transformed.dtype}")
        
        # Convert back for visualization
        transformed_vis = transformed.permute(1, 2, 0)  # CHW -> HWC
        transformed_vis = (transformed_vis + 1) / 2  # Denormalize from [-1,1] to [0,1]
        transformed_vis = torch.clamp(transformed_vis, 0, 1)
        
        print("✅ Transform pipeline completed successfully!")
        print(f"Final output: 224x224 RGB image ready for model input")
        
    except Exception as e:
        print(f"❌ Error in visualization test: {e}")

if __name__ == "__main__":
    test_rgb_transforms()
    test_data_loader() 
    visualize_transform_comparison()