#!/usr/bin/env python3
"""Test script for DINOv3 implementation"""

import torch
from models.dinov2 import get_dinov3_model


def test_dinov3_models():
    """Test DINOv3 model loading and inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test different DINOv3 model sizes
    test_models = ["s16", "b16", "l16"]

    for model_size in test_models:
        print(f"\n{'='*50}")
        print(f"Testing DINOv3-{model_size}")
        print(f"{'='*50}")

        try:
            # Load model
            model, embedding_dim = get_dinov3_model(device, model_size)
            print(f"✓ Model loaded successfully")
            print(f"✓ Embedding dimension: {embedding_dim}")

            # Test forward pass
            batch_size = 4
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            print(f"✓ Created dummy input: {dummy_input.shape}")

            with torch.no_grad():
                output = model(dummy_input)
                print(f"✓ Forward pass successful")
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Expected shape: ({batch_size}, {embedding_dim})")

                # Verify output shape
                assert output.shape == (
                    batch_size,
                    embedding_dim,
                ), f"Shape mismatch: {output.shape} vs ({batch_size}, {embedding_dim})"
                print(f"✓ Shape verification passed")

        except Exception as e:
            print(f"✗ Error testing DINOv3-{model_size}: {e}")
            continue

        print(f"✓ DINOv3-{model_size} test completed successfully")


if __name__ == "__main__":
    test_dinov3_models()
