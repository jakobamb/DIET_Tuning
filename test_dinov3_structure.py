"""Test script to understand DINOv3 model structure."""

from transformers import AutoModel

# Load a DINOv3 model to inspect its structure
model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

print("DINOv3 Model Structure:")
print(f"Model type: {type(model)}")
print(f"Model attributes: {dir(model)}")

# Check for encoder-like structures
if hasattr(model, "encoder"):
    print(f"Has encoder: True")
    print(f"Encoder type: {type(model.encoder)}")
    if hasattr(model.encoder, "layer"):
        print(f"Encoder has layers: {len(model.encoder.layer)}")
else:
    print("Has encoder: False")

if hasattr(model, "blocks"):
    print(f"Has blocks: True")
    print(f"Blocks type: {type(model.blocks)}")
    print(f"Number of blocks: {len(model.blocks)}")
else:
    print("Has blocks: False")

if hasattr(model, "layers"):
    print(f"Has layers: True")
    print(f"Layers type: {type(model.layers)}")
    print(f"Number of layers: {len(model.layers)}")
else:
    print("Has layers: False")

# Check the layer attribute specifically
if hasattr(model, "layer"):
    print(f"Has layer: True")
    print(f"Layer type: {type(model.layer)}")
    print(f"Number of layers: {len(model.layer)}")
    print(f"First layer type: {type(model.layer[0])}")
else:
    print("Has layer: False")

# Print all top-level attributes that might contain transformer blocks
for attr_name in dir(model):
    if not attr_name.startswith("_"):
        attr = getattr(model, attr_name)
        if hasattr(attr, "__len__") and not callable(attr):
            try:
                print(f"{attr_name}: length {len(attr)}")
            except Exception as e:
                print(f"{attr_name}: has length but can't determine: {e}")
