#!/usr/bin/env python3
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python check_model.py <model_file>")
    sys.exit(1)

model_file = sys.argv[1]

try:
    # Try to load the model
    data = torch.load(model_file, map_location='cpu')
    
    print(f"Model file: {model_file}")
    print(f"Type of data: {type(data)}")
    
    if isinstance(data, dict):
        print("\nKeys in the model file:")
        for key in data.keys():
            print(f"  - {key}: {type(data[key])}")
            if isinstance(data[key], torch.Tensor):
                print(f"    Shape: {data[key].shape}")
    
    elif isinstance(data, torch.nn.Module):
        print(f"\nModel class: {data.__class__.__name__}")
        print("\nModel state dict keys:")
        for key in data.state_dict().keys():
            print(f"  - {key}")
    
    # Try to save in JIT format for C++
    print("\nAttempting to save in TorchScript format...")
    
    # If it's a state dict, we can't directly convert to TorchScript
    if isinstance(data, dict):
        print("This appears to be a state dict, not a model. Cannot convert to TorchScript.")
        print("You need the model architecture to create a TorchScript model.")
    else:
        # Try to script the model
        scripted = torch.jit.script(data)
        output_file = model_file.replace('.pt', '_scripted.pt')
        scripted.save(output_file)
        print(f"Saved TorchScript model to: {output_file}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()