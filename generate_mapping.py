#!/usr/bin/env python3
"""Generate exact mapping based on tch variable names."""

import subprocess
import re
import json

# Run the dump_var_names binary and capture output
result = subprocess.run(
    ["cargo", "run", "--release", "--bin", "dump_var_names"],
    env={"DYLD_LIBRARY_PATH": "/Users/rishisachdev/libtorch/lib"},
    capture_output=True,
    text=True,
    cwd="/Users/rishisachdev/Documents/GitHub/rust_piebot/piebot"
)

# Parse the output to extract variable names and shapes
lines = result.stdout.strip().split('\n')
model_10x128 = {}
model_20x256 = {}
current_model = None

for line in lines:
    if "Variables for 10x128 model" in line:
        current_model = model_10x128
    elif "Variables for 20x256 model" in line:
        current_model = model_20x256
    elif " -> shape: " in line and current_model is not None:
        parts = line.split(" -> shape: ")
        if len(parts) == 2:
            var_name = parts[0].strip()
            shape_str = parts[1].strip()
            current_model[var_name] = shape_str

print(f"Found {len(model_10x128)} variables for 10x128 model")
print(f"Found {len(model_20x256)} variables for 20x256 model")

# Now create the mapping based on shapes
def create_mapping(tch_vars, num_blocks):
    mapping = {}
    
    # Conv block
    for name, shape in tch_vars.items():
        if name.startswith('conv_block.'):
            if shape == '[128, 16, 3, 3]' or shape == '[256, 16, 3, 3]':
                mapping[name] = 'convBlock1.conv1.weight'
            elif name == 'conv_block.bias' and (shape == '[128]' or shape == '[256]'):
                mapping[name] = 'convBlock1.conv1.bias'
            elif name == 'conv_block.weight__2':
                mapping[name] = 'convBlock1.bn1.weight'
            elif name == 'conv_block.bias__3':
                mapping[name] = 'convBlock1.bn1.bias'
            elif name == 'conv_block.running_mean':
                mapping[name] = 'convBlock1.bn1.running_mean'
            elif name == 'conv_block.running_var':
                mapping[name] = 'convBlock1.bn1.running_var'
    
    # Residual blocks
    for i in range(num_blocks):
        prefix = f'res_block_{i}'
        
        # Find all variables for this block
        block_vars = {k: v for k, v in tch_vars.items() if k.startswith(prefix + '.')}
        
        # Map based on shapes and patterns
        for name, shape in block_vars.items():
            suffix = name[len(prefix)+1:]  # Everything after "res_block_i."
            
            # Conv weights (3x3)
            if '3, 3]' in shape:
                if name == f'{prefix}.weight' or suffix.startswith('weight') and suffix.split('__')[0] == 'weight':
                    mapping[name] = f'residualBlocks.{i}.conv1.weight'
                else:
                    mapping[name] = f'residualBlocks.{i}.conv2.weight'
            
            # Biases and BN weights (1D tensors)
            elif shape == '[128]' or shape == '[256]':
                if name == f'{prefix}.bias':
                    mapping[name] = f'residualBlocks.{i}.conv1.bias'
                elif suffix.startswith('bias__'):
                    # Need to determine which layer based on suffix number
                    parts = suffix.split('__')
                    if len(parts) == 2:
                        num = int(parts[1])
                        # This is heuristic based on observed patterns
                        if i == 0:
                            if num == 9:
                                mapping[name] = f'residualBlocks.{i}.bn2.bias'
                            elif num == 12:
                                mapping[name] = f'residualBlocks.{i}.bn1.bias'
                            elif num == 15:
                                mapping[name] = f'residualBlocks.{i}.conv2.bias'
                        else:
                            # For other blocks, use relative position
                            block_base = 20 + (i - 1) * 12
                            if num == block_base + 1:
                                mapping[name] = f'residualBlocks.{i}.conv1.bias'
                            elif num == block_base + 3:
                                mapping[name] = f'residualBlocks.{i}.bn2.bias'
                            elif num == block_base + 4:
                                mapping[name] = f'residualBlocks.{i}.bn1.bias'
                            elif num == block_base + 7:
                                mapping[name] = f'residualBlocks.{i}.conv2.bias'
                elif suffix.startswith('weight__'):
                    parts = suffix.split('__')
                    if len(parts) == 2:
                        num = int(parts[1])
                        if i == 0:
                            if num == 8:
                                mapping[name] = f'residualBlocks.{i}.bn2.weight'
                            elif num == 14:
                                mapping[name] = f'residualBlocks.{i}.bn1.weight'
                        else:
                            block_base = 20 + (i - 1) * 12
                            if num == block_base + 2:
                                mapping[name] = f'residualBlocks.{i}.bn2.weight'
                            elif num == block_base + 5:
                                mapping[name] = f'residualBlocks.{i}.bn1.weight'
            
            # Running stats
            elif 'running_mean' in name:
                if suffix == 'running_mean':
                    mapping[name] = f'residualBlocks.{i}.bn1.running_mean'
                else:
                    mapping[name] = f'residualBlocks.{i}.bn2.running_mean'
            elif 'running_var' in name:
                if suffix == 'running_var':
                    mapping[name] = f'residualBlocks.{i}.bn1.running_var'
                else:
                    mapping[name] = f'residualBlocks.{i}.bn2.running_var'
    
    # Value head
    for name, shape in tch_vars.items():
        if name.startswith('value_head.'):
            if shape == '[1, 128, 1, 1]' or shape == '[1, 256, 1, 1]':
                mapping[name] = 'valueHead.conv1.weight'
            elif name == 'value_head.bias' and shape == '[1]':
                mapping[name] = 'valueHead.conv1.bias'
            elif 'weight__' in name:
                if shape == '[1]':
                    mapping[name] = 'valueHead.bn1.weight'
                elif shape == '[256, 64]':
                    mapping[name] = 'valueHead.fc1.weight'
                elif shape == '[1, 256]':
                    mapping[name] = 'valueHead.fc2.weight'
            elif 'bias__' in name and shape != '[1]':
                if shape == '[256]':
                    mapping[name] = 'valueHead.fc1.bias'
                else:
                    parts = name.split('__')
                    if len(parts) == 2:
                        num = int(parts[1])
                        # Heuristic based on observed patterns
                        if num in [129, 249]:  # These seem to be bn1.bias
                            mapping[name] = 'valueHead.bn1.bias'
                        elif num in [134, 254]:  # These seem to be fc2.bias
                            mapping[name] = 'valueHead.fc2.bias'
                        elif num in [132, 252]:  # These seem to be fc1.bias
                            mapping[name] = 'valueHead.fc1.bias'
            elif name == 'value_head.running_mean':
                mapping[name] = 'valueHead.bn1.running_mean'
            elif name == 'value_head.running_var':
                mapping[name] = 'valueHead.bn1.running_var'
    
    # Policy head
    for name, shape in tch_vars.items():
        if name.startswith('policy_head.'):
            if shape == '[2, 128, 1, 1]' or shape == '[2, 256, 1, 1]':
                mapping[name] = 'policyHead.conv1.weight'
            elif name == 'policy_head.bias' and shape == '[2]':
                mapping[name] = 'policyHead.conv1.bias'
            elif 'weight__' in name:
                if shape == '[2]':
                    mapping[name] = 'policyHead.bn1.weight'
                elif shape == '[4608, 128]' or shape == '[4608, 256]':
                    mapping[name] = 'policyHead.fc1.weight'
            elif 'bias__' in name:
                if shape == '[4608]':
                    mapping[name] = 'policyHead.fc1.bias'
                elif shape == '[2]':
                    mapping[name] = 'policyHead.bn1.bias'
            elif name == 'policy_head.running_mean':
                mapping[name] = 'policyHead.bn1.running_mean'
            elif name == 'policy_head.running_var':
                mapping[name] = 'policyHead.bn1.running_var'
    
    return mapping

# Create mappings
mapping_10x128 = create_mapping(model_10x128, 10)
mapping_20x256 = create_mapping(model_20x256, 20)

# Save mappings
with open('mapping_10x128.json', 'w') as f:
    json.dump({'tch_to_python': mapping_10x128, 'tch_vars': model_10x128}, f, indent=2)

with open('mapping_20x256.json', 'w') as f:
    json.dump({'tch_to_python': mapping_20x256, 'tch_vars': model_20x256}, f, indent=2)

print(f"\nCreated {len(mapping_10x128)} mappings for 10x128 model")
print(f"Created {len(mapping_20x256)} mappings for 20x256 model")

# Print inverse mapping for conversion
print("\nInverse mapping for conversion:")
inv_10x128 = {v: k for k, v in mapping_10x128.items()}
for py_name in sorted(inv_10x128.keys())[:10]:
    print(f"  '{py_name}': '{inv_10x128[py_name]}',")