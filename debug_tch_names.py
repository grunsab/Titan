#!/usr/bin/env python3
"""Debug script to understand tch variable names and create correct mapping."""

import torch
import subprocess
import re

# First, get the Rust variable names and shapes from our dump
rust_output = """
res_block_0.bias -> shape: [128]
res_block_0.bias__12 -> shape: [128]
res_block_0.bias__15 -> shape: [128]
res_block_0.bias__9 -> shape: [128]
res_block_0.running_mean -> shape: [128]
res_block_0.running_mean__16 -> shape: [128]
res_block_0.running_var -> shape: [128]
res_block_0.running_var__17 -> shape: [128]
res_block_0.weight -> shape: [128, 128, 3, 3]
res_block_0.weight__13 -> shape: [128, 128, 3, 3]
res_block_0.weight__14 -> shape: [128]
res_block_0.weight__8 -> shape: [128]
"""

# Parse the Rust output to understand the pattern
rust_vars = {}
for line in rust_output.strip().split('\n'):
    if ' -> shape: ' in line:
        parts = line.split(' -> shape: ')
        name = parts[0].strip()
        shape = parts[1].strip()
        rust_vars[name] = shape

# Now let's understand the pattern
print("Pattern analysis for res_block_0:")
for name, shape in sorted(rust_vars.items()):
    print(f"{name} : {shape}")

# Load Python model to see what we're mapping from
state_dict = torch.load('weights/AlphaZeroNet_10x128.pt', map_location='cpu')

print("\nPython residualBlocks.0 keys:")
for key in sorted([k for k in state_dict.keys() if k.startswith('residualBlocks.0.')]):
    print(f"{key} : {list(state_dict[key].shape)}")

# Based on shape matching, let's create the mapping
print("\nMapping based on shapes:")
print("res_block_0.weight (conv1.weight) -> [128, 128, 3, 3]")
print("res_block_0.bias (conv1.bias) -> [128]")
print("res_block_0.weight__8 (bn2.weight) -> [128]")
print("res_block_0.bias__9 (bn2.bias) -> [128]")
print("res_block_0.bias__12 (bn1.bias) -> [128]")
print("res_block_0.weight__13 (conv2.weight) -> [128, 128, 3, 3]")
print("res_block_0.weight__14 (bn1.weight) -> [128]")
print("res_block_0.bias__15 (conv2.bias) -> [128]")

# Create the correct mapping
def create_correct_mapping(num_blocks):
    mapping = {}
    
    # Conv block
    mapping['conv_block.weight'] = 'convBlock1.conv1.weight'
    mapping['conv_block.bias'] = 'convBlock1.conv1.bias'
    mapping['conv_block.weight__2'] = 'convBlock1.bn1.weight'
    mapping['conv_block.bias__3'] = 'convBlock1.bn1.bias'
    mapping['conv_block.running_mean'] = 'convBlock1.bn1.running_mean'
    mapping['conv_block.running_var'] = 'convBlock1.bn1.running_var'
    
    # Residual blocks
    for i in range(num_blocks):
        prefix = f'res_block_{i}'
        py_prefix = f'residualBlocks.{i}'
        
        # The pattern seems to be consistent but with different suffixes per block
        if i == 0:
            mapping[f'{prefix}.weight'] = f'{py_prefix}.conv1.weight'
            mapping[f'{prefix}.bias'] = f'{py_prefix}.conv1.bias'
            mapping[f'{prefix}.weight__8'] = f'{py_prefix}.bn2.weight'
            mapping[f'{prefix}.bias__9'] = f'{py_prefix}.bn2.bias'
            mapping[f'{prefix}.bias__12'] = f'{py_prefix}.bn1.bias'
            mapping[f'{prefix}.weight__13'] = f'{py_prefix}.conv2.weight'
            mapping[f'{prefix}.weight__14'] = f'{py_prefix}.bn1.weight'
            mapping[f'{prefix}.bias__15'] = f'{py_prefix}.conv2.bias'
            mapping[f'{prefix}.running_mean'] = f'{py_prefix}.bn1.running_mean'
            mapping[f'{prefix}.running_var'] = f'{py_prefix}.bn1.running_var'
            mapping[f'{prefix}.running_mean__16'] = f'{py_prefix}.bn2.running_mean'
            mapping[f'{prefix}.running_var__17'] = f'{py_prefix}.bn2.running_var'
        else:
            # For other blocks, the suffix pattern changes
            # Let's analyze block 1 from the earlier output
            base = 20 + (i - 1) * 12
            mapping[f'{prefix}.weight'] = f'{py_prefix}.conv1.weight'
            mapping[f'{prefix}.bias'] = f'{py_prefix}.conv1.bias'
            mapping[f'{prefix}.weight__{base}'] = f'{py_prefix}.bn2.weight'
            mapping[f'{prefix}.bias__{base + 1}'] = f'{py_prefix}.conv1.bias'  # This seems wrong - duplicate
            mapping[f'{prefix}.weight__{base + 2}'] = f'{py_prefix}.bn2.weight'  # This also seems wrong
            mapping[f'{prefix}.bias__{base + 3}'] = f'{py_prefix}.bn2.bias'
            mapping[f'{prefix}.bias__{base + 4}'] = f'{py_prefix}.bn1.bias'
            mapping[f'{prefix}.weight__{base + 5}'] = f'{py_prefix}.bn1.weight'
            mapping[f'{prefix}.weight__{base + 6}'] = f'{py_prefix}.conv2.weight'
            mapping[f'{prefix}.bias__{base + 7}'] = f'{py_prefix}.conv2.bias'
            mapping[f'{prefix}.running_mean'] = f'{py_prefix}.bn1.running_mean'
            mapping[f'{prefix}.running_var'] = f'{py_prefix}.bn1.running_var'
            mapping[f'{prefix}.running_mean__{base + 8}'] = f'{py_prefix}.bn2.running_mean'
            mapping[f'{prefix}.running_var__{base + 9}'] = f'{py_prefix}.bn2.running_var'
    
    return mapping

# Create inverse mapping
mapping = create_correct_mapping(10)
inv_mapping = {v: k for k, v in mapping.items()}

print("\nInverse mapping for conversion:")
for py_name in sorted([k for k in state_dict.keys() if k.startswith('residualBlocks.0.')])[:12]:
    if py_name in inv_mapping:
        print(f"'{py_name}': '{inv_mapping[py_name]}',")