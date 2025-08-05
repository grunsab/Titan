#!/usr/bin/env python3
"""Convert Python AlphaZero weights to Rust tch format - working version."""

import torch
import sys

def convert_to_tch_format(state_dict, num_blocks, num_filters):
    """Convert Python state dict to tch naming convention."""
    tch_dict = {}
    
    # Conv block
    tch_dict['conv_block.weight'] = state_dict['convBlock1.conv1.weight']
    tch_dict['conv_block.bias'] = state_dict['convBlock1.conv1.bias']
    tch_dict['conv_block.weight__2'] = state_dict['convBlock1.bn1.weight']
    tch_dict['conv_block.bias__3'] = state_dict['convBlock1.bn1.bias']
    tch_dict['conv_block.running_mean'] = state_dict['convBlock1.bn1.running_mean']
    tch_dict['conv_block.running_var'] = state_dict['convBlock1.bn1.running_var']
    
    # Residual blocks - based on the exact pattern observed
    for i in range(num_blocks):
        py_prefix = f'residualBlocks.{i}'
        tch_prefix = f'res_block_{i}'
        
        # Common for all blocks
        tch_dict[f'{tch_prefix}.weight'] = state_dict[f'{py_prefix}.conv1.weight']
        tch_dict[f'{tch_prefix}.bias'] = state_dict[f'{py_prefix}.conv1.bias']
        tch_dict[f'{tch_prefix}.running_mean'] = state_dict[f'{py_prefix}.bn1.running_mean']
        tch_dict[f'{tch_prefix}.running_var'] = state_dict[f'{py_prefix}.bn1.running_var']
        
        if i == 0:
            # First block has specific suffixes
            tch_dict[f'{tch_prefix}.weight__8'] = state_dict[f'{py_prefix}.bn2.weight']
            tch_dict[f'{tch_prefix}.bias__9'] = state_dict[f'{py_prefix}.bn2.bias']
            tch_dict[f'{tch_prefix}.bias__12'] = state_dict[f'{py_prefix}.bn1.bias']
            tch_dict[f'{tch_prefix}.weight__13'] = state_dict[f'{py_prefix}.conv2.weight']
            tch_dict[f'{tch_prefix}.weight__14'] = state_dict[f'{py_prefix}.bn1.weight']
            tch_dict[f'{tch_prefix}.bias__15'] = state_dict[f'{py_prefix}.conv2.bias']
            tch_dict[f'{tch_prefix}.running_mean__16'] = state_dict[f'{py_prefix}.bn2.running_mean']
            tch_dict[f'{tch_prefix}.running_var__17'] = state_dict[f'{py_prefix}.bn2.running_var']
        else:
            # Other blocks follow a pattern
            # Block 1: 20, 21, 24, 25, 26, 27, 28, 29
            # Block 2: 32, 33, 36, 37, 38, 39, 40, 41
            # Block 3: 44, 45, 48, 49, 50, 51, 52, 53
            # Pattern: base = 20 + (i-1)*12
            base = 20 + (i - 1) * 12
            
            tch_dict[f'{tch_prefix}.weight__{base}'] = state_dict[f'{py_prefix}.bn2.weight']
            tch_dict[f'{tch_prefix}.bias__{base + 1}'] = state_dict[f'{py_prefix}.bn2.bias']
            tch_dict[f'{tch_prefix}.bias__{base + 4}'] = state_dict[f'{py_prefix}.bn1.bias']
            tch_dict[f'{tch_prefix}.weight__{base + 5}'] = state_dict[f'{py_prefix}.conv2.weight']
            tch_dict[f'{tch_prefix}.weight__{base + 6}'] = state_dict[f'{py_prefix}.bn1.weight']
            tch_dict[f'{tch_prefix}.bias__{base + 7}'] = state_dict[f'{py_prefix}.conv2.bias']
            tch_dict[f'{tch_prefix}.running_mean__{base + 8}'] = state_dict[f'{py_prefix}.bn2.running_mean']
            tch_dict[f'{tch_prefix}.running_var__{base + 9}'] = state_dict[f'{py_prefix}.bn2.running_var']
    
    # Value head - pattern for 10x128
    if num_blocks == 10:
        tch_dict['value_head.weight'] = state_dict['valueHead.conv1.weight']
        tch_dict['value_head.bias'] = state_dict['valueHead.conv1.bias']
        tch_dict['value_head.weight__128'] = state_dict['valueHead.bn1.weight']
        tch_dict['value_head.bias__129'] = state_dict['valueHead.bn1.bias']
        tch_dict['value_head.running_mean'] = state_dict['valueHead.bn1.running_mean']
        tch_dict['value_head.running_var'] = state_dict['valueHead.bn1.running_var']
        tch_dict['value_head.weight__133'] = state_dict['valueHead.fc1.weight']
        tch_dict['value_head.bias__132'] = state_dict['valueHead.fc1.bias']
        tch_dict['value_head.weight__135'] = state_dict['valueHead.fc2.weight']
        tch_dict['value_head.bias__134'] = state_dict['valueHead.fc2.bias']
    else:  # 20x256
        tch_dict['value_head.weight'] = state_dict['valueHead.conv1.weight']
        tch_dict['value_head.bias'] = state_dict['valueHead.conv1.bias']
        tch_dict['value_head.weight__248'] = state_dict['valueHead.bn1.weight']
        tch_dict['value_head.bias__249'] = state_dict['valueHead.bn1.bias']
        tch_dict['value_head.running_mean'] = state_dict['valueHead.bn1.running_mean']
        tch_dict['value_head.running_var'] = state_dict['valueHead.bn1.running_var']
        tch_dict['value_head.weight__253'] = state_dict['valueHead.fc1.weight']
        tch_dict['value_head.bias__252'] = state_dict['valueHead.fc1.bias']
        tch_dict['value_head.weight__255'] = state_dict['valueHead.fc2.weight']
        tch_dict['value_head.bias__254'] = state_dict['valueHead.fc2.bias']
    
    # Policy head
    if num_blocks == 10:
        tch_dict['policy_head.weight'] = state_dict['policyHead.conv1.weight']
        tch_dict['policy_head.bias'] = state_dict['policyHead.conv1.bias']
        tch_dict['policy_head.weight__138'] = state_dict['policyHead.bn1.weight']
        tch_dict['policy_head.bias__139'] = state_dict['policyHead.bn1.bias']
        tch_dict['policy_head.running_mean'] = state_dict['policyHead.bn1.running_mean']
        tch_dict['policy_head.running_var'] = state_dict['policyHead.bn1.running_var']
        tch_dict['policy_head.weight__143'] = state_dict['policyHead.fc1.weight']
        tch_dict['policy_head.bias__142'] = state_dict['policyHead.fc1.bias']
    else:  # 20x256
        tch_dict['policy_head.weight'] = state_dict['policyHead.conv1.weight']
        tch_dict['policy_head.bias'] = state_dict['policyHead.conv1.bias']
        tch_dict['policy_head.weight__258'] = state_dict['policyHead.bn1.weight']
        tch_dict['policy_head.bias__259'] = state_dict['policyHead.bn1.bias']
        tch_dict['policy_head.running_mean'] = state_dict['policyHead.bn1.running_mean']
        tch_dict['policy_head.running_var'] = state_dict['policyHead.bn1.running_var']
        tch_dict['policy_head.weight__263'] = state_dict['policyHead.fc1.weight']
        tch_dict['policy_head.bias__262'] = state_dict['policyHead.fc1.bias']
    
    return tch_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_weights_working.py <input_weights.pt> [output_weights.pt]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.pt', '_rust.pt')
    
    # Load the state dict
    state_dict = torch.load(input_file, map_location='cpu')
    print(f"Loaded {len(state_dict)} parameters from {input_file}")
    
    # Detect architecture
    num_blocks = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('residualBlocks.')]) + 1
    num_filters = state_dict['convBlock1.conv1.weight'].shape[0]
    
    print(f"Detected architecture: {num_blocks} blocks, {num_filters} filters")
    
    # Convert
    tch_dict = convert_to_tch_format(state_dict, num_blocks, num_filters)
    
    # Save
    torch.save(tch_dict, output_file)
    print(f"\nSaved converted weights to {output_file}")
    print(f"Total parameters: {len(tch_dict)}")
    
    # Show some keys for verification
    print("\nSample keys:")
    for key in sorted(tch_dict.keys())[:10]:
        print(f"  {key}")