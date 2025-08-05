#!/usr/bin/env python3
"""Create a mapping of Rust variable names to help with conversion."""

import subprocess
import re

# Run the test_model binary and capture output
result = subprocess.run(
    ["DYLD_LIBRARY_PATH=/Users/rishisachdev/libtorch/lib", "cargo", "run", "--release", "--bin", "test_model"],
    shell=True,
    capture_output=True,
    text=True
)

# Extract variable names from output
lines = result.stdout.strip().split('\n')
var_names = []
for line in lines:
    if line.strip().startswith('res_block_') or line.strip().startswith('conv_block') or line.strip().startswith('value_head') or line.strip().startswith('policy_head'):
        var_names.append(line.strip())

print(f"Found {len(var_names)} variables")

# Group by component
components = {}
for name in var_names:
    base = name.split('.')[0]
    if base not in components:
        components[base] = []
    components[base].append(name)

# Print grouped
for comp, names in sorted(components.items()):
    print(f"\n{comp}:")
    for name in sorted(names):
        print(f"  {name}")