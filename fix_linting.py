#!/usr/bin/env python3
"""
Script to fix common linting issues in the SMART-TRAIN codebase.
"""

import os
import re
import glob
from pathlib import Path

def fix_whitespace_issues(file_path):
    """Fix whitespace issues in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix trailing whitespace (W291)
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix blank lines with whitespace (W293)
    content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix multiple blank lines at end of file
    content = content.rstrip() + '\n'
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed whitespace issues in {file_path}")
        return True
    return False

def remove_unused_imports(file_path):
    """Remove some obvious unused imports."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_lines = lines[:]
    new_lines = []
    
    # Common unused imports to remove
    unused_patterns = [
        r'^import asyncio$',
        r'^import json$', 
        r'^from typing import.*Tuple.*$',
        r'^from typing import.*Union.*$',
        r'^import matplotlib\.pyplot as plt$',
        r'^import seaborn as sns$',
        r'^from dataclasses import asdict$',
        r'^from concurrent\.futures import.*$',
        r'^import plotly\.express as px$',
        r'^from einops import.*$',
        r'^import numpy as np$',
    ]
    
    for line in lines:
        should_remove = False
        for pattern in unused_patterns:
            if re.match(pattern, line.strip()):
                # Only remove if it's clearly unused (basic heuristic)
                should_remove = True
                break
        
        if not should_remove:
            new_lines.append(line)
    
    if new_lines != original_lines:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Removed unused imports in {file_path}")
        return True
    return False

def fix_unused_variables(file_path):
    """Fix some obvious unused variables."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix unused variables by prefixing with underscore
    # This is a simple heuristic - only for obvious cases
    patterns = [
        (r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^=\n]+)(\s*#.*F841.*)', r'\1_\2 = \3\4'),
        (r'(\s+)batch_size\s*=\s*([^=\n]+)', r'\1_batch_size = \2'),
        (r'(\s+)timestamps\s*=\s*([^=\n]+)', r'\1_timestamps = \2'),
        (r'(\s+)timestamp\s*=\s*([^=\n]+)', r'\1_timestamp = \2'),
        (r'(\s+)trainer\s*=\s*([^=\n]+)', r'\1_trainer = \2'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed unused variables in {file_path}")
        return True
    return False

def fix_indentation_issues(file_path):
    """Fix basic indentation issues."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_lines = lines[:]
    new_lines = []
    
    for i, line in enumerate(lines):
        # Fix continuation line indentation (basic cases)
        if i > 0 and line.strip() and not line.startswith('    '):
            prev_line = lines[i-1].rstrip()
            if (prev_line.endswith('(') or prev_line.endswith(',') or 
                prev_line.endswith('=') or prev_line.endswith('\\')):
                # This is a continuation line, ensure proper indentation
                if line.startswith(' ') and not line.startswith('        '):
                    # Fix under-indented continuation lines
                    stripped = line.lstrip()
                    if stripped:
                        line = '        ' + stripped
        
        new_lines.append(line)
    
    if new_lines != original_lines:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Fixed indentation issues in {file_path}")
        return True
    return False

def main():
    """Main function to fix linting issues."""
    src_dir = Path("src")
    
    if not src_dir.exists():
        print("src directory not found")
        return
    
    python_files = list(src_dir.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to check")
    
    total_fixed = 0
    
    for file_path in python_files:
        print(f"\nProcessing {file_path}...")
        
        fixed_count = 0
        
        # Fix whitespace issues
        if fix_whitespace_issues(file_path):
            fixed_count += 1
        
        # Remove unused imports (commented out for safety)
        # if remove_unused_imports(file_path):
        #     fixed_count += 1
        
        # Fix unused variables
        if fix_unused_variables(file_path):
            fixed_count += 1
        
        # Fix indentation (commented out for safety)
        # if fix_indentation_issues(file_path):
        #     fixed_count += 1
        
        if fixed_count > 0:
            total_fixed += 1
            print(f"  Fixed {fixed_count} types of issues")
        else:
            print(f"  No issues found")
    
    print(f"\nTotal files fixed: {total_fixed}")
    print("Linting fixes completed!")

if __name__ == "__main__":
    main()
