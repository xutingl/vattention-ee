#!/usr/bin/env python3
"""
Analyze quality metrics from EE experiment outputs.
Compares rebatching vs other policies to show rebatching's quality advantage.
"""

import os
import re
import sys
import glob
from collections import defaultdict

def extract_metrics(filepath):
    """Extract key metrics from a log file."""
    metrics = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    
    # Extract metrics using regex
    patterns = {
        'throughput': r'Throughput: ([\d.]+) tokens/sec',
        'rougeL': r'RougeL: ([\d.]+)',
        'bert_score': r'Bert_score: ([\d.]+)',
        'num_ee_iter': r'Num EE iter: (\d+)',
        'num_no_ee_iter': r'Num non-EE iter: (\d+)',
        'num_would_ee_but_stay': r'Num seq would ee but stay: (\d+)',
        'num_forced_ee': r'Num seq would not ee but ee: (\d+)',
        'exited_tokens': r'Exited rates\(#tokens generated via ee vs\. non-ee\): \[(\d+), (\d+)\]',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'exited_tokens':
                metrics['ee_tokens'] = int(match.group(1))
                metrics['non_ee_tokens'] = int(match.group(2))
            else:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    metrics[key] = int(match.group(1))
    
    return metrics

def parse_filename(filename):
    """Extract configuration from filename."""
    # Pattern: req_500_batch_4_layer_24_conf_0.5_rebatching.txt
    pattern = r'req_(\d+)_batch_(\d+)_layer_(\d+)_conf_([\d.]+)_(\w+[-\w]*)'
    match = re.search(pattern, filename)
    if match:
        return {
            'num_requests': int(match.group(1)),
            'batch_size': int(match.group(2)),
            'layer': int(match.group(3)),
            'conf': float(match.group(4)),
            'policy': match.group(5).replace('_copy', '')
        }
    return None

def main(output_dir):
    """Analyze all results in a directory."""
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        sys.exit(1)
    
    # Find all txt files
    txt_files = glob.glob(os.path.join(output_dir, '*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {output_dir}")
        sys.exit(1)
    
    results = []
    
    for filepath in sorted(txt_files):
        filename = os.path.basename(filepath)
        config = parse_filename(filename)
        if not config:
            continue
            
        metrics = extract_metrics(filepath)
        if not metrics:
            continue
        
        results.append({**config, **metrics})
    
    if not results:
        print("No valid results found")
        sys.exit(1)
    
    # Group by (layer, conf) to compare policies
    grouped = defaultdict(list)
    for r in results:
        key = (r['layer'], r['conf'])
        grouped[key].append(r)
    
    # Print comparison table
    print("\n" + "="*120)
    print("QUALITY COMPARISON: rebatching vs other policies")
    print("="*120)
    print(f"{'Layer':<6} {'Conf':<6} {'Policy':<15} {'EE iter':<10} {'Forced EE':<12} {'RougeL':<10} {'BERT':<10} {'Throughput':<12}")
    print("-"*120)
    
    for (layer, conf), group in sorted(grouped.items()):
        # Sort by policy name for consistent ordering
        group = sorted(group, key=lambda x: x['policy'])
        
        for r in group:
            forced_ee = r.get('num_forced_ee', 0)
            rougeL = r.get('rougeL', 0)
            bert = r.get('bert_score', 0)
            throughput = r.get('throughput', 0)
            num_ee = r.get('num_ee_iter', 0)
            
            # Highlight forced exits (quality hit indicator)
            forced_str = f"{forced_ee:,}" if forced_ee == 0 else f"** {forced_ee:,} **"
            
            print(f"{layer:<6} {conf:<6} {r['policy']:<15} {num_ee:<10,} {forced_str:<12} {rougeL:<10.4f} {bert:<10.4f} {throughput:<12.2f}")
        
        print("-"*120)
    
    # Summary statistics
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("- 'Forced EE' = sequences forced to exit without meeting confidence threshold")
    print("- rebatching should have Forced EE = 0 (selective exits)")
    print("- Higher Forced EE = lower quality (lazy/median force batch exits)")
    print("- Compare RougeL and BERT scores: rebatching should be higher when EE is active")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to most recent sweep directory
        dirs = ['outputs_quality_comparison', 'outputs_xsum_sweep_feb_3', 'outputs_xsum_sweep_feb_2']
        for d in dirs:
            if os.path.exists(d):
                output_dir = d
                break
        else:
            print("Usage: python analyze_quality.py <output_directory>")
            sys.exit(1)
    else:
        output_dir = sys.argv[1]
    
    main(output_dir)
