#!/usr/bin/env python3
import json
import os
import glob
import numpy as np
from collections import defaultdict

def aggregate_benchmark_data(base_dir, json_filename, data_key_path):
    """
    base_dir: e.g., 'results/e2e_benchmark'
    json_filename: 'e2e_benchmark.json'
    data_key_path: list of keys to reach the dict of results, e.g., ['pipelines'] or ['kernels']
    """
    pattern = os.path.join(base_dir, "*", json_filename)
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found for {pattern}")
        return None

    all_data = defaultdict(lambda: defaultdict(list))
    
    config_labels = []
    
    for f_path in files:
        with open(f_path, 'r') as f:
            data = json.load(f)
            
            # Identify config labels (batch sizes or config names)
            if not config_labels:
                if 'batch_sizes' in data:
                    config_labels = data['batch_sizes']
                elif 'configs' in data:
                    config_labels = data['configs']
            
            target = data
            for key in data_key_path:
                target = target[key]
            
            # target is a dict: { "KernelName": [val1, val2, ...] }
            for name, values in target.items():
                for i, val in enumerate(values):
                    if val > 0: # Skip failures
                        all_data[name][i].append(val)

    aggregated = {}
    for name, configs in all_data.items():
        aggregated[name] = []
        for i in sorted(configs.keys()):
            vals = configs[i]
            stats = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": len(vals)
            }
            aggregated[name].append(stats)
            
    return {
        "config_labels": config_labels,
        "results": aggregated
    }

def main():
    merged = {}
    
    print("Aggregating E2E throughput results...")
    e2e_throughput = aggregate_benchmark_data('results/e2e_benchmark', 'e2e_benchmark.json', ['pipelines'])
    if e2e_throughput:
        merged['e2e_throughput'] = e2e_throughput

    print("Aggregating E2E latency results...")
    e2e_latency = aggregate_benchmark_data('results/e2e_benchmark', 'e2e_benchmark.json', ['latency_ms'])
    if e2e_latency:
        merged['e2e_latency'] = e2e_latency

    print("Aggregating Microbenchmark latency results...")
    micro_latency = aggregate_benchmark_data('results/micro_benchmark', 'micro_benchmark.json', ['kernels'])
    if micro_latency:
        merged['micro_latency'] = micro_latency

    with open('merged_data.json', 'w') as f:
        json.dump(merged, f, indent=2)
    
    print("\n✓  Saved merged_data.json")
    
    # Print a quick summary of speedup for Triton Ragged vs FA2 Varlen
    if 'e2e_throughput' in merged:
        print("\nSummary: E2E Throughput (Mean) Speedup (Triton vs FA2)")
        labels = merged['e2e_throughput']['config_labels']
        results = merged['e2e_throughput']['results']
        
        triton = results.get('Triton Ragged (ours)')
        fa2 = results.get('FlashAttention-2 (varlen)')
        
        if triton and fa2:
            print(f"{'BS':<5} | {'FA2 Mean':<10} | {'Triton Mean':<12} | {'Speedup':<8}")
            print("-" * 45)
            for i, label in enumerate(labels):
                t_val = triton[i]['mean']
                f_val = fa2[i]['mean']
                speedup = t_val / f_val if f_val > 0 else 0
                print(f"{label:<5} | {f_val:10.2f} | {t_val:12.2f} | {speedup:.2f}x")

if __name__ == "__main__":
    main()
