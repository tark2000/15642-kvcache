"""
Analyze BSA attention patterns:
1. Cross-layer block overlap: how well does layer L predict layer L+k?
2. Temporal locality: how often is the same block selected in consecutive decode steps?
"""
import sys
import os
import numpy as np
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import sim_config
from trace_reader import MobaTraceReader


def analyze_cross_layer_overlap(traces, num_layers, max_k=5):
    """
    For each layer L and offset k, compute:
    P(block selected at layer L+k | block selected at layer L)
    """
    print("\n=== Cross-Layer Block Overlap ===")
    print("Jaccard similarity between layer L and layer L+k selections")
    print(f"{'k':>4}", end="")
    for k in range(1, max_k + 1):
        print(f"  k={k}  ", end="")
    print()
    print("-" * (6 + 8 * max_k))

    # Average Jaccard per k across all layers and traces
    jaccard_by_k = defaultdict(list)

    for reader in traces:
        num_iters = reader.traces[0][0].shape[2]
        for cur_iter in range(num_iters):
            for cur_layer in range(num_layers):
                blocks_L = set(reader.traces[cur_layer][0][:, :, cur_iter].flatten().tolist())
                blocks_L.add(reader._get_last_block_id())

                for k in range(1, max_k + 1):
                    if cur_layer + k >= num_layers:
                        continue
                    blocks_Lk = set(reader.traces[cur_layer + k][0][:, :, cur_iter].flatten().tolist())
                    blocks_Lk.add(reader._get_last_block_id())

                    intersection = len(blocks_L & blocks_Lk)
                    union = len(blocks_L | blocks_Lk)
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_by_k[k].append(jaccard)

    # Print average per k
    print("avg ", end="")
    for k in range(1, max_k + 1):
        vals = jaccard_by_k[k]
        print(f"  {np.mean(vals):.4f}", end="")
    print()

    # Also print per-layer breakdown for k=1
    print("\nPer-layer Jaccard (k=1):")
    layer_jaccard = defaultdict(list)
    for reader in traces:
        num_iters = reader.traces[0][0].shape[2]
        for cur_iter in range(num_iters):
            for cur_layer in range(num_layers - 1):
                blocks_L = set(reader.traces[cur_layer][0][:, :, cur_iter].flatten().tolist())
                blocks_Lk = set(reader.traces[cur_layer + 1][0][:, :, cur_iter].flatten().tolist())
                intersection = len(blocks_L & blocks_Lk)
                union = len(blocks_L | blocks_Lk)
                jaccard = intersection / union if union > 0 else 0.0
                layer_jaccard[cur_layer].append(jaccard)

    for layer in range(0, num_layers - 1, 4):
        print(f"  layer {layer:02d}->{layer+1:02d}: {np.mean(layer_jaccard[layer]):.4f}")

    return jaccard_by_k


def analyze_temporal_locality(traces, num_layers):
    """
    For each decode step t, compute P(block at step t | block at step t-1).
    High value = LIRS-friendly (temporal locality explains LIRS performance).
    """
    print("\n=== Temporal Locality ===")
    print("Jaccard similarity between step t and step t-1 selections")

    jaccard_by_layer = defaultdict(list)
    jaccard_overall = []

    for reader in traces:
        num_iters = reader.traces[0][0].shape[2]
        for cur_layer in range(num_layers):
            for cur_iter in range(1, num_iters):
                blocks_t = set(reader.traces[cur_layer][0][:, :, cur_iter].flatten().tolist())
                blocks_prev = set(reader.traces[cur_layer][0][:, :, cur_iter - 1].flatten().tolist())
                blocks_t.add(reader._get_last_block_id())
                blocks_prev.add(reader._get_last_block_id())

                intersection = len(blocks_t & blocks_prev)
                union = len(blocks_t | blocks_prev)
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_by_layer[cur_layer].append(jaccard)
                jaccard_overall.append(jaccard)

    print(f"Overall avg temporal Jaccard: {np.mean(jaccard_overall):.4f}")
    print("\nPer-layer temporal Jaccard (sampled):")
    for layer in range(0, num_layers, 4):
        print(f"  layer {layer:02d}: {np.mean(jaccard_by_layer[layer]):.4f}")

    return jaccard_overall, jaccard_by_layer


def analyze_shallow_deep_correlation(traces, num_layers, shallow_cutoff=7):
    """
    P(block selected in deep layers | block selected in shallow layers)
    Measures how well shallow layers predict deep layers.
    """
    print(f"\n=== Shallow-Deep Correlation (shallow=layer 0-{shallow_cutoff-1}) ===")

    precision_list = []  # of shallow predictions that appear in deep layers
    recall_list = []     # of deep layer blocks that were predicted by shallow

    for reader in traces:
        num_iters = reader.traces[0][0].shape[2]
        for cur_iter in range(num_iters):
            # Shallow: blocks selected in layers 0 to shallow_cutoff-1
            shallow_blocks = set()
            for cur_layer in range(shallow_cutoff):
                shallow_blocks.update(reader.traces[cur_layer][0][:, :, cur_iter].flatten().tolist())
            shallow_blocks.add(reader._get_last_block_id())

            # Deep: blocks selected in layers shallow_cutoff to num_layers-1
            deep_blocks = set()
            for cur_layer in range(shallow_cutoff, num_layers):
                deep_blocks.update(reader.traces[cur_layer][0][:, :, cur_iter].flatten().tolist())
            deep_blocks.add(reader._get_last_block_id())

            if shallow_blocks and deep_blocks:
                tp = len(shallow_blocks & deep_blocks)
                precision = tp / len(shallow_blocks)
                recall = tp / len(deep_blocks)
                precision_list.append(precision)
                recall_list.append(recall)

    print(f"Precision (shallow→deep): {np.mean(precision_list):.4f}  "
          f"(how many shallow-selected blocks appear in deep layers)")
    print(f"Recall    (deep←shallow): {np.mean(recall_list):.4f}  "
          f"(how many deep-layer blocks were predicted by shallow layers)")


def main():
    config = sim_config()
    config.from_yaml("config.yaml")

    trace_dirs = sorted(glob(os.path.join(config.trace_dir, "trace*")))[:10]
    print(f"Analyzing {len(trace_dirs)} traces, {config.num_layers} layers")

    readers = [MobaTraceReader(trace_dir=td, verbose=False) for td in trace_dirs]

    analyze_cross_layer_overlap(readers, config.num_layers, max_k=5)
    analyze_temporal_locality(readers, config.num_layers)
    analyze_shallow_deep_correlation(readers, config.num_layers, shallow_cutoff=7)


if __name__ == "__main__":
    main()
