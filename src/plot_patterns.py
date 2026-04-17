import sys, os, numpy as np, matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import sim_config
from trace_reader import MobaTraceReader

def main():
    config = sim_config()
    config.from_yaml("config.yaml")
    trace_dirs = sorted(glob(os.path.join(config.trace_dir, "trace*")))[:10]
    print(f"Analyzing {len(trace_dirs)} traces...")
    readers = [MobaTraceReader(trace_dir=td, verbose=False) for td in trace_dirs]
    num_layers = config.num_layers
    max_k = 10

    jaccard_by_k = defaultdict(list)
    layer_jaccard_k1 = defaultdict(list)
    temporal_by_layer = defaultdict(list)

    for reader in readers:
        num_iters = reader.traces[0][0].shape[2]
        for cur_iter in range(num_iters):
            for cur_layer in range(num_layers):
                blocks_L = set(reader.traces[cur_layer][0][:,:,cur_iter].flatten().tolist())
                blocks_L.add(reader._get_last_block_id())
                for k in range(1, max_k+1):
                    if cur_layer+k >= num_layers: continue
                    blocks_Lk = set(reader.traces[cur_layer+k][0][:,:,cur_iter].flatten().tolist())
                    blocks_Lk.add(reader._get_last_block_id())
                    inter = len(blocks_L & blocks_Lk)
                    union = len(blocks_L | blocks_Lk)
                    jaccard_by_k[k].append(inter/union if union>0 else 0.0)
                if cur_layer < num_layers-1:
                    blocks_Lk = set(reader.traces[cur_layer+1][0][:,:,cur_iter].flatten().tolist())
                    inter = len(blocks_L & blocks_Lk)
                    union = len(blocks_L | blocks_Lk)
                    layer_jaccard_k1[cur_layer].append(inter/union if union>0 else 0.0)
            for cur_layer in range(num_layers):
                for cur_iter in range(1, num_iters):
                    bt = set(reader.traces[cur_layer][0][:,:,cur_iter].flatten().tolist())
                    bp = set(reader.traces[cur_layer][0][:,:,cur_iter-1].flatten().tolist())
                    bt.add(reader._get_last_block_id())
                    bp.add(reader._get_last_block_id())
                    inter = len(bt & bp)
                    union = len(bt | bp)
                    temporal_by_layer[cur_layer].append(inter/union if union>0 else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ks = list(range(1, max_k+1))
    means = [np.mean(jaccard_by_k[k]) for k in ks]
    stds  = [np.std(jaccard_by_k[k])  for k in ks]
    axes[0].plot(ks, means, marker='o', color='steelblue', linewidth=2)
    axes[0].fill_between(ks, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.2, color='steelblue')
    axes[0].axhline(y=np.mean(means), color='red', linestyle='--', alpha=0.5, label=f'avg={np.mean(means):.3f}')
    axes[0].set_xlabel('Layer offset k'); axes[0].set_ylabel('Jaccard similarity')
    axes[0].set_title('Cross-Layer Block Overlap\n(layer L vs layer L+k)')
    axes[0].set_ylim(0,1); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    layers = list(range(num_layers-1))
    layer_means = [np.mean(layer_jaccard_k1[l]) for l in layers]
    axes[1].bar(layers, layer_means, color='steelblue', alpha=0.7)
    axes[1].axhline(y=np.mean(layer_means), color='red', linestyle='--', alpha=0.5, label=f'avg={np.mean(layer_means):.3f}')
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Jaccard similarity')
    axes[1].set_title('Per-Layer Block Overlap (k=1)\n(layer L vs layer L+1)')
    axes[1].set_ylim(0,1); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    temp_means = [np.mean(temporal_by_layer[l]) for l in range(num_layers)]
    axes[2].bar(range(num_layers), temp_means, color='darkorange', alpha=0.7)
    axes[2].axhline(y=np.mean(temp_means), color='red', linestyle='--', alpha=0.5, label=f'avg={np.mean(temp_means):.3f}')
    axes[2].set_xlabel('Layer'); axes[2].set_ylabel('Jaccard similarity')
    axes[2].set_title('Temporal Locality per Layer\n(step t vs step t-1)')
    axes[2].set_ylim(0,1); axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pattern_analysis.png', dpi=300)
    print("Saved to results/pattern_analysis.png")

if __name__ == "__main__":
    main()
