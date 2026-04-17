"""
Hyperparameter sweep for ProactiveEvictionCache.
Sweeps lookahead_boost and shallow_boost values and reports hit rates.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import sim_config
from trace_reader import MobaTraceReader
from proactive_eviction import ProactiveEvictionCache

def run_proactive(cache_size_gb: float, lookahead_boost: float, shallow_boost: float,
                  trace_dirs: list, num_layers: int = 28) -> float:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from simulator import process_trace_with_proactive

    cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
    total_hit_rate = 0.0

    for trace_dir in trace_dirs:
        reader = MobaTraceReader(trace_dir=trace_dir, verbose=False)
        cache = ProactiveEvictionCache(
            cache_size=cache_size_bytes,
            beta=0.9,
            shallow_boost=shallow_boost,
            num_layers=num_layers,
        )
        miss_ratio, _ = process_trace_with_proactive(cache, reader)
        total_hit_rate += (1.0 - miss_ratio)

    return total_hit_rate / len(trace_dirs)

def main():
    config = sim_config()
    config.from_yaml("config.yaml")

    # Use first 5 traces for speed
    from glob import glob
    trace_dirs = sorted(glob(os.path.join(config.trace_dir, "trace*")))[:5]
    print(f"Using {len(trace_dirs)} traces")

    cache_sizes = [1.0, 1.5, 2.0, 2.5, 3.0]

    shallow_boosts = [0.0, 0.1, 0.3, 0.5]

    print(f"\n{'shallow':>10} {'1.0GB':>8} {'1.5GB':>8} {'2.0GB':>8} {'2.5GB':>8} {'3.0GB':>8}")
    print("-" * 55)

    for sb in shallow_boosts:
        row = f"{sb:>10.1f}"
        avg_total = 0.0
        for cs in cache_sizes:
            hr = run_proactive(cs, 0.0, sb, trace_dirs)
            row += f" {hr:>8.4f}"
            avg_total += hr
        avg = avg_total / len(cache_sizes)
        row += f"  avg={avg:.4f}"
        print(row)


if __name__ == "__main__":
    main()