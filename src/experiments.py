"""
Run experiments that compare cache hit rates across cache sizes and pinning vs non-pinning.

Two experiments are supported:
- Algorithm comparison: Y-axis = hit rate, X-axis = cache size (GB), one line per algorithm
- Pinning comparison: Y-axis = hit rate, X-axis = cache size (GB), comparing pinned vs non-pinned variants

This script uses MobaTraceReader to generate requests and can use both libcachesim-backed caches
and pure-Python caches implemented here (LRU) for pinning experiments.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Dict, Tuple
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import sim_config, SimConfig
from trace_reader import MobaTraceReader
from kvcache import BsaKVCache
import re
from glob import glob
import shutil
from pathlib import Path
from simulator import setup_cache, process_trace_with_momentum, LCS_ALGORITHMS, CUSTOM_ALGORITHMS
from momentum_cache import MomentumDecayCache

logger = logging.getLogger(__name__)


class PythonLRUCache:
    """Simple LRU cache with pinning support and sizes in bytes.

    Provides an API similar to custom Momentum cache for experiments.
    """

    def __init__(self, cache_size: int):
        self.capacity = cache_size
        self.occupied_bytes = 0
        self.cache = OrderedDict()  # obj_id -> (obj_size)
        self.pinned_objs = set()

        # Stats
        self.hits = 0
        self.misses = 0
        self.bytes_hits = 0
        self.bytes_misses = 0
        self.total_requests = 0

    def _evict_one(self) -> bool:
        # Evict least recently used object that is not pinned
        for obj_id in list(self.cache.keys()):
            if obj_id in self.pinned_objs:
                # Skip pinned objects
                continue
            obj_size = self.cache.pop(obj_id)
            self.occupied_bytes -= obj_size
            return True
        return False

    def access(self, obj_id: int, obj_size: int) -> bool:
        self.total_requests += 1
        if obj_id in self.cache:
            # hit: move to back
            self.cache.move_to_end(obj_id)
            self.hits += 1
            self.bytes_hits += obj_size
            return True
        # miss
        self.misses += 1
        self.bytes_misses += obj_size
        # ensure capacity
        while self.occupied_bytes + obj_size > self.capacity and len(self.cache) > 0:
            # If we can't evict anything, break to avoid infinite loop
            if not self._evict_one():
                break
        # If still not enough space, skip insertion (simulate fetching from origin)
        if self.occupied_bytes + obj_size <= self.capacity:
            self.cache[obj_id] = obj_size
            self.cache.move_to_end(obj_id)
            self.occupied_bytes += obj_size
        return False

    def process_trace(self, reader: MobaTraceReader) -> Tuple[float, float]:
        # Reset stats
        self.hits = self.misses = self.total_requests = 0
        self.bytes_hits = self.bytes_misses = 0
        # Use generate_requests_with_scores to get obj id & size & score tuples
        for obj_id, obj_size, score in reader.generate_requests_with_scores():
            self.access(obj_id, obj_size)
        if self.total_requests == 0:
            return 0.0, 0.0
        req_miss_ratio = self.misses / self.total_requests
        total_bytes = self.bytes_hits + self.bytes_misses
        bytes_miss_ratio = self.bytes_misses / total_bytes if total_bytes > 0 else 0.0
        return req_miss_ratio, bytes_miss_ratio

    def pin(self, obj_id: int):
        self.pinned_objs.add(obj_id)

    def unpin(self, obj_id: int):
        if obj_id in self.pinned_objs:
            self.pinned_objs.remove(obj_id)
            
def process_trace_with_lcs_cache(cache, reader: MobaTraceReader):
    """Process trace using libcachesim cache with BSA-granularity requests."""
    import libcachesim as lcs
    hits = 0
    total = 0
    for obj_id, obj_size, score in reader.generate_requests_with_scores():
        req = lcs.Request(obj_size=obj_size, obj_id=obj_id)
        hit = cache.find(req, update_cache=True)
        if hit is None:
            # miss
            if cache.need_eviction(req):
                cache.evict(req)
            if cache.can_insert(req):
                cache.insert(req)
        else:
            hits += 1
        total += 1
    miss_ratio = 1.0 - (hits / total if total > 0 else 0.0)
    return miss_ratio, miss_ratio

def run_algorithm_comparison(config_path: str, algorithms: List[str], cache_sizes: List[float], out_path: str, max_traces: int = 0):
    config = sim_config()
    config.from_yaml(config_path)

    # Prepare x axis
    x_sizes = cache_sizes
    results: Dict[str, List[float]] = {alg: [] for alg in algorithms}

    # Validate algorithm list
    valid_algorithms = set(list(LCS_ALGORITHMS.keys()) + list(CUSTOM_ALGORITHMS) + ["python_lru"])
    for alg in algorithms:
        if alg.lower() not in valid_algorithms:
            raise ValueError(f"Unknown algorithm {alg}. Valid algorithms: {sorted(valid_algorithms)}")

    for size in tqdm(x_sizes, desc="Cache sizes"):
        # Run each algorithm with this cache size
        for alg in algorithms:
            config.cache_size = float(size)  # in GB
            config.eviction_algorithm = alg
            # Average over traces
            num_traces = len([f for f in os.listdir(config.trace_dir) if os.path.isdir(os.path.join(config.trace_dir, f)) == False])
            # Discover trace directories: either traceNNNN subdirs or files in a single dir
            trace_subdirs = sorted([d for d in glob(os.path.join(config.trace_dir, "trace*")) if os.path.isdir(d)])
            if len(trace_subdirs) > 0:
                traces = trace_subdirs
            else:
                # Single directory contains trace files across IDs; treat as single "trace"
                traces = [config.trace_dir]
            total_req_hit_rate = 0.0
            for tidx, trace_dir in enumerate(traces):
                if max_traces and tidx >= max_traces:
                    break
                # If trace_dir is a flat folder with many traceNNNN files, create a per-seq folder
                if trace_dir == config.trace_dir:
                    # discover seq ids
                    files = os.listdir(config.trace_dir)
                    trace_ids = sorted(set([m.group(1) for f in files for m in [re.match(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy", f)] if m]))
                    # We'll process first max_traces seq ids
                    if len(trace_ids) == 0:
                        raise ValueError(f"No trace files found in {config.trace_dir}")
                    # If we are enumerating per-directory, we need to iterate seq ids separately
                    # So set traces to the per-seq directories (virtual via copying)
                    traces = []
                    for sid in trace_ids:
                        traces.append(os.path.join(config.trace_dir, f"trace{sid}"))
                    # Re-run enumeration from the beginning using the new traces list
                    for tidx2, trace_dir2 in enumerate(traces):
                        if max_traces and tidx2 >= max_traces:
                            break
                        # assemble a temporary folder that contains files for this seq
                        seqid = trace_dir2.split('trace')[-1]
                        tmp_dir = os.path.join(config.trace_dir, f"tmp_trace{seqid}")
                        os.makedirs(tmp_dir, exist_ok=True)
                        for f in os.listdir(config.trace_dir):
                            if f.startswith(f"trace{seqid}_"):
                                shutil.copy(os.path.join(config.trace_dir, f), os.path.join(tmp_dir, f))
                        trace_dir_use = tmp_dir
                        # run with trace_dir_use
                        reader = MobaTraceReader(trace_dir=trace_dir_use, verbose=config.verbose)
                        # cleanup will be done at end of iteration
                        cache_size_bytes = int(config.cache_size * 1024 * 1024 * 1024)
                        if alg.lower() == 'momentum_decay':
                            cache = setup_cache(config)
                            req_miss_ratio, _ = process_trace_with_momentum(cache, reader)
                        elif alg.lower() == 'proactive_eviction':
                            from proactive_eviction import ProactiveEvictionCache
                            from simulator import process_trace_with_proactive
                            cache = setup_cache(config)
                            req_miss_ratio, _ = process_trace_with_proactive(cache, reader)
                        elif alg.lower() == 'belady':
                            from belady_cache import BeladyCache
                            from simulator import process_trace_with_belady
                            cache = BeladyCache(int(config.cache_size * 1024 * 1024 * 1024))
                            req_miss_ratio, _ = process_trace_with_belady(cache, reader)
                        elif alg.lower() == 'python_lru':
                            cache = PythonLRUCache(int(config.cache_size * 1024 * 1024 * 1024))
                            req_miss_ratio, _ = cache.process_trace(reader)
                        else:
                            cache = setup_cache(config)
                            req_miss_ratio, _ = process_trace_with_lcs_cache(cache, reader)

                        req_hit_rate = 1.0 - req_miss_ratio
                        total_req_hit_rate += req_hit_rate
                        # cleanup tmp
                        shutil.rmtree(tmp_dir)
                    # After this per-seq enumerations, we skip the rest of outer loop for this size/alg
                    continue
                # trace_dir is set from traces list
                reader = MobaTraceReader(trace_dir=trace_dir, verbose=config.verbose)
                cache_size_bytes = int(config.cache_size * 1024 * 1024 * 1024)
                # Use setup_cache
                if alg.lower() == 'momentum_decay':
                    cache = setup_cache(config)
                    req_miss_ratio, _ = process_trace_with_momentum(cache, reader)
                elif alg.lower() == 'proactive_eviction':
                    from proactive_eviction import ProactiveEvictionCache
                    from simulator import process_trace_with_proactive
                    cache = setup_cache(config)
                    req_miss_ratio, _ = process_trace_with_proactive(cache, reader)
                elif alg.lower() == 'belady':
                    from belady_cache import BeladyCache
                    from simulator import process_trace_with_belady
                    cache = BeladyCache(int(config.cache_size * 1024 * 1024 * 1024))
                    req_miss_ratio, _ = process_trace_with_belady(cache, reader)
                elif alg.lower() == 'python_lru':
                    # Use our local Python LRU
                    cache = PythonLRUCache(int(config.cache_size * 1024 * 1024 * 1024))
                    req_miss_ratio, _ = cache.process_trace(reader)
                else:
                    cache = setup_cache(config)
                    req_miss_ratio, _ = process_trace_with_lcs_cache(cache, reader)

                req_hit_rate = 1.0 - req_miss_ratio
                total_req_hit_rate += req_hit_rate
            avg_hit_rate = total_req_hit_rate / max(1, min(len(traces), max_traces) if max_traces else len(traces))
            results[alg].append(avg_hit_rate)
            print(f"Alg={alg}, CacheSize={size}GB, AvgHitRate={avg_hit_rate:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    for alg, y in results.items():
        plt.plot(x_sizes, y, label=alg, marker='o')
    plt.xlabel('Cache size (GB)')
    plt.ylabel('Average Hit Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = out_path
    plt.savefig(save_path, dpi=300)
    print(f"Algorithm comparison plot saved to {save_path}")


def run_pinning_comparison(config_path: str, alg: str, cache_sizes: List[float], out_path: str, max_traces: int = 0):
    config = sim_config()
    config.from_yaml(config_path)

    x_sizes = cache_sizes
    pinned_rates = []
    unpinned_rates = []

    # We support pinning for momentum_decay and our Python LRU
    for size in tqdm(x_sizes, desc="Cache sizes (pinning)"):
        config.cache_size = float(size)
        cache_size_bytes = int(config.cache_size * 1024 * 1024 * 1024)

        # Helper to run traces with a cache instance
        def run_with_cache(cache_obj, pinned: bool) -> float:
            total_req_hit_rate = 0.0
            trace_subdirs = sorted([d for d in glob(os.path.join(config.trace_dir, "trace*")) if os.path.isdir(d)])
            if len(trace_subdirs) > 0:
                traces = trace_subdirs
            else:
                traces = [config.trace_dir]
            num_traces = len(traces)
            for tidx in range(num_traces):
                if max_traces and tidx >= max_traces:
                    break
                trace_dir = traces[tidx]
                # If trace directory is flat and contains many traceNNNN files, create per-seq tmp dir
                if trace_dir == config.trace_dir:
                    files = os.listdir(config.trace_dir)
                    trace_ids = sorted(set([m.group(1) for f in files for m in [re.match(r"trace(\d+)_layer(\d+)_(blocks|scores)\.npy", f)] if m]))
                    if len(trace_ids) == 0:
                        raise ValueError(f"No trace files found in {config.trace_dir}")
                    # map to selected seq id for tidx
                    sid = trace_ids[tidx]
                    tmp_dir = os.path.join(config.trace_dir, f"tmp_trace{sid}")
                    os.makedirs(tmp_dir, exist_ok=True)
                    for f in os.listdir(config.trace_dir):
                        if f.startswith(f"trace{sid}_"):
                            shutil.copy(os.path.join(config.trace_dir, f), os.path.join(tmp_dir, f))
                    trace_dir_use = tmp_dir
                else:
                    trace_dir_use = trace_dir
                reader = MobaTraceReader(trace_dir=trace_dir_use, verbose=config.verbose)
                # If pinned, pin the last block(s) for each layer/kvhead
                if pinned:
                    last_block = (config.context_length - 1) // config.block_size
                    # Convert last block -> obj IDs per the cache type
                    # For MomentumCache: the reader yields BSA or page object ids; we'll collect from a single iteration
                    # Here we will traverse a small slice of the trace to get obj ids to pin
                    for cur_layer in range(config.num_layers):
                        # Create obj ids for kvheads for last_block
                        if config.store_all_kvheads:
                            # For paged mode this pins pages; we approximate by not pinning in libcachesim
                            pass
                        else:
                            for kvhead in range(config.num_kvheads):
                                bsa = BsaKVCache(reader.seq_id, last_block, cur_layer, kvhead)
                                if isinstance(cache_obj, MomentumDecayCache):
                                    cache_obj.pin(bsa.get_obj_id())
                                elif hasattr(cache_obj, 'pin'):
                                    cache_obj.pin(bsa.get_obj_id())
                # Now process with cache
                if isinstance(cache_obj, MomentumDecayCache):
                    req_miss_ratio, _ = process_trace_with_momentum(cache_obj, reader)
                elif hasattr(cache_obj, 'process_trace'):
                    req_miss_ratio, _ = cache_obj.process_trace(reader)
                else:
                    # Fallback: attempt libcachesim cache
                    req_miss_ratio, _ = cache_obj.process_trace(reader)
                req_hit_rate = 1.0 - req_miss_ratio
                total_req_hit_rate += req_hit_rate
                if trace_dir == config.trace_dir:
                    shutil.rmtree(tmp_dir)
            avg = total_req_hit_rate / max(1, num_traces)
            return avg

        if alg.lower() == "momentum_decay":
            # unpinned
            cache_unpinned = MomentumDecayCache(cache_size=cache_size_bytes)
            rate_unpinned = run_with_cache(cache_unpinned, pinned=False)
            # pinned
            cache_pinned = MomentumDecayCache(cache_size=cache_size_bytes)
            rate_pinned = run_with_cache(cache_pinned, pinned=True)
        elif alg.lower() == "lru":
            cache_unpinned = PythonLRUCache(cache_size=cache_size_bytes)
            rate_unpinned = run_with_cache(cache_unpinned, pinned=False)
            cache_pinned = PythonLRUCache(cache_size=cache_size_bytes)
            rate_pinned = run_with_cache(cache_pinned, pinned=True)
        else:
            # Try libcachesim cache; we cannot guarantee pinning support
            cache_unpinned = setup_cache(config)
            rate_unpinned = run_with_cache(cache_unpinned, pinned=False)
            cache_pinned = setup_cache(config)
            # If the cache has 'pin' method, we will attempt to pin; else, run as-is
            rate_pinned = run_with_cache(cache_pinned, pinned=True)

        pinned_rates.append(rate_pinned)
        unpinned_rates.append(rate_unpinned)
        print(f"{alg} pinned/unpinned at size={size}GB: pinned={rate_pinned:.4f}, unpinned={rate_unpinned:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_sizes, unpinned_rates, label=f"{alg} unpinned", marker='o')
    plt.plot(x_sizes, pinned_rates, label=f"{alg} pinned", marker='x')
    plt.xlabel('Cache size (GB)')
    plt.ylabel('Average Hit Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = out_path
    plt.savefig(save_path, dpi=300)
    print(f"Pinning comparison plot saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--algorithms', type=str, default='momentum_decay,lru,s3fifo,fifo,lfu,arc,sieve,lirs,twoq,slru,random')
    # Accept either a single quoted comma-separated list or multiple values like: --cache-sizes 0.1 0.5 1.0
    parser.add_argument('--cache-sizes', nargs='+', default=['0.1,0.5,1,2'],
                        help='Cache sizes in GB, either comma-separated or space-separated entries')
    parser.add_argument('--max-traces', type=int, default=0, help='Maximum number of traces to process (0 means all)')
    parser.add_argument('--experiment', type=str, choices=['alg_compare', 'pin_compare'], default='alg_compare')
    parser.add_argument('--pin-alg', type=str, default='momentum_decay', help='Algorithm to use in pinning comparison (only used with pin_compare)')
    return parser.parse_args()


def main():
    args = parse_args()
    algorithms = [a.strip() for a in args.algorithms.split(',')]
    # Normalize cache size argument(s): args.cache_sizes could be ['0.1,0.5'] or ['0.1','0.5']
    cache_sizes = []
    for token in args.cache_sizes:
        # split on comma and whitespace
        parts = [p for p in re.split(r'[,\s]+', token.strip()) if p]
        for p in parts:
            try:
                cache_sizes.append(float(p))
            except ValueError:
                logger.warning(f"Ignoring invalid cache size: '{p}'")
    if args.experiment == 'alg_compare':
        run_algorithm_comparison(args.config, algorithms, cache_sizes, args.out, max_traces=args.max_traces)
    else:
        run_pinning_comparison(args.config, args.pin_alg, cache_sizes, args.out, max_traces=args.max_traces)


if __name__ == '__main__':
    main()
