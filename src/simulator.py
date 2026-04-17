from config import sim_config, SimConfig
from argparse import ArgumentParser
import libcachesim as lcs
from trace_reader import MobaTraceReader
from momentum_cache import MomentumDecayCache
import logging
import os
from tqdm import tqdm
from typing import Union
from proactive_eviction import ProactiveEvictionCache

logger = logging.getLogger(__name__)

# Custom eviction algorithms (not from libcachesim)
CUSTOM_ALGORITHMS = {"momentum_decay", "proactive_eviction", "belady"}

# libcachesim-based eviction algorithms
LCS_ALGORITHMS = {
    "lru": lcs.LRU,
    "fifo": lcs.FIFO,
    "lfu": lcs.LFU,
    "arc": lcs.ARC,
    "s3fifo": lcs.S3FIFO,
    "sieve": lcs.Sieve,
    "lirs": lcs.LIRS,
    "twoq": lcs.TwoQ,
    "slru": lcs.SLRU,
    "random": lcs.Random,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
    )


def is_custom_algorithm(algorithm: str) -> bool:
    return algorithm.lower() in CUSTOM_ALGORITHMS


def setup_cache(config: SimConfig) -> Union[lcs.CacheBase, MomentumDecayCache]:
    algorithm = config.eviction_algorithm.lower()
    cache_size_bytes = int(config.cache_size * 1024 * 1024 * 1024)
    
    if algorithm == "momentum_decay":
        cache = MomentumDecayCache(cache_size=cache_size_bytes, beta=0.9)
        logger.info(f"Using cache eviction algorithm: momentum_decay (beta=0.9)")
        return cache
    
    if algorithm == "proactive_eviction":
        cache = ProactiveEvictionCache(cache_size=cache_size_bytes, beta=0.9)
        logger.info(f"Using cache eviction algorithm: proactive_eviction")
        return cache
    
    if algorithm == "belady":
        from belady_cache import BeladyCache
        cache = BeladyCache(cache_size=cache_size_bytes)
        logger.info(f"Using cache eviction algorithm: belady (optimal)")
        return cache


    cache_class = LCS_ALGORITHMS.get(algorithm)
    if cache_class is None:
        raise ValueError(
            f"Unknown cache eviction algorithm: {config.eviction_algorithm}. "
            f"Available: {list(LCS_ALGORITHMS.keys()) + list(CUSTOM_ALGORITHMS)}"
        )
    cache = cache_class(cache_size=cache_size_bytes)
    logger.info(f"Using cache eviction algorithm: {config.eviction_algorithm}")
    return cache


def process_trace_with_momentum(cache: MomentumDecayCache, reader: MobaTraceReader):
    """Process trace using MomentumDecayCache with score information."""
    for obj_id, obj_size, score in reader.generate_requests_with_scores():
        cache.access(obj_id, obj_size, score)
    return cache.get_miss_ratio()

def process_trace_with_proactive(cache, reader: MobaTraceReader):
    """Process trace with hard lookahead protection + shallow layer boost."""
    from proactive_eviction import ProactiveEvictionCache
    from kvcache import BsaKVCache
    config = sim_config()

    num_iters = reader.traces[0][0].shape[2]
    num_layers = config.num_layers
    shallow_cutoff = int(num_layers * 0.25)

    # Precompute shallow layer important blocks per iter
    shallow_important = {}
    for cur_iter in range(num_iters):
        important = set()
        for cur_layer in range(shallow_cutoff):
            block_ids = reader.traces[cur_layer][0][:, :, cur_iter]
            for kvhead_id in range(config.num_heads // config.kv_group_size):
                start_col = kvhead_id * config.kv_group_size
                end_col = (kvhead_id + 1) * config.kv_group_size
                for bid in block_ids[:, start_col:end_col].flatten().tolist():
                    important.add((kvhead_id, bid))
        shallow_important[cur_iter] = important

    for cur_iter in range(num_iters):
        for cur_layer in range(num_layers):
            block_ids = reader.traces[cur_layer][0][:, :, cur_iter]
            raw_scores = reader.traces[cur_layer][1][:, :, cur_iter]
            softmax_scores = reader._softmax(raw_scores, axis=0)

            # Compute next layer protected obj_ids
            protected = set()
            if cur_layer + 1 < num_layers:
                next_block_ids = reader.traces[cur_layer + 1][0][:, :, cur_iter]
                for kvhead_id in range(config.num_heads // config.kv_group_size):
                    start_col = kvhead_id * config.kv_group_size
                    end_col = (kvhead_id + 1) * config.kv_group_size
                    for bid in next_block_ids[:, start_col:end_col].flatten().tolist():
                        bsa = BsaKVCache(reader.seq_id, bid, cur_layer + 1, kvhead_id)
                        protected.add(bsa.get_obj_id())

            # Set hard protection before accessing this layer
            cache.protect(protected)

            for kvhead_id in range(config.num_heads // config.kv_group_size):
                start_col = kvhead_id * config.kv_group_size
                end_col = (kvhead_id + 1) * config.kv_group_size
                cur_kvhead_block_ids = block_ids[:, start_col:end_col]
                cur_kvhead_scores = softmax_scores[:, start_col:end_col]

                block_score_map = {}
                for head_offset in range(cur_kvhead_block_ids.shape[1]):
                    for top_idx in range(cur_kvhead_block_ids.shape[0]):
                        bid = int(cur_kvhead_block_ids[top_idx, head_offset])
                        score = float(cur_kvhead_scores[bid, head_offset])
                        block_score_map[bid] = max(block_score_map.get(bid, 0.0), score)

                last_block = reader._get_last_block_id()
                block_score_map[last_block] = block_score_map.get(last_block, 1.0)

                for block_id, score in block_score_map.items():
                    # Shallow layer boost
                    if cur_layer >= shallow_cutoff:
                        if (kvhead_id, block_id) in shallow_important[cur_iter]:
                            score += cache.shallow_boost

                    bsa = BsaKVCache(reader.seq_id, block_id, cur_layer, kvhead_id)
                    cache.access(bsa.get_obj_id(), BsaKVCache.bytes(), score)

            cache.clear_protection()

    return cache.get_miss_ratio()

def process_trace_with_belady(cache, reader: MobaTraceReader):
    """Process trace using Belady's optimal eviction algorithm."""
    from belady_cache import BeladyCache
    from kvcache import BsaKVCache
    config = sim_config()

    # Step 1: Generate full access sequence
    access_sequence = []
    for obj_id, obj_size, score in reader.generate_requests_with_scores():
        access_sequence.append((obj_id, obj_size))

    # Step 2: Precompute next_access_time for each position
    INF = len(access_sequence) + 1
    next_access = [INF] * len(access_sequence)
    last_seen = {}
    for i in range(len(access_sequence) - 1, -1, -1):
        obj_id = access_sequence[i][0]
        if obj_id in last_seen:
            next_access[i] = last_seen[obj_id]
        last_seen[obj_id] = i

    # Step 3: Simulate
    for t, (obj_id, obj_size) in enumerate(access_sequence):
        cache.access(obj_id, obj_size, next_access[t])

    return cache.get_miss_ratio()

def get_num_traces(config: SimConfig) -> int:
    return len(os.listdir(config.trace_dir))


def main():
    setup_logging()

    args = parse_args()
    config = sim_config()
    config.from_yaml(args.config)

    logger.info(f"Simulation config: {config}")

    num_traces = get_num_traces(config)
    total_req_miss_ratio = 0
    total_bytes_miss_ratio = 0
    use_custom = is_custom_algorithm(config.eviction_algorithm)
    
    for i in tqdm(range(num_traces), desc="Simulating traces"):
        trace_dir = os.path.join(config.trace_dir, f"trace{i:04d}")
        print(f"Processing trace {trace_dir}")
        reader = MobaTraceReader(trace_dir=trace_dir, verbose=config.verbose)
        cache = setup_cache(config)
        
        if use_custom:
            if config.eviction_algorithm.lower() == "proactive_eviction":
                req_miss_ratio, bytes_miss_ratio = process_trace_with_proactive(cache, reader)
            else:
                req_miss_ratio, bytes_miss_ratio = process_trace_with_momentum(cache, reader)
        else:
            req_miss_ratio, bytes_miss_ratio = cache.process_trace(reader)
        
        total_req_miss_ratio += req_miss_ratio
        total_bytes_miss_ratio += bytes_miss_ratio
        logger.info(f"Cache occupied {cache.get_occupied_byte()} bytes")
    
    logger.info(f"Average request miss ratio: {total_req_miss_ratio / num_traces}")
    logger.info(f"Average bytes miss ratio: {total_bytes_miss_ratio / num_traces}")


if __name__ == "__main__":
    main()
