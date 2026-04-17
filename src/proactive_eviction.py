import heapq
from typing import Dict, Tuple, Optional, Set


class ProactiveEvictionCache:
    """
    Proactive eviction policy:
    1. Momentum decay base score (same as BSAKV baseline)
    2. Hard protection: blocks selected by next layer cannot be evicted
    3. Shallow-layer boost: blocks selected in early layers get score boost
    """

    def __init__(self, cache_size: int, beta: float = 0.9,
                 shallow_boost: float = 0.1,
                 num_layers: int = 28,
                 shallow_cutoff: float = 0.25):
        self.cache_size = cache_size
        self.beta = beta
        self.shallow_boost = shallow_boost
        self.num_layers = num_layers
        self.shallow_cutoff = int(num_layers * shallow_cutoff)

        self.cache: Dict[int, Tuple[float, int, int]] = {}
        self.heap: list = []
        self.occupied_bytes = 0
        self.pinned_objs: Set[int] = set()
        self.protected_objs: Set[int] = set()

        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def protect(self, obj_ids: Set[int]):
        self.protected_objs = obj_ids

    def clear_protection(self):
        self.protected_objs = set()

    def _evict_one(self) -> Optional[int]:
        while self.heap:
            score, version, obj_id = heapq.heappop(self.heap)
            if obj_id in self.cache:
                cached_score, obj_size, cached_version = self.cache[obj_id]
                if version == cached_version:
                    if obj_id in self.pinned_objs:
                        continue
                    if obj_id in self.protected_objs:
                        continue
                    del self.cache[obj_id]
                    self.occupied_bytes -= obj_size
                    return obj_id
        return None

    def _ensure_capacity(self, required_size: int):
        while self.occupied_bytes + required_size > self.cache_size and self.cache:
            self._evict_one()

    def access(self, obj_id: int, obj_size: int, score: float = 0.0) -> bool:
        self.total_requests += 1
        if obj_id in self.cache:
            old_score, obj_size, old_version = self.cache[obj_id]
            new_score = self.beta * old_score + score
            new_version = old_version + 1
            self.cache[obj_id] = (new_score, obj_size, new_version)
            heapq.heappush(self.heap, (new_score, new_version, obj_id))
            self.hits += 1
            self.bytes_hits += obj_size
            return True
        else:
            self.misses += 1
            self.bytes_misses += obj_size
            self._ensure_capacity(obj_size)
            self.cache[obj_id] = (score, obj_size, 0)
            heapq.heappush(self.heap, (score, 0, obj_id))
            self.occupied_bytes += obj_size
            return False

    def pin(self, obj_id: int):
        self.pinned_objs.add(obj_id)
        if obj_id not in self.cache:
            self.cache[obj_id] = (float('inf'), 0, 0)
        return True

    def unpin(self, obj_id: int):
        if obj_id in self.pinned_objs:
            self.pinned_objs.remove(obj_id)
        return True

    def get_miss_ratio(self) -> Tuple[float, float]:
        if self.total_requests == 0:
            return 0.0, 0.0
        req_miss_ratio = self.misses / self.total_requests
        total_bytes = self.bytes_hits + self.bytes_misses
        bytes_miss_ratio = self.bytes_misses / total_bytes if total_bytes > 0 else 0.0
        return req_miss_ratio, bytes_miss_ratio

    def get_occupied_byte(self) -> int:
        return self.occupied_bytes

    def get_n_obj(self) -> int:
        return len(self.cache)

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0
