from collections import OrderedDict
from typing import Dict, Tuple, Optional


class ORCACache:
    def __init__(self, cache_size: int, window: int = 3):
        self.cache_size = cache_size
        self.window = window

        self._step = 0

        self._cache: Dict[int, Tuple[int, int]] = {}
        self._cold_lru: OrderedDict[int, None] = OrderedDict()

        self.occupied_bytes = 0

        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def _is_hot(self, last_access_step: int) -> bool:
        return (self._step - last_access_step) <= self.window

    def _promote_to_hot(self, obj_id: int):
        self._cold_lru.pop(obj_id, None)

    def _demote_to_cold(self, obj_id: int):
        if obj_id in self._cache:
            self._cold_lru[obj_id] = None
            self._cold_lru.move_to_end(obj_id)

    def _evict_one(self) -> Optional[int]:
        for obj_id in list(self._cold_lru.keys()):
            if obj_id in self._cache:
                obj_size, _ = self._cache.pop(obj_id)
                self._cold_lru.pop(obj_id, None)
                self.occupied_bytes -= obj_size
                return obj_id
            else:
                self._cold_lru.pop(obj_id, None)
        if self._cache:
            lru_id = min(self._cache, key=lambda oid: self._cache[oid][1])
            obj_size, _ = self._cache.pop(lru_id)
            self._cold_lru.pop(lru_id, None)
            self.occupied_bytes -= obj_size
            return lru_id

        return None

    def _ensure_capacity(self, required_size: int):
        while self.occupied_bytes + required_size > self.cache_size and self._cache:
            evicted = self._evict_one()
            if evicted is None:
                break

    def _refresh_cold_tier(self):
        for obj_id, (obj_size, last_step) in self._cache.items():
            if not self._is_hot(last_step) and obj_id not in self._cold_lru:
                self._cold_lru[obj_id] = None
                self._cold_lru.move_to_end(obj_id, last=False)

    def access(self, obj_id: int, obj_size: int, score: float = 0.0) -> bool:
        self.total_requests += 1
        self._step += 1

        self._refresh_cold_tier()

        if obj_id in self._cache:
            obj_size_cached, _ = self._cache[obj_id]
            self._cache[obj_id] = (obj_size_cached, self._step)
            self._promote_to_hot(obj_id)
            self.hits += 1
            self.bytes_hits += obj_size_cached
            return True

        self.misses += 1
        self.bytes_misses += obj_size

        self._ensure_capacity(obj_size)

        self._cache[obj_id] = (obj_size, self._step)
        self.occupied_bytes += obj_size

        return False

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
        return len(self._cache)

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def get_hot_count(self) -> int:
        return sum(
            1 for _, last_step in self._cache.values()
            if self._is_hot(last_step)
        )

    def get_cold_count(self) -> int:
        return len(self._cold_lru)