from typing import Dict, Tuple


class BeladyCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cache: Dict[int, Tuple[int, int]] = {}
        self.occupied_bytes = 0
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def _evict_one(self):
        if not self.cache:
            return
        victim_id = max(self.cache.keys(), key=lambda oid: self.cache[oid][1])
        obj_size = self.cache[victim_id][0]
        del self.cache[victim_id]
        self.occupied_bytes -= obj_size

    def _ensure_capacity(self, required_size: int):
        while self.occupied_bytes + required_size > self.cache_size and self.cache:
            self._evict_one()

    def access(self, obj_id: int, obj_size: int, next_access_time: int) -> bool:
        self.total_requests += 1
        if obj_id in self.cache:
            self.cache[obj_id] = (obj_size, next_access_time)
            self.hits += 1
            self.bytes_hits += obj_size
            return True
        else:
            self.misses += 1
            self.bytes_misses += obj_size
            self._ensure_capacity(obj_size)
            self.cache[obj_id] = (obj_size, next_access_time)
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

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0