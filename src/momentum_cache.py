"""
MomentumDecay Cache Eviction Algorithm

This cache eviction algorithm tracks the moving momentum of attention scores
for each block. The momentum is updated as:
    x_t = beta * x_{t-1} + s_t

where:
    - x_t is the momentum score at time t
    - beta is the decay factor (default 0.9)
    - s_t is the normalized softmax score for this block

During eviction, the block with the lowest momentum score is evicted.
"""

import heapq
from typing import Dict, Tuple, Optional


class MomentumDecayCache:
    """
    Cache with momentum-decay based eviction.
    
    Evicts blocks with the lowest accumulated momentum score.
    Score update rule: x_t = beta * x_{t-1} + s_t
    
    Uses a min-heap with lazy deletion for O(log n) eviction.
    """
    
    def __init__(self, cache_size: int, beta: float = 0.9):
        """
        Args:
            cache_size: Maximum cache size in bytes
            beta: Decay factor for momentum (default 0.9)
        """
        self.cache_size = cache_size
        self.beta = beta
        
        # obj_id -> (momentum_score, obj_size, version)
        self.cache: Dict[int, Tuple[float, int, int]] = {}
        
        # Min-heap of (momentum_score, version, obj_id)
        # We use version to handle lazy deletion when scores are updated
        self.heap: list = []
        
        self.occupied_bytes = 0
        # Optional set of pinned object IDs: these objects should not be evicted
        self.pinned_objs = set()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0
    
    def _evict_one(self) -> Optional[int]:
        """Evict the entry with the lowest momentum score. O(log n) amortized."""
        while self.heap:
            score, version, obj_id = heapq.heappop(self.heap)
            
            # Check if this entry is still valid (not stale)
            if obj_id in self.cache:
                cached_score, obj_size, cached_version = self.cache[obj_id]
                if version == cached_version:
                    # Skip eviction if obj is pinned
                    if obj_id in self.pinned_objs:
                        continue
                    # Valid entry - evict it
                    del self.cache[obj_id]
                    self.occupied_bytes -= obj_size
                    return obj_id
            # Stale entry - continue to next
        
        return None
    
    def _ensure_capacity(self, required_size: int):
        """Evict entries until we have enough space."""
        # If there are only pinned objects that occupy the cache, we won't be able
        # to free up space by evicting pinned objects, so we guard against infinite loop.
        while self.occupied_bytes + required_size > self.cache_size and self.cache:
            self._evict_one()
    
    def access(self, obj_id: int, obj_size: int, score: float = 0.0) -> bool:
        """
        Access an object in the cache.
        
        Args:
            obj_id: Unique identifier for the object
            obj_size: Size of the object in bytes
            score: The normalized softmax score for this access
            
        Returns:
            True if cache hit, False if cache miss
        """
        self.total_requests += 1
        
        if obj_id in self.cache:
            # Cache hit - update momentum score
            old_score, obj_size, old_version = self.cache[obj_id]
            new_score = self.beta * old_score + score
            new_version = old_version + 1
            
            # Update cache entry with new version
            self.cache[obj_id] = (new_score, obj_size, new_version)
            
            # Push new entry to heap (old one becomes stale)
            heapq.heappush(self.heap, (new_score, new_version, obj_id))
            
            self.hits += 1
            self.bytes_hits += obj_size
            return True
        else:
            # Cache miss - need to insert
            self.misses += 1
            self.bytes_misses += obj_size
            
            # Ensure we have capacity
            self._ensure_capacity(obj_size)
            
            # Insert new entry with version 0
            self.cache[obj_id] = (score, obj_size, 0)
            heapq.heappush(self.heap, (score, 0, obj_id))
            self.occupied_bytes += obj_size
            
            return False

    def pin(self, obj_id: int):
        """Pin an object so it won't be evicted by the cache.

        Returns True if the object was already in cache or was added as a placeholder.
        """
        self.pinned_objs.add(obj_id)
        # If it's not in the cache, we can optionally add it with zero size placeholder
        if obj_id not in self.cache:
            # We can't know the obj_size here; caller should ensure it's added via access
            # We'll create a placeholder entry with score = infinity to prevent eviction
            self.cache[obj_id] = (float('inf'), 0, 0)
            return True
        return True

    def unpin(self, obj_id: int):
        if obj_id in self.pinned_objs:
            self.pinned_objs.remove(obj_id)
            return True
        return False
    
    def get_miss_ratio(self) -> Tuple[float, float]:
        """
        Returns (request_miss_ratio, bytes_miss_ratio)
        """
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
        """Reset hit/miss statistics."""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0
