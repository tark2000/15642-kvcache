from typing import Dict, Tuple, Optional, List
from collections import OrderedDict


class RadixNode:
    def __init__(self, obj_id: int, obj_size: int):
        self.obj_id = obj_id
        self.obj_size = obj_size
        self.children: Dict[int, "RadixNode"] = {}
        self.parent: Optional["RadixNode"] = None
        self.last_access_time: int = 0
        self.ref_count: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class SGLangCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        self.nodes: Dict[int, RadixNode] = {}

        self._root = RadixNode(obj_id=-1, obj_size=0)
        self._root.last_access_time = 0

        self.occupied_bytes = 0
        self._time = 0

        self._lru_leaves: OrderedDict[int, None] = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def _touch(self, node: RadixNode):
        self._time += 1
        node.last_access_time = self._time
        if node.is_leaf() and node.obj_id != -1:
            self._lru_leaves.move_to_end(node.obj_id)

    def _register_leaf(self, node: RadixNode):
        if node.obj_id != -1:
            self._lru_leaves[node.obj_id] = None

    def _unregister_leaf(self, node: RadixNode):
        self._lru_leaves.pop(node.obj_id, None)

    def _evict_one(self) -> Optional[int]:
        for obj_id in list(self._lru_leaves.keys()):
            node = self.nodes.get(obj_id)
            if node is None:
                self._lru_leaves.pop(obj_id, None)
                continue
            if node.ref_count > 0:
                continue
            if not node.is_leaf():
                self._unregister_leaf(node)
                continue

            self._unregister_leaf(node)
            parent = node.parent
            if parent is not None:
                parent.children.pop(obj_id, None)
                if parent.is_leaf() and parent.obj_id != -1:
                    self._register_leaf(parent)
                    self._lru_leaves.move_to_end(
                        parent.obj_id,
                        last=False
                    )

            self.occupied_bytes -= node.obj_size
            del self.nodes[obj_id]
            return obj_id

        return None

    def _ensure_capacity(self, required_size: int):
        while self.occupied_bytes + required_size > self.cache_size and self.nodes:
            evicted = self._evict_one()
            if evicted is None:
                break

    def access(
        self,
        obj_id: int,
        obj_size: int,
        parent_id: Optional[int] = None,
    ) -> bool:
        
        self.total_requests += 1

        if obj_id in self.nodes:
            node = self.nodes[obj_id]
            self._touch(node)
            self.hits += 1
            self.bytes_hits += obj_size
            return True

        self.misses += 1
        self.bytes_misses += obj_size

        self._ensure_capacity(obj_size)

        if parent_id is not None and parent_id in self.nodes:
            parent_node = self.nodes[parent_id]
        else:
            parent_node = self._root

        self._time += 1
        node = RadixNode(obj_id=obj_id, obj_size=obj_size)
        node.parent = parent_node
        node.last_access_time = self._time

        if parent_node.is_leaf() and parent_node.obj_id != -1:
            self._unregister_leaf(parent_node)

        parent_node.children[obj_id] = node
        self.nodes[obj_id] = node
        self.occupied_bytes += obj_size

        self._register_leaf(node)

        return False

    def pin(self, obj_id: int) -> bool:
        if obj_id in self.nodes:
            self.nodes[obj_id].ref_count += 1
            return True
        return False

    def unpin(self, obj_id: int) -> bool:
        if obj_id in self.nodes:
            node = self.nodes[obj_id]
            node.ref_count = max(0, node.ref_count - 1)
            if node.ref_count == 0 and node.is_leaf():
                self._register_leaf(node)
            return True
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
        return len(self.nodes)

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.bytes_hits = 0
        self.bytes_misses = 0

    def get_tree_depth(self) -> int:
        def _depth(node: RadixNode) -> int:
            if not node.children:
                return 0
            return 1 + max(_depth(c) for c in node.children.values())
        return _depth(self._root)

    def get_n_leaves(self) -> int:
        return len(self._lru_leaves)