from collections import deque
from typing import Deque, Dict, List, Optional

from inference.types import RequestState

from .prefix_cache import PrefixCache, PrefixCacheEntry


class KVBlockManager:
    def __init__(
        self,
        *,
        max_kv_cache_tokens: int,
        block_size: int,
        num_blocks: Optional[int] = None,
        enable_prefix_cache: bool = False,
        max_prefix_cache_entries: int = 64,
    ):
        if max_kv_cache_tokens <= 0:
            raise ValueError("max_kv_cache_tokens must be > 0")
        if block_size <= 0:
            raise ValueError("kv_block_size must be > 0")

        derived_blocks = max_kv_cache_tokens // block_size
        if num_blocks is None:
            num_blocks = derived_blocks
        if num_blocks <= 0:
            raise ValueError("num_kv_blocks must be > 0")

        self.max_kv_cache_tokens = int(max_kv_cache_tokens)
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self._free_block_ids: Deque[int] = deque(range(self.num_blocks))
        self._allocated_blocks = 0
        self._block_ref_counts: Dict[int, int] = {}
        self._prefix_cache: Optional[PrefixCache] = None
        if enable_prefix_cache:
            self._prefix_cache = PrefixCache(max_entries=max_prefix_cache_entries)

    @property
    def allocated_blocks(self) -> int:
        return self._allocated_blocks

    @property
    def prefix_cache_enabled(self) -> bool:
        return self._prefix_cache is not None

    @property
    def prefix_cache_size(self) -> int:
        return 0 if self._prefix_cache is None else self._prefix_cache.size

    def _required_blocks(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return (int(token_count) + self.block_size - 1) // self.block_size

    def _align_to_blocks(self, token_count: int) -> int:
        """Return the largest multiple of block_size not exceeding token_count."""
        if token_count <= 0:
            return 0
        return (int(token_count) // self.block_size) * self.block_size

    def _acquire_block(self, block_id: int) -> None:
        self._block_ref_counts[block_id] = self._block_ref_counts.get(block_id, 0) + 1
        if self._block_ref_counts[block_id] == 1:
            self._allocated_blocks += 1

    def _release_block(self, block_id: int) -> None:
        current = self._block_ref_counts.get(block_id, 0)
        if current <= 0:
            return
        current -= 1
        if current == 0:
            self._block_ref_counts.pop(block_id, None)
            self._free_block_ids.append(block_id)
            self._allocated_blocks = max(0, self._allocated_blocks - 1)
        else:
            self._block_ref_counts[block_id] = current

    def _allocate_new_blocks(self, count: int) -> List[int]:
        """Allocate ``count`` fresh blocks and increment their refcounts."""
        if count <= 0:
            return []
        if count > len(self._free_block_ids):
            raise RuntimeError("not enough free blocks")
        block_ids = [self._free_block_ids.popleft() for _ in range(count)]
        for block_id in block_ids:
            self._acquire_block(block_id)
        return block_ids

    def _prefix_match(
        self, scope: str, token_ids: List[int]
    ) -> Optional[PrefixCacheEntry]:
        if self._prefix_cache is None:
            return None
        return self._prefix_cache.match_longest(scope, token_ids)

    def init_request_from_cache(
        self, request: RequestState, scope: str = "default"
    ) -> int:
        """Try to seed ``request.block_table`` from the prefix cache.

        Returns the number of cached tokens that can be skipped.  The caller
        must set ``request.prompt_cursor`` and ``request.live_kv_tokens`` to
        this value.
        """
        if self._prefix_cache is None:
            return 0

        entry = self._prefix_match(scope, request.prompt_token_ids)
        if entry is None:
            return 0

        # Only whole blocks can be reused safely.
        # Leave a private final page to recompute logits. Never write a shared
        # page, including on an exact prompt hit (no copy-on-write needed).
        reusable_tokens = self._align_to_blocks(
            min(len(entry.token_ids), len(request.prompt_token_ids) - 1)
        )
        if reusable_tokens <= 0:
            return 0

        reusable_blocks = reusable_tokens // self.block_size
        block_ids = list(entry.block_ids[:reusable_blocks])
        for block_id in block_ids:
            self._acquire_block(block_id)
        self._prefix_cache.acquire(entry)

        request.block_table = block_ids
        return reusable_tokens

    def cache_prefix(
        self, scope: str, token_ids: List[int], block_table: List[int]
    ) -> None:
        """Cache a completed prompt prefix if prefix caching is enabled."""
        if self._prefix_cache is None:
            return

        cacheable_tokens = self._align_to_blocks(len(token_ids))
        if cacheable_tokens <= 0:
            return

        cacheable_blocks = cacheable_tokens // self.block_size
        block_ids = block_table[:cacheable_blocks]
        if len(block_ids) != cacheable_blocks:
            return

        entry, evicted = self._prefix_cache.insert(
            scope, token_ids[:cacheable_tokens], block_ids
        )
        if entry is not None:
            # The cache now holds a reference to these blocks.
            for block_id in block_ids:
                self._acquire_block(block_id)
        for victim in evicted:
            for block_id in victim.block_ids:
                self._release_block(block_id)

    def can_allocate_for(self, request: RequestState, total_tokens: int) -> bool:
        while True:
            existing_blocks = len(request.block_table)
            if existing_blocks == 0 and self._prefix_cache is not None:
                entry = self._prefix_cache.match_longest(
                    "default", request.prompt_token_ids
                )
                if entry is not None:
                    reusable = min(
                        len(entry.token_ids), len(request.prompt_token_ids) - 1
                    )
                    existing_blocks = self._align_to_blocks(reusable) // self.block_size
            if self._required_blocks(total_tokens) - existing_blocks <= len(
                self._free_block_ids
            ):
                return True
            # Re-evaluate the match after each eviction; it may have been evicted.
            if not self._evict_prefix():
                return False

    def _evict_prefix(self) -> bool:
        victim = None if self._prefix_cache is None else self._prefix_cache.evict_lru()
        if victim is None:
            return False
        for block_id in victim.block_ids:
            self._release_block(block_id)
        return True

    def ensure_capacity(self, request: RequestState, total_tokens: int) -> bool:
        required_blocks = self._required_blocks(total_tokens)
        additional_blocks = required_blocks - len(request.block_table)
        if additional_blocks <= 0:
            return True
        while additional_blocks > len(self._free_block_ids):
            if not self._evict_prefix():
                return False

        new_blocks = self._allocate_new_blocks(additional_blocks)
        request.block_table.extend(new_blocks)
        return True

    def release(self, request: RequestState) -> None:
        if not request.block_table:
            return
        for block_id in request.block_table:
            self._release_block(block_id)
        request.block_table.clear()

    def clear_prefix_cache(self) -> None:
        if self._prefix_cache is not None:
            for entry in self._prefix_cache.clear():
                for block_id in entry.block_ids:
                    self._release_block(block_id)
