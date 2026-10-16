import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PrefixCacheEntry:
    """A single cached prefix.

    The ``scope`` field is the tenant-isolation seam: today it is always
    ``"default"``; in the future it can be set to a tenant identifier so
    different tenants never share KV blocks.
    """

    scope: str
    token_ids: Tuple[int, ...]
    block_ids: Tuple[int, ...]
    last_accessed: float = field(default_factory=time.perf_counter)


class PrefixCache:
    """In-memory prefix cache keyed by (scope, token_ids).

    The cache is deliberately block-size agnostic.  Callers are responsible
    for inserting and reusing only whole blocks.  This keeps the cache core
    simple and makes it easy to add tenant isolation later by changing only
    the ``scope`` argument.
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max(max_entries, 0)
        self._entries: Dict[str, PrefixCacheEntry] = {}

    def _hash(self, scope: str, token_ids: Tuple[int, ...]) -> str:
        hasher = hashlib.sha256()
        hasher.update(scope.encode("utf-8"))
        hasher.update(b"\x00")
        for token in token_ids:
            hasher.update(token.to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def match_longest(
        self, scope: str, token_ids: List[int]
    ) -> Optional[PrefixCacheEntry]:
        """Return the cached entry with the longest matching prefix.

        The matching entry is guaranteed to have ``entry.token_ids`` as a
        prefix of ``token_ids`` and the same ``scope``.
        """
        if self.max_entries == 0 or not self._entries:
            return None

        token_tuple = tuple(token_ids)
        best_entry: Optional[PrefixCacheEntry] = None
        best_len = 0

        for entry in self._entries.values():
            if entry.scope != scope:
                continue
            entry_len = len(entry.token_ids)
            if entry_len > len(token_tuple):
                continue
            if token_tuple[:entry_len] != entry.token_ids:
                continue
            if entry_len > best_len:
                best_len = entry_len
                best_entry = entry

        if best_entry is not None:
            best_entry.last_accessed = time.perf_counter()

        return best_entry

    def insert(
        self, scope: str, token_ids: List[int], block_ids: List[int]
    ) -> Tuple[Optional[PrefixCacheEntry], List[PrefixCacheEntry]]:
        """Insert a prefix into the cache, evicting old entries if needed.

        Returns:
            A tuple of (new_entry_or_none, evicted_entries).  Callers
            must release the physical blocks belonging to ``evicted_entries``.
        """
        evicted: List[PrefixCacheEntry] = []
        if self.max_entries == 0:
            return None, evicted

        token_tuple = tuple(token_ids)
        block_tuple = tuple(block_ids)
        key = self._hash(scope, token_tuple)

        existing = self._entries.get(key)
        if existing is not None:
            existing.last_accessed = time.perf_counter()
            # No new ownership was created. Callers must not acquire again.
            return None, evicted

        while len(self._entries) >= self.max_entries:
            victim = self.evict_lru()
            if victim is None:
                break
            evicted.append(victim)

        if len(self._entries) >= self.max_entries:
            # No entry can be inserted. Active pages remain protected by block refs.
            return None, evicted

        entry = PrefixCacheEntry(
            scope=scope,
            token_ids=token_tuple,
            block_ids=block_tuple,
        )
        self._entries[key] = entry
        return entry, evicted

    def evict_lru(self) -> Optional[PrefixCacheEntry]:
        if not self._entries:
            return None
        oldest_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
        return self._entries.pop(oldest_key)

    def acquire(self, entry: PrefixCacheEntry) -> None:
        """Mark an entry as recently used."""
        entry.last_accessed = time.perf_counter()

    def clear(self) -> List[PrefixCacheEntry]:
        """Remove all entries.  Callers must release returned block references."""
        entries = list(self._entries.values())
        self._entries.clear()
        return entries
