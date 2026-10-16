import unittest

from inference.kv.prefix_cache import PrefixCache


class PrefixCacheTests(unittest.TestCase):
    def test_match_returns_none_when_empty(self):
        cache = PrefixCache()
        self.assertIsNone(cache.match_longest("default", [1, 2, 3]))

    def test_exact_match(self):
        cache = PrefixCache()
        cache.insert("default", [1, 2, 3], [10, 11, 12])
        entry = cache.match_longest("default", [1, 2, 3])
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token_ids, (1, 2, 3))
        self.assertEqual(entry.block_ids, (10, 11, 12))

    def test_longest_partial_match(self):
        cache = PrefixCache()
        cache.insert("default", [1, 2], [10, 11])
        cache.insert("default", [1, 2, 3], [10, 11, 12])
        cache.insert("default", [1, 2, 3, 4], [10, 11, 12, 13])

        entry = cache.match_longest("default", [1, 2, 3, 4, 5])
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token_ids, (1, 2, 3, 4))
        self.assertEqual(entry.block_ids, (10, 11, 12, 13))

    def test_shorter_match_when_longer_does_not_apply(self):
        cache = PrefixCache()
        cache.insert("default", [1, 2, 3], [10, 11, 12])
        cache.insert("default", [1, 2, 3, 4], [10, 11, 12, 13])

        entry = cache.match_longest("default", [1, 2, 3, 99])
        self.assertIsNotNone(entry)
        self.assertEqual(entry.token_ids, (1, 2, 3))

    def test_scope_isolation(self):
        cache = PrefixCache()
        cache.insert("default", [1, 2, 3], [10, 11, 12])
        cache.insert("tenant-a", [1, 2, 3], [20, 21, 22])

        default_entry = cache.match_longest("default", [1, 2, 3, 4])
        tenant_entry = cache.match_longest("tenant-a", [1, 2, 3, 4])

        self.assertEqual(default_entry.block_ids, (10, 11, 12))
        self.assertEqual(tenant_entry.block_ids, (20, 21, 22))

    def test_insert_duplicate_does_not_duplicate(self):
        cache = PrefixCache(max_entries=2)
        first, _ = cache.insert("default", [1, 2], [10, 11])
        second, _ = cache.insert("default", [1, 2], [99, 99])
        self.assertEqual(first.block_ids, (10, 11))
        self.assertIsNone(second)
        self.assertEqual(len(cache), 1)

    def test_eviction_when_full(self):
        cache = PrefixCache(max_entries=2)
        _, evicted = cache.insert("default", [1], [10])
        self.assertEqual(evicted, [])
        _, evicted = cache.insert("default", [2], [20])
        self.assertEqual(evicted, [])
        _, evicted = cache.insert("default", [3], [30])
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0].token_ids, (1,))
        self.assertEqual(len(cache), 2)

    def test_eviction_updates_last_accessed(self):
        cache = PrefixCache(max_entries=2)
        entry1, _ = cache.insert("default", [1], [10])
        entry2, _ = cache.insert("default", [2], [20])
        # Touch entry 1 so it is more recently used than entry 2.
        cache.acquire(cache.match_longest("default", [1]))
        _, evicted = cache.insert("default", [3], [30])
        self.assertEqual(evicted[0].token_ids, (2,))
        self.assertIs(cache.match_longest("default", [1]), entry1)
        self.assertIsNone(cache.match_longest("default", [2]))

    def test_zero_max_entries_disables_cache(self):
        cache = PrefixCache(max_entries=0)
        entry, _ = cache.insert("default", [1, 2], [10, 11])
        self.assertIsNone(entry)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.match_longest("default", [1, 2]))

    def test_clear_removes_all_entries(self):
        cache = PrefixCache()
        cache.insert("default", [1], [10])
        cache.clear()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
