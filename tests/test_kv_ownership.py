from inference import SamplingConfig
from inference.kv import KVBlockManager
from inference.types import RequestState


def request(tokens=(1, 2, 3, 4)):
    return RequestState.from_prompt("r", list(tokens), SamplingConfig(), 4, 999, 0)


def test_duplicate_insert_does_not_leak_block_references():
    manager = KVBlockManager(
        max_kv_cache_tokens=8,
        block_size=2,
        enable_prefix_cache=True,
        max_prefix_cache_entries=1,
    )
    first = request()
    assert manager.ensure_capacity(first, 4)
    manager.cache_prefix("default", first.prompt_token_ids, first.block_table)
    manager.cache_prefix("default", first.prompt_token_ids, first.block_table)
    manager.release(first)
    second = request((9, 8, 7, 6))
    assert manager.ensure_capacity(second, 4)
    manager.cache_prefix("default", second.prompt_token_ids, second.block_table)
    manager.release(second)
    assert manager.allocated_blocks == 2


def test_cache_pressure_reclaims_idle_pages():
    manager = KVBlockManager(
        max_kv_cache_tokens=4, block_size=2, enable_prefix_cache=True
    )
    first = request()
    assert manager.ensure_capacity(first, 4)
    manager.cache_prefix("default", first.prompt_token_ids, first.block_table)
    manager.release(first)
    assert manager.ensure_capacity(request((9, 8, 7, 6)), 4)


def test_exact_prefix_never_reuses_the_page_that_will_be_written():
    manager = KVBlockManager(
        max_kv_cache_tokens=8, block_size=2, enable_prefix_cache=True
    )
    first = request()
    manager.ensure_capacity(first, 4)
    manager.cache_prefix("default", first.prompt_token_ids, first.block_table)
    second = request()
    reused = manager.init_request_from_cache(second)
    assert reused == 2
    assert manager.ensure_capacity(second, 4)
    assert second.block_table[-1] != first.block_table[-1]


def test_eviction_preserves_live_request_references():
    manager = KVBlockManager(
        max_kv_cache_tokens=4, block_size=2, enable_prefix_cache=True
    )
    first = request()
    manager.ensure_capacity(first, 4)
    manager.cache_prefix("default", first.prompt_token_ids, first.block_table)
    assert not manager.ensure_capacity(request((9, 8)), 2)
    assert manager.allocated_blocks == 2
    manager.release(first)
    assert manager.allocated_blocks == 0
