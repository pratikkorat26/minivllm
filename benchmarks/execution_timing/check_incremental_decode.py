"""Check native incremental decoding against full prefixes and time CPU work."""

import json
from pathlib import Path
import statistics
import time

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

ROOT = Path(__file__).resolve().parents[2]


def main():
    tokenizer = Tokenizer.from_file(str(ROOT / "LFM3-1.7B/tokenizer.json"))
    cases = [
        tokenizer.encode(text).ids
        for text in (
            " Hello world!  café 👩🏽‍💻 中文 日本語 مرحبا\n\n```python\nprint(42)\n```",
            "<|im_start|>assistant\n<think>\nTesting 😊\n</think>\n<|im_end|>",
        )
    ]
    for path in sorted((ROOT / "benchmarks/quality_assessment").glob("*greedy.json")):
        cases.extend(row["token_ids"] for row in json.loads(path.read_text())["rows"])
    checked = 0
    for ids in cases:
        decoder, text = DecodeStream(skip_special_tokens=False), ""
        for i, token in enumerate(ids):
            text += decoder.step(tokenizer, token) or ""
            expected = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
            if not expected.endswith("\ufffd"):
                assert text == expected, (i, text, expected)
                checked += 1
    timings = []
    source = tokenizer.encode("Writing readable software takes care. " * 1000).ids
    for length in (64, 512, 2048):
        ids, full, incremental = source[:length], [], []
        for _ in range(5):
            start = time.perf_counter()
            for i in range(length):
                tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
            full.append(time.perf_counter() - start)
            decoder = DecodeStream(skip_special_tokens=False)
            start = time.perf_counter()
            for token in ids:
                decoder.step(tokenizer, token)
            incremental.append(time.perf_counter() - start)
        timings.append(
            dict(
                tokens=length,
                full_prefix_s=statistics.median(full),
                incremental_s=statistics.median(incremental),
            )
        )
    report = dict(
        kind="cpu_tokenizer_microbenchmark",
        cases=len(cases),
        matching_valid_utf8_prefixes=checked,
        timings=timings,
        caveat="Tokenizer time only; not model or serving throughput.",
    )
    output = ROOT / "benchmarks/overlap_optimization/incremental_decode.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
