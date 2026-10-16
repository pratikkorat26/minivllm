"""Run identical streaming chat workloads against this server or vLLM (stdlib only)."""

import argparse
import asyncio
import json
import time
import urllib.request
from pathlib import Path


def request_one(args, index):
    prompt = [
        "Explain KV caching in one paragraph.",
        "Explain continuous batching.",
        "Describe token budget scheduling.",
        "Describe paged attention.",
    ][index % 4]
    payload = dict(
        model=args.model,
        messages=[dict(role="user", content=prompt)],
        max_tokens=args.tokens,
        temperature=0,
        top_p=1,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    # This engine accepts these sampling extensions; vLLM also supports them.
    payload.update(top_k=0, repetition_penalty=1)
    req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    chunks = 0
    tokens = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                tokens = event["usage"]["completion_tokens"]
            if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                last = time.perf_counter()
                first = first or last
                chunks += 1
    elapsed = time.perf_counter() - start
    if tokens is None:
        raise RuntimeError(
            "Server must provide usage; text chunk counts are not token counts"
        )
    return dict(
        prompt=prompt,
        completion_tokens=tokens,
        content_chunks=chunks,
        ttft_ms=None if first is None else (first - start) * 1000,
        tpot_ms=None
        if tokens < 2 or chunks < 2
        else (last - first) * 1000 / (tokens - 1),
        latency_ms=elapsed * 1000,
    )


async def run(args):
    await asyncio.to_thread(request_one, args, 0)
    rows = []
    for concurrency in args.concurrency:
        requests = []
        start = time.perf_counter()
        for _ in range(args.repeats):
            requests.extend(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(request_one, args, i)
                        for i in range(concurrency)
                    )
                )
            )
        elapsed = time.perf_counter() - start
        rows.append(
            dict(
                concurrency=concurrency,
                elapsed_s=elapsed,
                output_tokens_per_s=sum(r["completion_tokens"] for r in requests)
                / elapsed,
                requests=requests,
            )
        )
    result = dict(
        kind="http_chat_closed_loop",
        url=args.url,
        model=args.model,
        label=args.label,
        tokens=args.tokens,
        repeats=args.repeats,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gemma-3-270m-it")
    parser.add_argument("--label", default="minivllm")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.repeats, *args.concurrency) <= 0:
        parser.error("tokens, repeats and concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
