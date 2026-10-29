import argparse
import json
import urllib.request


def run_non_stream(url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    print(parsed["choices"][0]["message"]["content"])


def run_stream(url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_part = line[len("data: ") :]
            if data_part == "[DONE]":
                break
            event = json.loads(data_part)
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content")
            if text:
                print(text, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local FastAPI OpenAI-like endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--prompt", default="Give me one short line about LLM inference.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    if args.stream:
        run_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)
    else:
        run_non_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
