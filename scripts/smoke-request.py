#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path

from script_utils import atomic_json


# Send one deterministic chat request and persist the response.
def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the vLLM chat endpoint")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "In one sentence, explain expert parallelism.",
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
            "reasoning_effort": "low",
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)

    choices = result.get("choices")
    if not choices or not choices[0].get("message"):
        raise RuntimeError(f"Unexpected completion response: {result}")

    atomic_json(args.output, result)
    print(choices[0]["message"].get("content", ""))


if __name__ == "__main__":
    main()
