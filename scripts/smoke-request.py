#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path

from script_utils import atomic_json



def build_repeated_prompt(
    target_tokens: int,
    token_count,
    unit: str = "x ",
) -> tuple[str, int]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")

    low = 0
    high = target_tokens
    while token_count(unit * high) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(unit * middle) <= target_tokens:
            low = middle
        else:
            high = middle

    candidates = ((low, token_count(unit * low)), (high, token_count(unit * high)))
    repetitions, observed_tokens = min(
        candidates,
        key=lambda candidate: (
            abs(candidate[1] - target_tokens),
            candidate[1] > target_tokens,
        ),
    )
    if observed_tokens > target_tokens:
        repetitions = low
        observed_tokens = token_count(unit * repetitions)
    return unit * repetitions, observed_tokens

# Send one deterministic chat request and persist the response.
def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the vLLM chat endpoint")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    content = "In one sentence, explain expert parallelism."
    effective_input_tokens = None
    if args.input_tokens is not None:
        tokenize_url = f"{args.base_url.rstrip('/')}/tokenize"

        def chat_token_count(candidate: str) -> int:
            tokenize_payload = json.dumps(
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": candidate}],
                    "add_generation_prompt": True,
                    "add_special_tokens": False,
                }
            ).encode()
            tokenize_request = urllib.request.Request(
                tokenize_url,
                data=tokenize_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(tokenize_request, timeout=300) as response:
                tokenized = json.load(response)
            tokens = tokenized.get("tokens")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) for token in tokens
            ):
                raise RuntimeError(
                    f"Unexpected tokenize response from {tokenize_url}: {tokenized}"
                )
            return len(tokens)

        content, effective_input_tokens = build_repeated_prompt(
            args.input_tokens,
            chat_token_count,
        )
        if effective_input_tokens < args.input_tokens - 1:
            raise RuntimeError(
                "Could not construct the requested context length: "
                f"target={args.input_tokens}, observed={effective_input_tokens}"
            )

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "reasoning_effort": "low",
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    request_timeout = 7200 if args.input_tokens is not None else 300
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        result = json.load(response)

    choices = result.get("choices")
    if not choices or not choices[0].get("message"):
        raise RuntimeError(f"Unexpected completion response: {result}")

    if effective_input_tokens is not None:
        result["_smoke"] = {
            "requested_input_tokens": args.input_tokens,
            "effective_input_tokens": effective_input_tokens,
            "max_output_tokens": args.max_tokens,
        }
    atomic_json(args.output, result)
    print(choices[0]["message"].get("content", ""))


if __name__ == "__main__":
    main()
