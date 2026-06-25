#!/usr/bin/env python3
"""Issue #656 client: text request (warms/stores cache), then a video request
(long context -> sliding window slides -> SWA group offload).

Run after serve.sh is up:  python3 client.py
Optionally bump --concurrency to make stale-block interior holes more likely.
"""
import argparse
import concurrent.futures
import time

import requests

MODEL = "google/gemma-4-31B-it"
# Short clip from the HF VLM test set (same as the issue). Needs internet.
VIDEO_URL = (
    "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"
)


def post(api, payload):
    t = time.time()
    r = requests.post(api, json=payload, timeout=600)
    dt = time.time() - t
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:500]} ({dt:.1f}s)"
    j = r.json()
    u = j.get("usage", {})
    return (
        f"OK ({dt:.1f}s) prompt_tokens={u.get('prompt_tokens')} "
        f"completion_tokens={u.get('completion_tokens')}\n"
        f"  {j['choices'][0]['message']['content'][:200]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel video requests (raises eviction pressure)")
    args = ap.parse_args()
    api = f"{args.host}/v1/chat/completions"

    text_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello, please describe yourself."}],
        "max_tokens": 100,
        "temperature": 0.1,
    }
    video_payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is happening in the video?"},
                {"type": "video_url", "video_url": {"url": VIDEO_URL}},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    print("== text request ==")
    print(post(api, text_payload))

    print(f"\n== {args.concurrency} video request(s) ==")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for out in ex.map(lambda _: post(api, video_payload), range(args.concurrency)):
            print(out)


if __name__ == "__main__":
    main()
