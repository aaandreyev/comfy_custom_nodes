#!/usr/bin/env python3
"""Pre-compile torch.compile shape caches for every crop bucket of a workflow.

Run ON THE POD after ComfyUI is up, pointing at the same API-format workflow
the backend submits. For each bucket (W, H) the script uploads a dummy RGBA
image of exactly that size (alpha-cut center rectangle = inpaint mask, like a
clipspace painted-masked image), patches the LoadImage node and executes the
workflow once. The first pass per shape pays the inductor compilation; every
later real request in that bucket runs from cache.

Usage:
  python warmup_compile_buckets.py --workflow inpaint_api.json --dry-run
  python warmup_compile_buckets.py --workflow inpaint_api.json --steps 1
  python warmup_compile_buckets.py --workflow inpaint_api.json --quick
  python warmup_compile_buckets.py --workflow inpaint_api.json --host https://xxx.pinggy-free.link

The bucket list must match ZeroDriftInpaintCrop settings (size_bucket_px /
bucket_min_px): sides default to 512..2048 step 128-ish grid below.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.request
import uuid
import zlib

_UA = {"User-Agent": "warmup-compile-buckets/1.0"}


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "https://" + host
    return host


def _get_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers=dict(_UA))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_shapes(sides: list[int], aspect_limit: float, strip_side: int, quick: bool) -> list[tuple[int, int]]:
    shapes = []
    for w in sides:
        for h in sides:
            aspect_ok = max(w, h) / min(w, h) <= aspect_limit
            strip_ok = strip_side in (w, h)
            square = w == h
            if quick:
                if square or strip_ok:
                    shapes.append((w, h))
            elif aspect_ok or strip_ok:
                shapes.append((w, h))
    return shapes


def dummy_rgba_png(width: int, height: int) -> bytes:
    """Gray RGBA PNG with a transparent center rectangle (the inpaint mask)."""
    x0, x1 = int(width * 0.07), int(width * 0.93)
    y0, y1 = int(height * 0.07), int(height * 0.93)
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            g = 96 + ((x * 7 + y * 13) % 64)
            a = 0 if (y0 <= y < y1 and x0 <= x < x1) else 255
            rows += bytes((g, g, g, a))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))


def upload_png(host: str, name: str, payload: bytes) -> str:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{host}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **_UA},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["name"]


def submit_and_wait(host: str, prompt: dict, timeout_s: float) -> float:
    payload = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(f"{host}/prompt", data=payload,
                                 headers={"Content-Type": "application/json", **_UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        prompt_id = json.loads(resp.read())["prompt_id"]
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        history = _get_json(f"{host}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"prompt {prompt_id} failed: {json.dumps(status)[:500]}")
            if status.get("completed") or entry.get("outputs"):
                return time.monotonic() - started
        time.sleep(2.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--sides", default="512,640,768,896,1024,1152,1280,1408,1536")
    ap.add_argument("--aspect-limit", type=float, default=2.0)
    ap.add_argument("--strip-side", type=int, default=1536,
                    help="Always include WxH combos containing this side (tile-edge strips).")
    ap.add_argument("--quick", action="store_true",
                    help="Warm only squares and strip combos (fast subset).")
    ap.add_argument("--steps", type=int, default=1, help="Sampler steps override for warmup runs.")
    ap.add_argument("--batch", type=int, default=1,
                    help="Latent batch size to warm (torch.compile caches per batch size too). "
                         "Sets `amount` on a RepeatLatentBatch feeding the sampler, inserting one if absent.")
    ap.add_argument("--timeout", type=float, default=900.0, help="Per-run timeout, seconds.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sides = sorted({int(s) for s in args.sides.split(",") if s.strip()})
    shapes = build_shapes(sides, args.aspect_limit, args.strip_side, args.quick)
    shapes.sort(key=lambda wh: wh[0] * wh[1])
    print(f"buckets to warm: {len(shapes)}")
    print(" ".join(f"{w}x{h}" for w, h in shapes))
    if args.dry_run:
        return

    host = _normalize_host(args.host)
    stats = _get_json(f"{host}/system_stats")
    device = (stats.get("devices") or [{}])[0]
    print(f"connected to {host}: {device.get('name', '?')} "
          f"vram_free={device.get('vram_free', 0) / 2 ** 30:.1f}GiB", flush=True)

    workflow = json.load(open(args.workflow))
    load_nodes = [nid for nid, n in workflow.items() if n["class_type"] == "LoadImage"]
    if len(load_nodes) != 1:
        raise SystemExit(f"expected exactly one LoadImage node, found {load_nodes}; adapt the script")
    load_id = load_nodes[0]
    sampler_ids = []
    for nid, n in workflow.items():
        if "KSampler" in n["class_type"] and "steps" in n["inputs"]:
            print(f"override node {nid} steps {n['inputs']['steps']} -> {args.steps}")
            n["inputs"]["steps"] = int(args.steps)
            sampler_ids.append(nid)

    if args.batch > 1 or any(n["class_type"] == "RepeatLatentBatch" for n in workflow.values()):
        repeats = [nid for nid, n in workflow.items() if n["class_type"] == "RepeatLatentBatch"]
        if repeats:
            for nid in repeats:
                print(f"override node {nid} RepeatLatentBatch amount -> {args.batch}")
                workflow[nid]["inputs"]["amount"] = int(args.batch)
        else:
            for nid in sampler_ids:
                src = workflow[nid]["inputs"].get("latent_image")
                if isinstance(src, list):
                    new_id = f"warmup_batch_{nid}"
                    workflow[new_id] = {"class_type": "RepeatLatentBatch",
                                        "inputs": {"samples": src, "amount": int(args.batch)}}
                    workflow[nid]["inputs"]["latent_image"] = [new_id, 0]
                    print(f"inserted RepeatLatentBatch (amount {args.batch}) before sampler {nid}")

    total_started = time.monotonic()
    for i, (w, h) in enumerate(shapes, start=1):
        name = upload_png(host, f"warmup_{w}x{h}.png", dummy_rgba_png(w, h))
        workflow[load_id]["inputs"]["image"] = name
        elapsed = submit_and_wait(host, workflow, args.timeout)
        print(f"[{i}/{len(shapes)}] {w}x{h}: {elapsed:.1f}s", flush=True)
    print(f"done in {(time.monotonic() - total_started) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
