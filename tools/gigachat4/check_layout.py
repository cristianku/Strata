#!/usr/bin/env python3
"""Check the pinned GigaChat4-708B-A41B tensor layout. Offline unless --hf is set."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, build_opener

if __package__:
    from .memory_budget import DTYPE_BYTES, INDEX_SHA, NoRedirect, REPO, REV, tensor_schema
else:
    from memory_budget import DTYPE_BYTES, INDEX_SHA, NoRedirect, REPO, REV, tensor_schema


INDEX = "model.safetensors.index.json"
# Size and digest of the original JSON bytes, before parsing.
JSON_FILES = {
    "config.json": (4337, "08249cf71a363ba9855b5f7c2494e79012f04189cb110692c2e83d82d84c41b1"),
    INDEX: (9157175, INDEX_SHA),
}
SHARDS = tuple(f"model-{i:05d}-of-00045.safetensors" for i in range(1, 46))
# Observed payload start (8-byte prefix + padded JSON) in each pinned shard.
# Refuse a changed length *before* issuing the second HTTP request.
START = dict(zip(SHARDS, (
    6648, 51648, 142744, 142528, 142744, 142632, 142632, 142744, 142528, 142752,
    142520, 142744, 142632, 142632, 142744, 142424, 142088, 142600, 142912,
    142816, 142800, 142928, 142696, 142504, 142376, 142904, 142808, 142808,
    142928, 142712, 142936, 141960, 142904, 142816, 142808, 142920, 142704,
    142936, 142720, 142160, 142808, 142808, 142384, 141608, 6227168,
), strict=True))
# SHA-256 of the concatenated 32-byte canonical header digests in shard order.
HEADERS_SHA = "285a4b9f322275dfa7f94a40394767c7e749596308193b07a510b211a3c21965"
MTP = "model.layers.64."
FP = "model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
BF = "model.layers.3.self_attn.indexer.weights_proj.weight"
COLS = 7168
SAMPLE_CAP = 32 * 1024
SAMPLE_PLAN = (
    # name, tensor-relative byte offset, byte count, dtype, fixture shape
    (FP, 574 * COLS, 2 * COLS, "F8_E4M3", [2, COLS]),
    (FP + "_scale_inv", 4 * (COLS // 128) * 4, (COLS // 128) * 4, "F32", [1, COLS // 128]),
    (BF, 63 * COLS * 2, COLS * 2, "BF16", [1, COLS]),
)
SAMPLE_BYTES = sum(row[2] for row in SAMPLE_PLAN)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=unique)


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def local_bytes(path, limit):
    require(path.is_file(), f"missing file: {path}")
    with path.open("rb") as fp:
        raw = fp.read(limit + 1)
    require(len(raw) <= limit, f"{path}: exceeds {limit} bytes")
    return raw


class PinnedHF:
    """No Hub SDK or file downloader; all reads, including JSON, require an exact 206."""

    def __init__(self):
        # urllib's default redirect handler reads the redirect body. Handle redirects
        # ourselves and close them unread, even if a server returns a huge response.
        self.opener = build_opener(NoRedirect())
        self.bytes = 0

    def read(self, name, start, length, *, total=None, payload=False):
        end = start + length - 1
        if name in JSON_FILES:
            safe = (start, length) == (0, JSON_FILES[name][0]) and not payload
        elif name in START:
            safe = ((start, length) in ((0, 8), (8, START[name] - 8)) if not payload else
                    total is not None and START[name] <= start <= end < total and 0 < length <= SAMPLE_CAP)
        else:
            safe = False
        require(safe, "request outside the allowed pinned ranges")
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}?layout_range={start}-{end}"
        for _ in range(6):
            u = urlsplit(url)
            host = u.hostname or ""
            require(u.scheme == "https" and u.port in (None, 443) and not u.username and not u.password
                    and any(host == d or host.endswith("." + d) for d in ("hf.co", "huggingface.co")),
                    "refused non-Hugging Face redirect")
            req = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
            try:
                response = self.opener.open(req, timeout=40)
            except HTTPError as exc:
                with exc:
                    if exc.code not in (301, 302, 303, 307, 308) or not exc.headers.get("Location"):
                        raise ValueError(f"{name}: HTTP {exc.code}; body refused") from None
                    url = urljoin(url, exc.headers["Location"])
                    continue
            with response as r:
                cr = re.fullmatch(rf"bytes {start}-{end}/([0-9]+)", r.headers.get("Content-Range", ""))
                cl = r.headers.get("Content-Length")
                require(r.status == 206 and cr and (cl is None or cl == str(length))
                        and r.headers.get("Content-Encoding", "identity").lower() == "identity",
                        f"{name}: HTTP {r.status} or incorrect range/encoding; body refused")
                size = int(cr[1])
                require(end < size and (total is None or total == size), f"{name}: file size changed")
                data = r.read(length)  # never r.read(), even for a purported 206
                require(len(data) == length, f"{name}: short range")
                self.bytes += length
                return data, size
        raise ValueError(f"{name}: too many redirects")

    def header(self, shard):
        prefix, total = self.read(shard, 0, 8)
        n, = struct.unpack("<Q", prefix)
        require(8 + n == START[shard] and 8 + n < total, f"{shard}: unsafe or unpinned header length")
        raw, _ = self.read(shard, 8, n, total=total)
        return [8 + n, total, parse(raw)]


def pinned_json(name, raw):
    size, sha = JSON_FILES[name]
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha,
            f"{name}: not the original JSON at {REV}")
    return parse(raw)


def check_names(index, schema):
    files = index["weight_map"]
    missing = schema.keys() - files.keys()
    extra = files.keys() - schema.keys()
    require(not (missing or extra), f"index names: missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]}")
    require(len(files) == 95616 and set(files.values()) == set(SHARDS), "wrong tensor/shard count")
    return files


def group(name):
    return "mtp" if name.startswith(MTP) else "base"


def check_header(shard, obj, names, schema):
    """Accept plain JSON or [payload_start, shard_size, header] from the range cache."""
    reported_size = None
    if isinstance(obj, list):
        require(len(obj) == 3 and type(obj[0]) is int and type(obj[1]) is int and obj[0] == START[shard],
                f"{shard}: bad cached prefix/size")
        _, reported_size, obj = obj
    require(isinstance(obj, dict), f"{shard}: expected a safetensors header object")
    metadata = obj.get("__metadata__", {})
    require(isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values()),
            f"{shard}: invalid __metadata__")
    actual = obj.keys() - {"__metadata__"}
    require(actual == names, f"{shard}: missing {sorted(names - actual)[:2]}, unexpected {sorted(actual - names)[:2]}")
    spans = []
    for name in names:
        tensor = obj[name]
        dt, shape = schema[name]
        require(isinstance(tensor, dict) and tensor.get("dtype") == dt, f"{name}: expected dtype {dt}")
        dims = tensor.get("shape")
        require(isinstance(dims, list) and all(type(d) is int for d in dims) and tuple(dims) == shape,
                f"{name}: expected shape {shape}, got {dims}")
        offsets = tensor.get("data_offsets")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets)
                and offsets[0] >= 0 and offsets[1] - offsets[0] == DTYPE_BYTES[dt] * math.prod(shape),
                f"{name}: invalid data_offsets")
        spans.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor, f"{shard}: gap or overlap at {cursor}")
        cursor = end
    size = START[shard] + cursor
    require(reported_size is None or size == reported_size, f"{shard}: spans do not fill shard")
    return obj, size


def validate(index, load_header=None, save=None):
    schema = dict(tensor_schema())  # same lossless layout used by the memory inventory
    files = check_names(index, schema)
    counts = {"base": Counter(), "mtp": Counter()}
    sizes = Counter()
    by_shard = defaultdict(set)
    for name, (dt, shape) in schema.items():
        part = group(name)
        counts[part][dt] += 1
        sizes[part] += math.prod(shape) * DTYPE_BYTES[dt]
        by_shard[files[name]].add(name)
    require(sum(sizes.values()) == index["metadata"]["total_size"], "index total_size differs from layout")
    report = {
        "revision": REV, "mode": "headers" if load_header else "names_only",
        "names_checked": len(files), "headers_checked": 0, "header_bytes_including_prefix": 0,
        "base": {"names": counts["base"].total()}, "mtp": {"names": counts["mtp"].total()},
    }
    samples = {}
    if load_header:
        manifest = hashlib.sha256()
        for shard in SHARDS:
            h, size = check_header(shard, load_header(shard), by_shard[shard], schema)
            manifest.update(hashlib.sha256(compact(h)).digest())
            if save:
                (save / (shard + ".json")).write_bytes(compact([START[shard], size, h]))
            for key in (FP, FP + "_scale_inv", BF):
                if key in h:
                    samples[key] = (shard, size, START[shard] + h[key]["data_offsets"][0])
        require(manifest.hexdigest() == HEADERS_SHA, "headers disagree with pinned offsets/metadata digest")
        report.update(headers_checked=len(SHARDS), header_bytes_including_prefix=sum(START.values()))
        for part in counts:
            report[part].update(dtypes=dict(counts[part]), tensor_bytes=sizes[part])
    return report, samples


def sample_projections(hf, locations, cap, runner):
    """Use two real FP8 rows, their scale row and one real BF16 row. CPU only."""
    require(SAMPLE_BYTES <= cap <= SAMPLE_CAP, f"--sample-projections requires {SAMPLE_BYTES}..{SAMPLE_CAP} bytes")
    import torch  # optional; no model code/weights loaded
    require(sys.byteorder == "little" and hasattr(os, "memfd_create") and Path("/proc/self/fd").is_dir(),
            "projection oracle requires little-endian Linux with memfd and /proc")
    header, chunks = {}, []
    cursor = 0
    for key, offset, length, dt, shape in SAMPLE_PLAN:
        shard, size, base = locations[key]
        data, _ = hf.read(shard, base + offset, length, total=size, payload=True)
        chunks.append(data)
        header[key] = {"dtype": dt, "shape": shape, "data_offsets": [cursor, cursor + length]}
        cursor += length
    torch.set_num_threads(1)
    x = ((torch.arange(COLS) % 17) - 8).float() / 8
    fp, scale, bf = [torch.frombuffer(bytearray(buf), dtype=t).float() for buf, t in zip(
        chunks, (torch.float8_e4m3fn, torch.float32, torch.bfloat16))]
    want = torch.cat(((fp.reshape(2, COLS) * scale.repeat_interleave(128)) @ x, bf.reshape(1, COLS) @ x))
    # FileWeights opens a regular file. Hand the child a RAM-only descriptor;
    # the temporary directory contains just a symlink, never any weight bytes.
    with os.fdopen(os.memfd_create("projection-fixture"), "wb") as ram, tempfile.TemporaryDirectory() as tmp:
        h = compact(header)
        h += b" " * (-len(h) % 8)
        ram.write(struct.pack("<Q", len(h)) + h + b"".join(chunks))
        ram.flush()
        (Path(tmp) / "model.safetensors").symlink_to(f"/proc/self/fd/{ram.fileno()}")
        out = subprocess.run([str(runner.resolve()), tmp], check=True, capture_output=True, timeout=60,
                             pass_fds=(ram.fileno(),))
        actual = torch.tensor(json.loads(out.stdout), dtype=torch.float32)
    require(actual.shape == want.shape and torch.allclose(actual, want, rtol=1e-4, atol=1e-4),
            f"projection mismatch: native={actual.tolist()} torch={want.tolist()}")
    return {"payload_bytes_read": SAMPLE_BYTES, "fp8_rows": [574, 575], "bf16_rows": [63],
            "max_abs_error": float((actual - want).abs().max()), "torch": str(torch.__version__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="local pinned config.json (also pass --index)")
    source.add_argument("--hf", action="store_true", help="exact byte ranges from the pinned Hugging Face revision")
    p.add_argument("--index", type=Path, help="local pinned model.safetensors.index.json")
    p.add_argument("--headers", type=Path, help="directory of 45 <shard>.json headers; also reusable with --hf")
    p.add_argument("--save-metadata", type=Path, help="save verified metadata to a new directory (requires --hf)")
    p.add_argument("--sample-projections", type=int, metavar="BYTE_CAP", help="opt in to restricted weight slices (requires --hf)")
    p.add_argument("--projection-runner", type=Path, help="compiled projection probe; see docs/gigachat4/LAYOUT.md")
    args = p.parse_args()
    if bool(args.config) != bool(args.index):
        p.error("--config and --index must be used together")
    if args.save_metadata and (not args.hf or args.save_metadata.exists()):
        p.error("--save-metadata requires --hf and a new destination directory")
    if args.sample_projections is not None or args.projection_runner:
        if not (args.hf and args.projection_runner and args.projection_runner.is_file()
                and args.sample_projections is not None and SAMPLE_BYTES <= args.sample_projections <= SAMPLE_CAP):
            p.error(f"sampling requires --hf, an existing --projection-runner and an explicit cap of {SAMPLE_BYTES}..{SAMPLE_CAP} bytes")
    hf = PinnedHF() if args.hf else None
    raw = {}
    local = {"config.json": args.config, INDEX: args.index}
    for name, (size, _) in JSON_FILES.items():
        raw[name] = hf.read(name, 0, size, total=size)[0] if hf else local_bytes(local[name], size)
    pinned_json("config.json", raw["config.json"])
    index = pinned_json(INDEX, raw[INDEX])
    load_header = hf.header if hf else None
    if args.headers:
        def load_header(shard):
            return parse(local_bytes(args.headers / (shard + ".json"), 16 << 20))
    with ExitStack() as stack:
        stage = None
        if args.save_metadata:
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=args.save_metadata.parent)))
            stage = tmp / "metadata"
            (stage / "headers").mkdir(parents=True)
        report, locations = validate(index, load_header, stage / "headers" if stage else None)
        if stage:
            for name, data in raw.items():
                (stage / name).write_bytes(data)
            stage.rename(args.save_metadata)
    if args.sample_projections is not None:
        report["projection_sample"] = sample_projections(hf, locations, args.sample_projections, args.projection_runner)
    report["http_body_bytes_read"] = hf.bytes if hf else 0
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        sys.exit(f"layout check failed: {exc}")
