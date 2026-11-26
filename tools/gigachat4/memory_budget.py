#!/usr/bin/env python3
"""GigaChat4 target-only sizing and routing trace replay. Standard library only."""
import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError


REPO = "ai-sage/GigaChat4-708B-A41B"
REV = "2baf8b0cf71f1528c14e7a274877d9c8fbf22697"
INDEX_SHA = "f358bf4a2f1570920a0ff9c6c37362babb09ab2ee139df09f3f900c6c475dc50"
GIB = 1 << 30
DTYPE_BYTES = {"BF16": 2, "F8_E4M3": 1, "F32": 4}
HEADER_CAP = 8 << 20  # per shard; observed largest is 6,227,160 bytes


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def header_range(shard, a, b):
    """Read exactly the prefix or the announced JSON range. Refuse a 200 before reading."""
    if not re.fullmatch(r"model-000(?:0[1-9]|[1-3][0-9]|4[0-5])-of-00045\.safetensors", shard):
        raise ValueError("unexpected shard")
    if (a, b) != (0, 7) and not (a == 8 and a <= b < 8 + HEADER_CAP):
        raise ValueError("only the length prefix and its capped header are allowed")
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}?header_range={a}-{b}"
    opener = build_opener(NoRedirect())
    for _ in range(6):
        if urlsplit(url).scheme != "https":
            raise ValueError("non-HTTPS redirect")
        try:
            r = opener.open(Request(url, headers={"Range": f"bytes={a}-{b}",
                                                  "Accept-Encoding": "identity"}), timeout=40)
        except HTTPError as e:
            with e:
                if e.code not in (301, 302, 303, 307, 308) or "Location" not in e.headers:
                    raise
                url = urljoin(url, e.headers["Location"])
                continue
        with r:
            n = b - a + 1
            cr = re.fullmatch(rf"bytes {a}-{b}/(\d+)", r.headers.get("Content-Range", ""))
            if (r.status != 206 or not cr or int(cr[1]) <= b
                    or int(r.headers.get("Content-Length", n)) != n
                    or r.headers.get("Content-Encoding", "identity") != "identity"):
                raise ValueError(f"{shard}: server did not honor exact range; body refused")
            data = r.read(n)
            if len(data) != n:
                raise ValueError("short header read")
            return data, int(cr[1])
    raise ValueError("too many redirects")


def fetch_header(shard):
    prefix, total = header_range(shard, 0, 7)
    n, = struct.unpack("<Q", prefix)
    if not (2 <= n <= HEADER_CAP and 8 + n < total):
        raise ValueError(f"{shard}: unsafe header length")
    data, total2 = header_range(shard, 8, 7 + n)
    if total != total2:
        raise ValueError("inconsistent shard length")
    return shard, json.loads(data), n + 8, total


def read_index(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != INDEX_SHA:
        raise ValueError("wrong index: expected the pinned official checkpoint")
    return json.loads(data)


# (values, bytes including metadata) from gguf/constants.py, GGML_QUANT_SIZES.
# Both matrix widths of an expert are divisible by all these block lengths.
BLOCK = {"IQ2_XS": (256, 74), "Q2_K": (256, 84), "Q3_K": (256, 110),
         "Q4_K": (256, 144), "Q8_0": (32, 34), "BF16": (1, 2)}
EXPERT_PARAMS = 3 * 7168 * 2048
N_EXPERTS = 61 * 256
ROUTES = 61 * 8


def expert_bytes(fmt):
    if fmt == "official":
        return EXPERT_PARAMS + 3 * (7168 // 128) * (2048 // 128) * 4
    values, size = BLOCK[fmt]
    return EXPERT_PARAMS // values * size


def tensor_schema():
    """Lossless name/shape/dtype templates transcribed from all 45 pinned SF headers."""
    def bf(name, *shape):
        return name, ("BF16", shape)

    def fp(prefix, out_dim, in_dim):
        yield prefix + ".weight", ("F8_E4M3", (out_dim, in_dim))
        yield prefix + ".weight_scale_inv", ("F32", ((out_dim + 127) // 128, (in_dim + 127) // 128))

    def norm(prefix, w):
        yield bf(prefix + ".weight", w)
        yield bf(prefix + ".gate_up_projection.weight", 16, w)
        yield bf(prefix + ".gate_down_projection.weight", w, 16)

    vocab = (128256, 7168)
    yield bf("model.embed_tokens.weight", *vocab)
    yield bf("lm_head.weight", *vocab)
    yield from norm("model.norm", 7168)
    for i in range(65):
        p = f"model.layers.{i}."
        for stem in ("input_layernorm", "post_attention_layernorm", "post_self_attn_layernorm",
                     "post_feedforward_layernorm"):
            yield from norm(p + stem, 7168)
        a = p + "self_attn."
        if i % 4 != 3 and i < 64:
            yield bf(a + "A_log", 64)
            yield bf(a + "dt_bias", 64)
            yield bf(a + "conv1d.weight", 16384, 1, 4)
            yield bf(a + "norm.weight", 128)
            yield bf(a + "in_proj_ba.weight", 128, 7168)
            yield from fp(a + "in_proj_qkvz", 24576, 7168)
            yield from fp(a + "out_proj", 7168, 8192)
        else:
            yield bf(a + "gate_proj.weight", 8192, 7168)
            yield bf(a + "indexer.k_norm.weight", 128)
            yield bf(a + "indexer.weights_proj.weight", 64, 7168)
            yield bf(a + "indexer.wk.weight", 128, 7168)
            yield from norm(a + "q_a_layernorm", 1536)
            yield from norm(a + "kv_a_layernorm", 512)
            for stem, o, d in (("indexer.wq_b", 8192, 1536), ("q_a_proj", 1536, 7168),
                               ("q_b_proj", 12288, 1536), ("kv_a_proj_with_mqa", 576, 7168),
                               ("kv_b_proj", 16384, 512), ("o_proj", 7168, 8192)):
                yield from fp(a + stem, o, d)
        m = p + "mlp."
        if i < 3 or i == 64:
            width = 18432
        else:
            width = 2048
            yield bf(m + "gate.weight", 256, 7168)
            yield bf(m + "gate.e_score_correction_bias", 256)
            for e in range(256):
                ep = m + f"experts.{e}."
                yield from fp(ep + "up_proj", width, 7168)
                yield from fp(ep + "gate_proj", width, 7168)
                yield from fp(ep + "down_proj", 7168, width)
            m += "shared_experts."
        yield bf(m + "gate_proj.weight", width, 7168)
        yield bf(m + "up_proj.weight", width, 7168)
        yield bf(m + "down_proj.weight", 7168, width)
        if i == 64:
            yield bf(p + "eh_proj.weight", 7168, 14336)
            yield bf(p + "embed_tokens.weight", *vocab)
            yield bf(p + "shared_head.head.weight", *vocab)
            for stem in ("enorm", "hnorm", "shared_head.norm"):
                yield from norm(p + stem, 7168)


def verify_headers(index_path, cached=None):
    """Match ALL names, placement, dtype, shape and contiguous byte spans to the index."""
    j = read_index(index_path)
    files = j["weight_map"]
    schema = dict(tensor_schema())
    if schema.keys() != files.keys():
        raise ValueError("schema/index keys differ")
    shards = sorted(set(files.values()))
    def load(shard):
        if cached is None:
            return fetch_header(shard)
        n, total, h = json.loads((cached / (shard + ".json")).read_text())
        return shard, h, n, total

    seen, headers, payload = set(), 0, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for shard, h, n, total in pool.map(load, shards):
            headers += n
            cursor = 0
            h.pop("__metadata__", None)
            for key, v in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                dt, shape = schema[key]
                end = cursor + math.prod(shape) * DTYPE_BYTES[dt]
                if (key in seen or files[key] != shard or v["dtype"] != dt or tuple(v["shape"]) != shape
                        or v["data_offsets"] != [cursor, end]):
                    raise ValueError(f"header/index mismatch: {shard}: {key}")
                cursor = end
                seen.add(key)
            if n + cursor != total:
                raise ValueError(f"shard length mismatch: {shard}")
            payload += cursor
    if seen != schema.keys() or payload != j["metadata"]["total_size"]:
        raise ValueError("incomplete inventory")
    return {"revision": REV, "shards": len(shards), "tensors": len(seen),
            "header_bytes_including_prefix": headers, "tensor_bytes": payload,
            "all_shard_bytes": payload + headers}


def inventory():
    groups, layers = defaultdict(Counter), defaultdict(Counter)
    for name, (dt, shape) in tensor_schema():
        match = re.match(r"model\.layers\.(\d+)\.", name)
        i = int(match[1]) if match else None
        if i is None:
            group = "global"
        elif i == 64:
            group = "mtp"
        elif ".experts." in name:
            group = "routed"
        elif ".shared_experts." in name:
            group = "shared"
        else:
            group = "nonexpert"
        count = math.prod(shape)
        size = count * DTYPE_BYTES[dt]
        scale = name.endswith(".weight_scale_inv")
        groups[group].update(tensors=1, coefficients=0 if scale else count,
                             source_bytes=size, fp32_scale_bytes=size if scale else 0)
        if i is not None and i < 64:
            layers[i][group] += size
    return groups, layers


def repack_fixed(fmt):
    """What-if: block-quantize aligned large matrices including head/embed/shared; keep router BF16."""
    if fmt not in BLOCK:
        raise ValueError("matrix repack requires a block format or BF16")
    values, size = BLOCK[fmt]
    totals = Counter()
    for name, (_, shape) in tensor_schema():
        if (name.startswith("model.layers.64.") or ".experts." in name or name.endswith(".weight_scale_inv")):
            continue
        i = int(name.split(".")[2]) if name.startswith("model.layers.") else "global"
        n = math.prod(shape)
        if len(shape) == 2 and min(shape) >= 256 and shape[-1] % values == 0 and ".mlp.gate." not in name:
            totals[i] += n // values * size
        else:
            totals[i] += n * 2
    return totals.pop("global"), totals


def budget(fmt, gpu, host, context, batch, latent, gpu_reserve, host_reserve, fixed_pack="source"):
    g, layers = inventory()
    global_bytes = g["global"]["source_bytes"]
    layer_bytes = {i: v["nonexpert"] + v["shared"] for i, v in layers.items()}
    source_fixed = global_bytes + sum(layer_bytes.values())
    if fixed_pack == "matrices":
        global_bytes, layer_bytes = repack_fixed(fmt)
    layer_fixed, peak_layer = sum(layer_bytes.values()), max(layer_bytes.values())
    fixed = global_bytes + layer_fixed
    params = sum(v["coefficients"] for k, v in g.items() if k != "mtp")
    active = params - (N_EXPERTS - ROUTES) * EXPERT_PARAMS
    blob = expert_bytes(fmt)
    state = state_bytes(context, batch, latent)
    # One sequence decoded at a time or sequential execution for multiple sessions.
    # Two top-8 buffers per tier, independent of the resident cache, including misses.
    staging = 2 * 8 * blob
    def policy(layer_stream):
        gpu_fixed = global_bytes + 2 * peak_layer if layer_stream else fixed
        host_fixed = layer_fixed if layer_stream else 0
        gpu_base = sum(state.values()) + gpu_fixed + staging + gpu_reserve
        # In layer-stream mode copy from resident pinned layer matrices (no third copy).
        host_base = host_fixed + staging + host_reserve
        gs, hs = [max(0, min(N_EXPERTS, (total - base) // blob))
                  for total, base in ((gpu, gpu_base), (host, host_base))]
        return {"within_assumed_budgets": gpu_base <= gpu and host_base <= host,
                "minimum_gpu_bytes_incl_reserve": gpu_base, "minimum_host_bytes_incl_reserve": host_base,
                "gpu_cache_slots": gs, "host_cache_slots": hs,
                "max_demand_gpu_hit_fraction": min(1, gs / ROUTES),
                "min_routed_pcie_bytes_per_token": max(0, ROUTES - gs) * blob,
                "min_routed_disk_bytes_per_token": max(0, ROUTES - gs - hs) * blob,
                "peak_budgeted_gpu_bytes": gpu_base + gs * blob,
                "peak_budgeted_host_bytes": host_base + hs * blob,
                "unavoidable_fixed_pcie_bytes_per_token": host_fixed}
    return {"revision": REV, "groups": g, "layer_source_bytes": layers,
            "base_coefficients": params, "base_params_excluding_router_buffers": params - 61 * 256,
            "active_full_tensor_coefficients": active,
            "active_one_embedding_row_coefficients": active - 128256 * 7168 + 7168,
            "base_source_bytes": source_fixed + g["routed"]["source_bytes"],
            "full_source_bytes": source_fixed + g["routed"]["source_bytes"] + g["mtp"]["source_bytes"],
            "fixed_source_bytes_incl_shared": source_fixed, "selected_fixed_bytes": fixed,
            "selected_fixed_policy": fixed_pack, "selected_global_bytes": global_bytes,
            "selected_max_layer_fixed_bytes": peak_layer, "state_bytes": state,
            # Separate from the proposed GPU-state budgets: native Vec<float> on CPU,
            # shared rotary key once. Logical length only, not vector capacity or RSS.
            "native_host_logical_state_bytes": batch * (
                context * 16 * (64*(128+128) + 64 + 128) * 4
                + 48*(64*128*128 + 16384*4)*4),
            "context": context, "sequences": batch, "latent": latent, "expert_format": fmt,
            "gpu_budget_bytes": gpu, "host_budget_bytes": host,
            "gpu_reserve_bytes": gpu_reserve, "host_reserve_bytes": host_reserve,
            "staging_bytes_per_tier": staging, "base_repacked_bytes": fixed + N_EXPERTS * blob,
            "all_expert_miss_bytes_per_token": ROUTES * blob,
            "ideal_all_base_uniform_bit_bytes": {q: (params*q + 7)//8 for q in (2, 3, 4, 8, 16)},
            "mixed_routed_formats": {f: {"expert_bytes": expert_bytes(f),
                                         "base_bytes_if_fixed_source": source_fixed + N_EXPERTS * expert_bytes(f),
                                         "active_routed_bytes": ROUTES * expert_bytes(f)}
                                     for f in ("official", *BLOCK)},
            "gpu_fixed": policy(False), "host_layer_fixed": policy(True)}


def state_bytes(context, batch, latent):
    # BF16 K/V/index + FP32 recurrent + BF16 convolution ring; fixed for this revision.
    kv = (512 + 64 if latent else 64 * (192 + 128)) * 2 * 16
    index = 128 * 2 * 16
    return {"kv": kv * context * batch, "index": index * context * batch,
            "recurrent": 48 * 64 * 128 * 128 * 4 * batch,
            "conv": 48 * (2 * 32 * 128 + 64 * 128) * 4 * 2 * batch}


def admit(cache, keys, slots):
    """Touch group hits first; complete demand in router order, then trim to capacity."""
    if not slots:
        return
    for key in keys:
        cache[key] = None
        cache.move_to_end(key)
        if len(cache) > slots:
            cache.popitem(last=False)


def replay(records, fmt, gpu_bytes, host_bytes, predict=0, warmup=0):
    """One sequence, in order. Both budgets are net expert-cache bytes."""
    blob = expert_bytes(fmt)
    gs, hs = (min(N_EXPERTS, b // blob) for b in (gpu_bytes, host_bytes))
    if min(gs, hs, warmup) < 0 or not (0 <= predict <= 8):
        raise ValueError("invalid cache budget or prediction depth")
    gpu, host = OrderedDict(), OrderedDict()
    previous, by_layer = {}, defaultdict(Counter)
    token0, events = None, 0
    for r in records:
        t, layer, ids = r["token"], r["layer"], r["experts"]
        if (type(t) is not int or t < 0 or type(layer) is not int or not isinstance(ids, list)
                or len(ids) != 8 or any(type(e) is not int or not 0 <= e < 256 for e in ids)
                or len(set(ids)) != 8):
            raise ValueError(f"bad route at record {events + 1}")
        if token0 is None:
            token0 = t
        if t != token0 + events // 61 or layer != 3 + events % 61:
            raise ValueError("expected complete, consecutive tokens, real decoder layers 3..63")
        keys = [(layer, e) for e in ids]
        # Score the previous observation for THIS layer before looking up current hits
        # or admitting any current demands. Predictions never change cache contents.
        forecast = previous.get(layer, ())[:predict]
        need_prefetch = [k for k in forecast if k not in gpu]
        hits = [k for k in keys if k in gpu]
        misses = [k for k in keys if k not in gpu]
        warm = [k for k in misses if k in host]
        cold = [k for k in misses if k not in host]
        if events >= warmup * 61:
            c = by_layer[layer]
            c.update(events=1, demanded=len(keys), gpu_hits=len(hits), host_hits=len(warm),
                     disk_misses=len(cold), all_gpu_events=int(not misses),
                     predicted=len(forecast), correct=len(set(forecast).intersection(keys)),
                     prefetch_candidates=len(need_prefetch),
                     useful_candidates=len(set(need_prefetch).intersection(misses)))
        admit(host, warm + cold, hs)
        admit(gpu, hits + misses, gs)
        previous[layer] = keys
        events += 1
    if events <= 61 * warmup or events % 61:
        raise ValueError("no scored tokens or incomplete final token")
    c = Counter()
    for layer_counts in by_layer.values():
        c.update(layer_counts)
    return {"format": fmt, "blob_bytes": blob, "gpu_slots": gs, "host_slots": hs,
            "tokens": events // 61 - warmup, "warmup_tokens": warmup,
            "policy": "two independent initially empty global LRUs",
            "demand_pcie_bytes": (c["host_hits"] + c["disk_misses"]) * blob,
            "demand_nvme_bytes": c["disk_misses"] * blob,
            "gpu_hit_fraction": c["gpu_hits"] / c["demanded"],
            "host_hit_given_gpu_miss": (c["host_hits"] / (c["host_hits"] + c["disk_misses"])
                                        if c["host_hits"] + c["disk_misses"] else None),
            "prediction_precision": c["correct"] / c["predicted"] if c["predicted"] else None,
            "unneeded_prefetch_bytes": (c["prefetch_candidates"] - c["useful_candidates"]) * blob,
            "counts": c, "per_layer": by_layer}


def link_bounds(pcie_bytes, disk_bytes, pcie_gbs, disk_gbs):
    p, d = pcie_bytes / (pcie_gbs * 1e9), disk_bytes / (disk_gbs * 1e9)
    return {"pcie_bytes": pcie_bytes, "disk_bytes": disk_bytes, "pcie_seconds": p, "disk_seconds": d,
            "perfect_overlap_seconds_floor": max(p, d), "no_overlap_seconds_floor": p + d}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("budget", help="target-only mixed-precision weights, cache plans and I/O floors")
    b.add_argument("--gpu-gib", type=float, default=16)
    b.add_argument("--host-gib", type=float, default=64)
    b.add_argument("--gpu-reserve-gib", type=float, default=2, help="assumed runtime/scratch/driver headroom")
    b.add_argument("--host-reserve-gib", type=float, default=8, help="assumed OS/runtime/page-cache headroom")
    b.add_argument("--context", type=int, default=8192)
    b.add_argument("--sequences", type=int, default=1)
    b.add_argument("--kv", choices=("reference", "latent"), default="reference")
    b.add_argument("--fixed", choices=("source", "matrices"), default="source",
                   help="preserve fixed tensors as shipped, or repack eligible matrices with --format")
    b.add_argument("--gpu-hit", type=float, default=0, help="hypothetical routed byte hit fraction")
    b.add_argument("--host-hit", type=float, default=0, help="conditional on GPU miss")
    r = sp.add_parser("replay", help="replay measured, chronological base decode routes; no inference")
    r.add_argument("trace", type=Path, help="native --routes JSONL: one token/layer/experts record per MoE layer")
    r.add_argument("--warmup-tokens", type=int, default=0,
                   help="process this many prefix positions without scoring (e.g. length of native prompt)")
    r.add_argument("--gpu-cache-gib", type=float, required=True, help="net routed residency budget")
    r.add_argument("--host-cache-gib", type=float, required=True, help="net routed residency budget")
    r.add_argument("--predict", type=int, choices=range(9), default=0,
                   help="score first N routes from previous token at same layer; no prefetch issued")
    for p in (r, b):
        p.add_argument("--format", choices=("official", *BLOCK), default="Q4_K")
        p.add_argument("--pcie-gbs", type=float, default=12, help="assumed one-way effective ceiling, decimal GB/s")
        p.add_argument("--nvme-gbs", type=float, default=3, help="assumed total read ceiling, decimal GB/s")
    inv = sp.add_parser("inventory", help="list all exact keys, dtypes and shapes (JSON lines)")
    inv.add_argument("--prefix", default="")
    v = sp.add_parser("verify", help="check the pinned official index against every shard header (network opt-in)")
    v.add_argument("index", type=Path)
    v.add_argument("--cached", type=Path, help="offline: directory of [header_bytes, file_bytes, header] JSON snapshots")
    args = ap.parse_args()
    try:
        if args.cmd == "inventory":
            for k, (dt, s) in tensor_schema():
                if k.startswith(args.prefix):
                    print(json.dumps({"name": k, "dtype": dt, "shape": s, "bytes": math.prod(s)*DTYPE_BYTES[dt]}))
            return
        if args.cmd == "verify":
            result = verify_headers(args.index, args.cached)
        else:
            if any(not math.isfinite(v) or v <= 0 for v in (args.pcie_gbs, args.nvme_gbs)):
                raise ValueError("bandwidths must be positive and finite")
            def gib(n):
                if not math.isfinite(n) or n < 0:
                    raise ValueError("GiB values must be nonnegative and finite")
                return int(n * GIB)

            if args.cmd == "budget":
                if not (0 <= args.gpu_hit <= 1 and 0 <= args.host_hit <= 1 and
                        1 <= args.context <= 1048576 and args.sequences >= 1):
                    raise ValueError("invalid hit fraction, context, or sequence count")
                result = budget(args.format, gib(args.gpu_gib), gib(args.host_gib), args.context,
                                args.sequences, args.kv == "latent", gib(args.gpu_reserve_gib),
                                gib(args.host_reserve_gib), args.fixed)
                miss = result["all_expert_miss_bytes_per_token"] * (1 - args.gpu_hit)
                blob_bytes = expert_bytes(args.format)
                for policy in ("gpu_fixed", "host_layer_fixed"):
                    p = result[policy]
                    fixed = p["unavoidable_fixed_pcie_bytes_per_token"]
                    disk = miss * (1 - args.host_hit)
                    p["assumptions_pass_capacity_bounds"] = (p["within_assumed_budgets"]
                                                              and args.gpu_hit <= p["max_demand_gpu_hit_fraction"]
                                                              and miss * args.host_hit <= p["host_cache_slots"] * blob_bytes)
                    p["io_floor_per_token"] = link_bounds(fixed + miss, disk, args.pcie_gbs, args.nvme_gbs)
                result.update(assumed_gpu_hit=args.gpu_hit, assumed_conditional_host_hit=args.host_hit)
            else:
                with args.trace.open(encoding="utf-8") as f:
                    records = (json.loads(s) for s in f if s.strip())
                    result = replay(records, args.format, gib(args.gpu_cache_gib),
                                    gib(args.host_cache_gib), args.predict, args.warmup_tokens)
                result["routed_only_io_floor_for_trace"] = link_bounds(result["demand_pcie_bytes"],
                                                                       result["demand_nvme_bytes"],
                                                                       args.pcie_gbs, args.nvme_gbs)
            result.update(assumed_pcie_GBps=args.pcie_gbs, assumed_nvme_GBps=args.nvme_gbs)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
