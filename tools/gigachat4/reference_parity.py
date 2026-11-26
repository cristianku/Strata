"""CPU parity oracle for the first four blocks of the pinned GigaChat4 model.

See docs/gigachat4/PARITY.md for setup. No model code is downloaded at runtime.
"""

import argparse
from contextlib import ExitStack
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

from compare_trace import compare

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from safetensors.torch import load_file, save_file


REVISION = "2baf8b0cf71f1528c14e7a274877d9c8fbf22697"
PINNED = {
    "modeling_gigachat4.py": "b3ebfbb2e18cb90f638887699a720a9894c28afa2aea58b9c77fa6249e2d7c46",
    "configuration_gigachat4.py": "84b06eca3c86c1baf0fa1118828e98ed1834c18c5c20c18ae83c46e77319a22f",
    "config.json": "08249cf71a363ba9855b5f7c2494e79012f04189cb110692c2e83d82d84c41b1",
    "model.safetensors.index.json": "f358bf4a2f1570920a0ff9c6c37362babb09ab2ee139df09f3f900c6c475dc50",
}
SEED = 70841
INDEX = "model.safetensors.index.json"
SHARDS = [f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
BLOCK = 128
FP8 = torch.float8_e4m3fn
# SHA-256 of compact sorted JSON {tensor name: dtype} for our 164 keys,
# taken from the safetensors headers at REVISION (no weight payload needed).
OFFICIAL_MIXED_DTYPES = "c8bd9e0b31172e9c49f411697a3ee4b2224b546926dd5eff44a12ad1e21b9562"


def pinned_model(source):
    for name, digest in PINNED.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{source / name} does not match official revision {REVISION}")
    # Private package for upstream's relative import. Do not import an __init__.py
    # from the download or call from_pretrained() on the 708B model.
    package = ModuleType("_gigachat4_pinned")
    package.__path__ = [str(source.resolve())]
    sys.modules[package.__name__] = package
    return import_module(package.__name__ + ".modeling_gigachat4")


def tiny_config(source):
    cfg = json.loads((source / "config.json").read_bytes())
    # Keep all model-wide options (including the original YaRN parameters and
    # context length). Shrink widths, head counts, depth and top-k budgets.
    cfg.pop("quantization_config")
    cfg.update(
        vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=4, full_attention_layers=[3], num_nextn_predict_layers=0,
        num_attention_heads=4, num_key_value_heads=4, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8,
        qk_head_dim=16, head_dim=8, v_head_dim=8,
        index_n_heads=8, index_head_dim=16, index_topk=2, indexer_norm_type="LlamaRMSNorm",
        n_routed_experts=8, num_experts_per_tok=2, dtype="float32",
    )
    return cfg


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_checkpoint(path, cfg, state, weight_map):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"fixture output requires a new or empty directory: {path}")
    total = sum(t.nbytes for t in state.values())
    for shard in SHARDS:
        save_file({k: state[k] for k, v in weight_map.items() if v == shard},
                  str(path / shard), metadata={"format": "pt"})
    write_json(path / INDEX, {"metadata": {"total_size": total}, "weight_map": weight_map})
    write_json(path / "config.json", cfg)
    return total


def make_fixture(path, cfg, model, source):
    state = model.state_dict()
    official = json.loads((source / INDEX).read_bytes())["weight_map"]
    if extra := state.keys() - official.keys():
        raise ValueError(f"fixture tensor names absent from official index: {sorted(extra)}")
    # Fill every parameter and persistent buffer, in stable key order. Nontrivial
    # learned norms, router bias and time constants exercise the associated paths.
    rng = torch.Generator().manual_seed(SEED)
    weight_map = {}
    for name, weight in sorted(state.items()):
        if weight.ndim >= 2:
            fan_in = math.prod(weight.shape[1:])
            std = 0.7 if name.endswith("embed_tokens.weight") else 0.8 / math.sqrt(fan_in)
            weight.normal_(0.0, std, generator=rng)
        elif name.endswith("e_score_correction_bias"):
            weight.uniform_(-0.12, 0.12, generator=rng)
        elif name.endswith("A_log"):
            weight.uniform_(-1.3, 0.2, generator=rng)
        elif name.endswith("dt_bias"):
            weight.uniform_(-0.7, 0.7, generator=rng)
        elif name.endswith(".weight"):
            weight.uniform_(0.8, 1.2, generator=rng)
        else:
            raise ValueError(f"no fixture initializer for {name}")
        weight_map[name] = SHARDS[int(".experts." in name)]
    total = write_checkpoint(path, cfg, state, weight_map)
    print(f"Wrote {len(state)} FP32 tensors ({total} payload bytes), seed {SEED}: {path}")


def read_checkpoint(path, cfg):
    if json.loads((path / "config.json").read_bytes()) != cfg:
        raise ValueError(f"expected the tiny parity config in {path}")
    index = json.loads((path / INDEX).read_bytes())
    weight_map = index["weight_map"]
    if set(weight_map.values()) != set(SHARDS):
        raise ValueError("expected the two synthetic fixture shards")
    state = {}
    for shard in SHARDS:
        tensors = load_file(str(path / shard), device="cpu")
        if set(tensors) != {k for k, v in weight_map.items() if v == shard}:
            raise ValueError(f"weight index disagrees with {shard}")
        state.update(tensors)
    if index["metadata"]["total_size"] != sum(t.nbytes for t in state.values()):
        raise ValueError(f"weight index has incorrect total_size: {path}")
    return state, weight_map


def quantize_blocks(weight):
    """Independent absmax scale per row/column tile, including ragged edges."""
    rows, cols = weight.shape
    if not weight.isfinite().all():
        raise ValueError("cannot quantize non-finite weights")
    packed = torch.empty_like(weight, dtype=FP8)
    scales = torch.empty(((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK))
    limit = torch.finfo(FP8).max
    for r in range(0, rows, BLOCK):
        for c in range(0, cols, BLOCK):
            tile = weight[r:r+BLOCK, c:c+BLOCK]
            amax = tile.abs().amax().item()
            scales[r//BLOCK, c//BLOCK] = amax / limit if amax else 1.0
            scale = scales[r//BLOCK, c//BLOCK]
            packed[r:r+BLOCK, c:c+BLOCK] = (tile / scale).clamp(-limit, limit).to(FP8)
    return packed, scales


def load_mixed(args, cfg, fp32, weight_map, model):
    """Generate (optionally), then independently unpack the on-disk checkpoint."""
    official = json.loads((args.source / INDEX).read_bytes())["weight_map"]
    qcfg = json.loads((args.source / "config.json").read_bytes())["quantization_config"]
    if qcfg["weight_block_size"] != [BLOCK, BLOCK] or qcfg["fmt"] != "e4m3":
        raise ValueError("unexpected upstream FP8 format")
    mixed_cfg = dict(cfg, dtype="bfloat16", quantization_config=qcfg)
    expected, dtypes, mixed_map = {}, {}, weight_map.copy()
    for name, w in fp32.items():
        key = name + "_scale_inv"
        if key in official:
            expected[name], expected[key] = quantize_blocks(w)
            mixed_map[key] = mixed_map[name]
            dtypes[name], dtypes[key] = "F8_E4M3", "F32"
        else:
            expected[name] = w.bfloat16()
            dtypes[name] = "BF16"
    manifest = json.dumps(dtypes, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(manifest).hexdigest() != OFFICIAL_MIXED_DTYPES:
        raise ValueError("mixed dtype plan does not match the pinned official header subset")
    if args.make_mixed_fixture:
        total = write_checkpoint(args.mixed_model, mixed_cfg, expected, mixed_map)
        print(f"Wrote {len(expected)} mixed tensors ({total} payload bytes): {args.mixed_model}")
    raw, _ = read_checkpoint(args.mixed_model, mixed_cfg)
    if raw.keys() != expected.keys():
        raise ValueError("incorrect mixed tensor names")
    # This prevents reporting 'quantization drift' between unrelated fixtures.
    for name, target in expected.items():
        t = raw[name]
        if (t.dtype != target.dtype or t.shape != target.shape or
                not torch.equal(t.view(torch.uint8), target.view(torch.uint8))):
            raise ValueError(f"mixed tensor is not the deterministic conversion of --model: {name}")
    # HF runs the unmodified class in FP32, using only values recovered from disk.
    # In particular, never pass quantization_config to HF or quantize activations.
    dequant = {}
    for name in fp32:
        w = raw[name].float()
        if raw[name].dtype == FP8:
            scales = raw[name + "_scale_inv"].repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            w = w * scales[:w.shape[0], :w.shape[1]]
        dequant[name] = w
    model.load_state_dict(dequant, strict=True)


def forward(model, tokens, cache=None, use_cache=True):
    steps = [{"layers": [None] * len(model.model.layers), "attention": [None] * len(model.model.layers),
              "routes": [[] for _ in model.model.layers]}
             for _ in tokens]
    offset = cache.get_seq_length() if cache is not None else 0

    def tensor_hook(idx, field):
        def save(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            for step, row in zip(steps, output[0].tolist(), strict=True):
                step[field][idx] = row
        return save

    def router_hook(idx):
        def save(_module, _args, output):
            for step, row in zip(steps, output[0].tolist(), strict=True):
                step["routes"][idx] = row
        return save

    def dsa_hook(module, _args, output):
        # In eager attention a masked column has exactly zero probability.
        attended = output[1][0].ne(0).any(dim=0)
        for i, row in enumerate(attended):
            t = offset + i
            if row[t + 1:].any() or row.sum().item() != min(t + 1, module.config.index_topk):
                raise ValueError(f"official DSA top-k/causality invariant failed at token {t}")

    with ExitStack() as hooks:
        for i, layer in enumerate(model.model.layers):
            hooks.enter_context(layer.register_forward_hook(tensor_hook(i, "layers")))
            hooks.enter_context(layer.self_attn.register_forward_hook(tensor_hook(i, "attention")))
            if hasattr(layer.mlp, "gate"):
                hooks.enter_context(layer.mlp.gate.register_forward_hook(router_hook(i)))
            if getattr(layer.self_attn, "use_dsa", False):
                hooks.enter_context(layer.self_attn.register_forward_hook(dsa_hook))
        out = model(input_ids=torch.tensor([tokens]), past_key_values=cache, use_cache=use_cache,
                    return_dict=True)
    for step, row in zip(steps, out.logits[0].tolist(), strict=True):
        step["logits"] = row
    if use_cache:
        # A batched prefill exposes only the state of its last token. A single
        # token forward exposes every layer's complete state for that step.
        steps[-1]["recurrent"] = [state[0].flatten().tolist() if state is not None else []
                                  for state in out.past_key_values.recurrent_states]
    return steps, out.past_key_values


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def cached_trace(model, prompt, generate, split):
    steps = []
    cache = None
    for part in ([[token] for token in prompt] if split else [prompt]):
        new, cache = forward(model, part, cache)
        steps.extend(new)
    generated = []
    for i in range(generate):
        token = argmax(steps[-1]["logits"])
        generated.append(token)
        if i + 1 < generate:
            new, cache = forward(model, [token], cache)
            steps.extend(new)
    return {"steps": steps, "generated": generated}


def validate(model, args, path, ref_path, native_path, label=""):
    prompt, generate = args.prompt, args.generate
    # Three independent cache lifetimes: token-wise decode, prompt prefill +
    # decode, and one uncached causal pass over the entire trajectory.
    ref = cached_trace(model, prompt, generate, split=True)
    cached = cached_trace(model, prompt, generate, split=False)
    compare(label + "HF batched prompt + decode vs HF incremental", ref, cached, 3e-5, 1e-5)
    trajectory = prompt + ref["generated"][:-1]
    full, _ = forward(model, trajectory, use_cache=False)
    full_trace = {"steps": full, "generated": [argmax(s["logits"]) for s in full[len(prompt)-1:]][:generate]}
    compare(label + "HF uncached prefill vs HF incremental", ref, full_trace, 3e-5, 1e-5)
    if ref_path:
        write_json(ref_path, ref)
    if args.runner:
        with TemporaryDirectory(prefix="gigachat4-trace-") as tmp:
            trace = Path(tmp) / "cpp.json"
            subprocess.run([str(args.runner.resolve()), *args.runner_arg, "--model", str(path.resolve()),
                            "--tokens", args.tokens, "--generate", str(generate),
                            "--ignore-eos", "--trace", str(trace)], check=True)
            compare(label + "C++ vs HF incremental", ref, json.loads(trace.read_bytes()), 1e-4, 2e-5, True)
    elif native_path:
        compare(label + "external trace vs HF incremental", ref, json.loads(native_path.read_bytes()), 1e-4, 2e-5, True)
    return ref


def report_drift(model, prompt, full, mixed, report):
    # Teacher-force the FP32 trajectory to isolate storage error. Comparing
    # free-running logits after a different generated token confounds the result.
    trajectory = prompt + full["generated"][:-1]
    same_history = full["generated"][:-1] == mixed["generated"][:-1]
    forced = mixed if same_history else cached_trace(model, trajectory, generate=0, split=True)
    err, routes, argmax_changes, routed = [], [], [], 0
    for i, (a, b) in enumerate(zip(full["steps"], forced["steps"], strict=True)):
        err.extend(x - y for x, y in zip(a["logits"], b["logits"], strict=True))
        if argmax(a["logits"]) != argmax(b["logits"]):
            argmax_changes.append(i)
        for layer, (old, new) in enumerate(zip(a["routes"], b["routes"], strict=True)):
            if old:
                routed += 1
            if sorted(old) != sorted(new):
                routes.append(dict(step=i, layer=layer, fp32=sorted(old), mixed=sorted(new)))
    drift = dict(
        weight_comparison="BF16/block FP8 weights, FP32 arithmetic vs FP32 weights and arithmetic",
        teacher_forced_ids=trajectory, logit_count=len(err), max_abs_logit_error=max(map(abs, err)),
        rms_logit_error=math.sqrt(math.fsum(x*x for x in err) / len(err)),
        argmax_changed_at_steps=argmax_changes, routed_layer_steps=routed, route_changes=routes,
        greedy_fp32=full["generated"], greedy_mixed=mixed["generated"],
    )
    if report:
        write_json(report, drift)
    print(f"DRIFT (HF vs HF, not a parity threshold): same {len(trajectory)} input IDs, "
          f"max |logit diff|={drift['max_abs_logit_error']:.6g}, RMS={drift['rms_logit_error']:.6g}; "
          f"argmax changed {len(argmax_changes)}/{len(trajectory)}, routes changed {len(routes)}/{routed}; "
          f"free greedy FP32={full['generated']} mixed={mixed['generated']}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="pinned HF source files and index")
    parser.add_argument("--model", required=True, type=Path, help="synthetic fixture directory")
    parser.add_argument("--make-fixture", action="store_true", help="write synthetic weights before validation")
    parser.add_argument("--mixed-model", type=Path, help="also validate BF16/FP8 fixture derived from --model")
    parser.add_argument("--make-mixed-fixture", action="store_true", help="write the derived mixed checkpoint")
    parser.add_argument("--tokens", default="1,5,8,13,21,34,8")
    parser.add_argument("--generate", type=int, default=4, help="fixed-length greedy continuation; no EOS stop")
    parser.add_argument("--reference-trace", type=Path, help="also save the validated HF trace as JSON")
    parser.add_argument("--mixed-reference-trace", type=Path, help="save the dequantized mixed HF trace as JSON")
    parser.add_argument("--mixed-compare-trace", type=Path, help="compare an existing mixed-checkpoint C++ trace")
    parser.add_argument("--drift-report", type=Path, help="save HF mixed-vs-FP32 drift metrics as JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runner", type=Path, help="execute C++ runner and compare its trace")
    group.add_argument("--compare-trace", type=Path, help="compare an existing C++ trace")
    parser.add_argument("--runner-arg", action="append", default=[], help="repeat for each additional runner argument")
    args = parser.parse_args()
    prompt = [int(s) for s in args.tokens.split(",")]
    args.prompt = prompt
    if not args.mixed_model and any((args.make_mixed_fixture, args.mixed_reference_trace,
                                     args.mixed_compare_trace, args.drift_report)):
        parser.error("mixed fixture options require --mixed-model")
    if args.mixed_compare_trace and args.runner:
        parser.error("--mixed-compare-trace cannot be combined with --runner")
    if args.runner_arg and not args.runner:
        parser.error("--runner-arg requires --runner")
    if not all(0 <= t < 64 for t in prompt) or args.generate < 0:
        parser.error("expected token IDs 0..63 and nonnegative --generate")
    cfg = tiny_config(args.source)
    if len(prompt) + max(args.generate - 1, 0) <= cfg["index_topk"]:
        parser.error("supply enough tokens to exercise sparse DSA selection")
    if version("transformers") != "5.8.1":
        parser.error("install transformers==5.8.1 for the pinned reference")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    upstream = pinned_model(args.source)
    print(f"Official {REVISION}; torch={torch.__version__}, transformers={version('transformers')}, "
          f"safetensors={version('safetensors')}, CPU FP32 eager")
    config = upstream.GigaChat4Config(**cfg)
    config._attn_implementation = "eager"
    model = upstream.GigaChat4ForCausalLM(config).eval()
    if args.make_fixture:
        make_fixture(args.model, cfg, model, args.source)
    fp32, weight_map = read_checkpoint(args.model, cfg)
    if any(t.dtype != torch.float32 for t in fp32.values()):
        raise ValueError("--model must contain only FP32 weights")
    model.load_state_dict(fp32, strict=True)
    full = validate(model, args, args.model, args.reference_trace, args.compare_trace)
    if args.mixed_model:
        load_mixed(args, cfg, fp32, weight_map, model)
        mixed = validate(model, args, args.mixed_model, args.mixed_reference_trace,
                         args.mixed_compare_trace, label="BF16/FP8 storage, FP32 compute: ")
        report_drift(model, prompt, full, mixed, args.drift_report)


if __name__ == "__main__":
    main()
