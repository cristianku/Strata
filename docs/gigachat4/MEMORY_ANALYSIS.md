# GigaChat 4 708B: what must fit and what must move

**The instruction-tuned target alone occupies 665.70 GiB in the official mixed FP8 checkpoint.** Its routed experts account for 640.66 GiB. For one token, only 8 of 256 experts in each of 61 layers run: 20.02 GiB of routed weights and scales in the source format. The remaining 25.04 GiB of fixed weights include the shared experts. Keeping the fixed weights resident is a different problem from caching the routed ones.

These are **header-verified sizes and conditional bounds**, not a generation benchmark. Sizing uses the [pinned official checkpoint](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697) without reading its payload. A separate [row test](LAYOUT.md) read 28,896 actual payload bytes. No full-model routing hit rate, quantization quality, throughput or hardware fit was measured. GiB = 2³⁰ bytes; GB/s = 10⁹ bytes/s.

## Checkpoint inventory

“Base” below means the 64-layer target of the **instruction-tuned** checkpoint, not the separately published base-training model. The [config](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/config.json) and [index](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/model.safetensors.index.json) were checked against all 45 actual shard headers. For each name, the [calculator](../../tools/gigachat4/memory_budget.py) checks its shard, dtype, shape and contiguous byte extent. Across all shards: 47,046 FP8 weight tensors, 47,046 FP32 scale tensors and 1,524 BF16 tensors. `ignored_layers` in the config is not a substitute for inspecting dtypes.

| Disjoint group | Tensors | Non-scale coefficients | Stored bytes, including scales | Of which FP32 scales |
|---|---:|---:|---:|---:|
| Global: embedding, LM head, final GatedNorm | 5 | 1,838,914,560 | 3,677,829,120 | 0 |
| Base layers: attention, dense FFNs, routers and norms, excluding shared/routed experts | 1,683 | 15,464,985,856 | 17,839,601,664 | 3,198,464 |
| Base shared experts, always on | 183 | 2,686,451,712 | 5,372,903,424 | 0 |
| Base routed experts, all 15,616 | 93,696 | 687,731,638,272 | 687,899,541,504 | 167,903,232 |
| **Base target** | **95,567** | **707,721,990,400** | **714,789,875,712** | **171,101,696** |
| Optional MTP, `model.layers.64.*` | 49 | 2,513,325,184 | 4,912,973,280 | 27,872 |
| **All indexed tensors** | **95,616** | **710,235,315,584** | **719,702,848,992** | **171,129,568** |

The final byte count equals the index's `metadata.total_size`; the 8-byte prefixes and JSON headers add 12,276,816 bytes, giving **719,715,125,808 bytes** for the files. The optional MTP owns separately stored `embed_tokens.weight`, `shared_head.head.weight`, `eh_proj.weight`, its norms and one dense-FFN/DSA layer. Do not count these as target weights or assume they share storage. The [repository tree](https://huggingface.co/api/models/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/dflash_draft) also lists an optional `dflash_draft/model.safetensors` file of **6,479,425,008 bytes**; it is outside the index and **all** totals above. Skipping MTP at inference does not shrink source shards stored on disk.

Per base decoder layer, in **source bytes** (zero-based IDs):

| Layers | Count | Non-expert | Shared | All routed | Total layer |
|---|---:|---:|---:|---:|---:|
| 0, 1, 2: GDN + dense FFN | 3 | 1,031,520,768 | 0 | 0 | 1,031,520,768 |
| 3, 7, …, 63: DSA + MoE | 16 | 239,624,160 | 88,080,384 | 11,277,041,664 | 11,604,746,208 |
| Other layers from 4 through 62: GDN + MoE | 45 | 242,467,840 | 88,080,384 | 11,277,041,664 | 11,607,589,888 |

Selected names and shapes are below. Matrix order is `[out, in]`. The calculator's `inventory` command emits **every** exact name, shape, dtype and byte length without contacting the Hub.

| Name or prefix | Checked tensors |
|---|---|
| `model.embed_tokens.weight`, `lm_head.weight` | Each BF16 `[128256,7168]`, 1,838,678,016 bytes each; untied. |
| `model.layers.0.mlp.` | `up_proj.weight`, `gate_proj.weight`: BF16 `[18432,7168]`; `down_proj.weight`: BF16 `[7168,18432]`. |
| `model.layers.4.self_attn.` | `in_proj_qkvz.weight`: FP8 `[24576,7168]`; `out_proj.weight`: FP8 `[7168,8192]`; `in_proj_ba.weight`: BF16 `[128,7168]`; `conv1d.weight`: BF16 `[16384,1,4]`. |
| `model.layers.3.self_attn.` | FP8 `q_a_proj.weight` `[1536,7168]`, `q_b_proj.weight` `[12288,1536]`, `kv_a_proj_with_mqa.weight` `[576,7168]`, `kv_b_proj.weight` `[16384,512]`, `o_proj.weight` `[7168,8192]`, `indexer.wq_b.weight` `[8192,1536]`. `gate_proj.weight` `[8192,7168]`, `indexer.wk.weight` `[128,7168]`, `indexer.weights_proj.weight` `[64,7168]` are BF16. |
| `model.layers.3.mlp.gate.` | `weight`: BF16 `[256,7168]`; `e_score_correction_bias`: BF16 `[256]` (a registered buffer, not a trained matrix). |
| `model.layers.3.mlp.experts.0.` | FP8 `gate_proj.weight`, `up_proj.weight` `[2048,7168]`, `down_proj.weight` `[7168,2048]`. Corresponding `*.weight_scale_inv`: FP32 `[16,56]`, `[16,56]`, `[56,16]`. |
| `model.layers.3.mlp.shared_experts.` | Same three weight shapes, **BF16**, no block scales. |
| `model.layers.64.eh_proj.weight`, `model.layers.64.shared_head.head.weight` | BF16 `[7168,14336]` and `[128256,7168]`; draft only. |

Scale grids use `ceil(out/128) × ceil(in/128)`, not truncated dimensions. For example `model.layers.3.self_attn.kv_a_proj_with_mqa.weight_scale_inv` is `[5,56]`. Look up the scale by its own index entry; it can be in a different shard from the weight.

### What “41B active” counts

One routed expert has `3 × 7168 × 2048 = 44,040,192` coefficients. Per token, `61 × 8 = 488` distinct `(layer, expert)` pairs give **21,491,613,696** routed coefficients. Add **19,990,352,128** fixed coefficients (including shared): **41,481,965,824** with both embedding tables counted in full. A normal token looks up one input embedding row while projecting all output logits; counting just that row gives **40,562,633,984**. Both conventions round to the model card's 41B. The non-scale counts include the router's `61 × 256 = 15,616` correction-buffer entries. Excluding them gives **707,721,974,784** target parameter entries.

The corresponding source weight footprint of the selected full tensors is 48,387,194,880 bytes, or 46,548,531,200 counting one input row. That is **not** the minimum PCIe or disk traffic: the non-experts may be resident, and expert data may hit in two cache tiers. Conversely, adding CPU/GPU compute, KV traffic and prefill makes “41B times bits” an inadequate performance estimate.

## Weight format: ideal bits vs stored blocks

If **all** target coefficients occupied exactly `q` bits, without even one byte of scales, the target-only sizes for `q = 2 / 3 / 4 / 8 / 16` would be **164.78 / 247.17 / 329.56 / 659.12 / 1,318.23 GiB**. The 2–8 bit numbers are packing floors, not files.

Here are two explicit repack policies. **Routed only** changes the routed experts, keeping the other 25.04 GiB byte-for-byte as shipped. **Large matrices** also repacks the fixed 2-D matrices whose dimensions are both ≥256 and whose row width is divisible by the block size, except the router; head, embedding and shared experts qualify. Other non-routed weights are BF16. Repacked tensors replace their source FP8 scales. The real illustrative block sizes come from [GGML block definitions](../../engine/third_party/strata-gguf-py/gguf/constants.py) (`GGML_QUANT_SIZES`).

| Routed format | Values / bytes per block | Effective bits/value | Bytes/expert | Target GiB, routed only | Target GiB, large matrices | Cold routed GB/token |
|---|---:|---:|---:|---:|---:|---:|
| Official E4M3 + FP32 grid | 16,384 / 16,388 (full block) | 8.00195 | 44,050,944 | 665.70 | — | 21.497 |
| IQ2_XS | 256 / 74 | 2.3125 | 12,730,368 | 210.19 | 190.91 | 6.212 |
| Q2_K | 256 / 84 | 2.625 | 14,450,688 | 235.21 | 216.65 | 7.052 |
| Q3_K | 256 / 110 | 3.4375 | 18,923,520 | 300.26 | 283.57 | 9.235 |
| Q4_K | 256 / 144 | 4.5 | 24,772,608 | 385.32 | 371.08 | 12.089 |
| Q8_0 | 32 / 34 | 8.5 | 46,792,704 | 705.57 | 700.52 | 22.835 |
| BF16 | 1 / 2 | 16 | 88,080,384 | 1,306.04 | 1,318.23 | 42.983 |

Both expert matrix widths are block-aligned. The bytes include code/scale/minimum metadata internal to the listed formats. File indices, tokenizer, container headers, per-tensor padding, filesystem blocks, runtime slot guards and quantizer workspace are additional. Coexisting source and converted files require space for both; for example 670.28 GiB of official shards plus ~371.08 GiB for the Q4_K matrix policy, before temporary files. These are not sizes or quality measurements of a published GGUF. No low-bit pack, implementation of these kernels for GigaChat on Volta, or acceptance of the resulting outputs is implied.

## Attention and recurrent state

One cached sequence, no draft and no prefix sharing. The [official attention and cache code](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/modeling_gigachat4.py) matters here: the reference expands K/V **before** the cache update. The indexer retains one separate 128-wide post-RoPE key per position in each DSA layer. Top-2,048 limits selected positions per query, **not** stored history.

* **Reference layout, assuming BF16 K/V/index:** per token, across 16 DSA layers, `16 × 2 × [64 × (192 + 128) + 128]` = **659,456 bytes** (655,360 K/V + 4,096 index). The repeated rotary key is inside the expanded 192.
* **Proposed latent layout, same dtype:** retain one 512-wide normalized latent, one 64-wide rotary key and one 128-wide index key: `16 × 2 × (512 + 64 + 128)` = **22,528 bytes/token**. Needs reconstruction or an equivalent absorbed MLA kernel and parity tests. This is not the reference cache.
* **48 GDN layers:** FP32 recurrent `[64,128,128]` per layer = **192 MiB**; four BF16 convolution-history entries over `2 × 32 × 128 + 64 × 128 = 16,384` channels = **6 MiB**. Total **198 MiB/sequence**, independent of context. The PyTorch fallback maintains FP32 recurrence; charge another copy for each simultaneous branch or snapshot that needs one.
* **Current C++ runner, CPU or `--cuda`: charge all state to host RAM.** All 48 GDN layers carry **204 MiB/sequence** in FP32: 192 MiB recurrent plus 12 MiB of conv history. Each of the 16 [DSA layers](../../engine/src/gigachat4/dsa.cpp) stores `64 × 128` non-rotary K values, **one shared 64-value rotary tail**, `64 × 128` V values and one 128-value index key per position. With [FP32 host vectors](../../engine/src/gigachat4/model.hpp), this is `16 × (64 × 128 + 64 + 64 × 128 + 128) × 4` = **1,060,864 bytes/token**. The CUDA backend accelerates projections; it requires **no GPU residency for GDN, KV or index history**. At 8,192 positions this means 8.293 GiB of host state before host weight cache and scratch.

| History (tokens) | Reference BF16 + GDN, GiB | Latent BF16 + GDN, GiB | Current C++ logical host state, GiB |
|---:|---:|---:|---:|
| 8,192 | 5.225 | 0.365 | 8.293 |
| 32,768 | 20.318 | 0.881 | 32.574 |
| 131,072 | 80.693 | 2.943 | 129.699 |
| 1,048,576 | 644.193 | 22.193 | 1,036.199 |

Multiply by independent sequences; weights stay shared. These are **logical state sizes**, not peak RSS. The [C++ runner](../../engine/src/gigachat4/main.cpp) caps DSA row reservations at the request's `prompt_tokens + max(generate − 1, 0)` positions. A vector can have unused capacity between geometric growth steps; old and new allocations can coexist while it resizes. The PyTorch reference has different peaks: cache concatenation can hold old and new tensors at once, and the initial V view can retain its larger projection. The eager DSA indexer builds FP32 `[batch, query, history]` scores, masks and per-head score tiles; unchunked long prefill can exceed the KV budget alone. Model context length is not evidence that 1M fits on the example hardware.

## A concrete residency policy

The following **proposed whole-expert policies** let us test whether given hit assumptions are even possible. They are not the allocation policies of the current CLI. Assume serial one-token decoding, single device with separate host RAM, preallocated state on GPU, no draft, and in-place consumption of prepacked weights. No duplicated model arena. The example charges **2 GiB GPU** for driver, scratch and allocator, **8 GiB host** for OS, runtime and file-cache headroom, and **two top-8 expert staging buffers on each tier**, separately from its cache. At Q4_K these buffers cost **378 MiB per tier**. Reserves are adjustable assumptions, not measured peak scratch.

1. **GPU-fixed.** All global/layer non-expert and shared weights live in VRAM. GPU minimum = fixed weights + state + staging + reserve. Host minimum = staging + reserve. Use remaining bytes for complete expert slots (`floor(free / expert_bytes)`; at most 15,616). Hits in VRAM need no host or disk read; a GPU miss transfers one expert from RAM, first reading it from disk if absent there.
2. **Host-layer-fixed.** Keep globals resident on GPU and all layers' fixed weights resident in host RAM. Charge GPU for **two copies of the largest fixed layer** instead of all layers. Read each fixed layer across PCIe once per token, even on perfect routed hits. Host minimum additionally includes all fixed layer bytes. Stage fixed layers directly from that resident memory. This can fit a smaller GPU at a severe transfer cost.

Example at 8,192 tokens, Q4_K experts, one sequence. All RAM/VRAM capacities are GiB. Minima include reserves; cache slots fill the remaining budget. “Fits” here only means the **listed allocations** fit.

| GPU / RAM | Fixed placement and format | State | Minimum GPU / host GiB | GPU / host expert slots | Consequence |
|---|---|---|---:|---:|---|
| 16 / 64 | GPU-fixed, source | latent | 27.778 / 8.369 | — | Fixed allocation alone rules this out. |
| 16 / 64 | Host-layer-fixed, source | reference | 12.940 / 29.987 | 132 / 1,474 | Peak budget incl. slots 15.986 / 63.995; **23.213 GB/token** fixed H→D. |
| 16 / 64 | GPU-fixed, large matrices | reference | 18.391 / 8.369 | — | Even quantized fixed matrices do not fit this layout. |
| 16 / 64 | GPU-fixed, large matrices | latent | 13.532 / 8.369 | 106 / 2,411 | Peak budget 15.978 / 63.994. Both latent KV and quantized fixed-weight kernels required. |
| 48 / 64 | GPU-fixed, source | reference | 32.637 / 8.369 | 665 / 2,411 | Peak budget 47.980 / 63.994. Possible residency is not an observed hit rate. |

Use real available memory after other processes, cgroup limits, page cache and pinned allocation limits. Reprice any extra staging, dequantized copy, page alignment, batched/prefill scratch, output/logit tensors and layer- or sequence-level concurrency. For UMA, charge physical storage once; for multiple discrete GPUs use each card's local budget and link/topology, not a pooled VRAM number. Setup/loading/conversion can peak above steady decode.

### What existing Strata code does

* The released family's [ExpertCache](../../engine/include/strata/core/expert_cache.hpp) already keys slots by `(layer, expert)`, supports sized/guarded slots, profile fills and layer-local admission. [FileExpertSource](../../engine/src/core/expert_source.cpp) has mmap, pread/staging and cache-aware prefetch paths. CPU expert misses can be **computed on the CPU** instead of copied over PCIe; that exchanges weight DMA for host memory reads plus CPU compute and activation/result transfers. Do not use a Qwen profile, geometry or benchmark to claim a GigaChat hit rate.
* The new [GigaChat file reader](WEIGHTS.md) caches exact raw ranges (weight and scale ranges separately). Its [CUDA path](CUDA.md) caches matrix row tiles, not complete experts. A resident tile bypasses the host weight and scale reads; input and output vectors still travel. Fixed and routed tiles compete for the same budget. Staging, input/output vectors and the reader's active tile/decoded scales are outside cache limits. **Do not apply the whole-expert replay numbers as measured CLI transfers.** Compare the runner's actual logical read/DMA counters and device memory peaks with the policy model.

## PCIe 3.0 / NVMe limits for a single generated token

Let `D = 488 × expert_bytes`, `g` = GPU hit fraction, `h` = host hit fraction **conditional on GPU misses**. For the proposed GPU-compute policies:

```
PCIe bytes = fixed bytes streamed + D × (1 − g)
SSD bytes  =                       D × (1 − g) × (1 − h)
latency ≥ max(PCIe bytes / PCIe B/s, SSD bytes / SSD B/s)   # perfect overlap
latency ≥ sum of the two times                             # serialized I/O
```

PCIe 3.0 is 8 GT/s/lane with 128b/130b encoding. The one-way physical ceiling before packet overhead is **15.754 GB/s ×16**, 7.877 ×8, 3.938 ×4. For one Gen3 ×4 SSD and a ×16 GPU, Q4_K all-cold routed traffic alone takes **≥3.069 s of SSD** and **≥0.767 s of PCIe** even at those physical ceilings. For a more concrete illustration, assume 12 GB/s to the GPU and 3 GB/s total disk reads. These are input assumptions, not measurements of a drive or V100:

| All routed misses; fixed resident | H→D s/token at 12 GB/s (also RAM-hot, GPU-cold) | SSD s/token at 3 GB/s | Cold tok/s ceiling, perfect overlap |
|---|---:|---:|---:|
| IQ2_XS | 0.518 | 2.071 | 0.483 |
| Q2_K | 0.588 | 2.351 | 0.425 |
| Q3_K | 0.770 | 3.078 | 0.325 |
| Q4_K | 1.007 | 4.030 | 0.248 |
| Official FP8 + scales | 1.791 | 7.166 | 0.140 |
| Q8_0 | 1.903 | 7.612 | 0.131 |
| BF16 | 3.582 | 14.328 | 0.070 |

For Q4_K, a **hypothetical** `g=.75, h=.90` is 3.022 GB H→D and 0.302 GB SSD: floors **0.252 s overlapped** or **0.353 s serialized**, *if the cache is large enough*. All-GPU hits remove routed I/O, not the ~41B-active computation. In the host-layer-fixed source policy, fixed H→D alone takes ≥1.934 s/token at 12 GB/s, including on all-GPU routed hits. An initially cold process must first bring fixed weights into RAM/VRAM; these tables exclude startup.

Capacity matters even with an oracle: 488 distinct `(layer, expert)` keys per token means at most `min(gpu_slots,488)` demand hits from the GPU cache at the start of each token. The host can cover at most `host_slots` additional requests. In the 16/64 GPU-fixed Q4_K latent example, **106 slots cap GPU hits at 21.72%**. At least 382 experts or **9.463 GB/token** must cross PCIe, ≥0.789 s/token at 12 GB/s **even with a warm host**. Conversely, 5 tok/s in the GPU-fixed Q4_K example needs at least 80.15% GPU hits and no more than 4.96% of all routed bytes from disk at these assumed rates, before any compute. A larger cache allows but does not guarantee those rates.

The bounds omit packet/filesystem overhead, read amplification, page faults, IOPS, SSD thermals, shared-root-port contention, CPU DRAM bandwidth, GPU memory bandwidth, quant decode, routing dependencies and compute. Real throughput can be strictly lower. Page-cache hits are not physical disk reads. Batching may share weights across positions; these single-sequence decode numbers are not prefill or batched throughput.

## Reproduce and replay

From the repository root, using Python 3. No dependencies or model files needed for the offline inventory/planner:

```sh
python3 tools/gigachat4/memory_budget.py inventory --prefix model.layers.64.
python3 tools/gigachat4/memory_budget.py budget --format Q4_K --fixed matrices \
  --kv latent --context 8192 --gpu-gib 16 --host-gib 64 --gpu-hit 0 --host-hit 1
python3 -m unittest discover -s tools/gigachat4 -p test_memory_budget.py
```

The planner prints exact integer sizes, per-layer groups, both policies, capacity tests, and bandwidth floors as JSON. To independently re-check the source, give `verify` the pinned index file. It accepts only its known SHA-256 (`f358bf4a2f1570920a0ff9c6c37362babb09ab2ee139df09f3f900c6c475dc50`). It requests **only** bytes 0–7 and then bytes 8–(7 + declared JSON length), capped at 8 MiB per shard. It checks exact `206 Content-Range` before reading, rejects full responses and never falls back to fetching a shard.

```sh
python3 tools/gigachat4/memory_budget.py verify /path/to/model.safetensors.index.json
```

Pass `--routes routes.jsonl` to `strata-gigachat4` to record native routing for full-size profiling. The runner streams one JSON object per **MoE layer per processed position**, including the prompt. `token` is the zero-based input position, not a vocabulary ID; dense layers emit no line. For example:

```json
{"token":0,"layer":3,"experts":[12,43,7,200,3,18,77,90]}
```

Replay reads this JSONL directly by default. If the prompt contained **seven tokens**, use them to warm caches and prediction history, then score only subsequent decode inputs:

```sh
python3 tools/gigachat4/memory_budget.py replay routes.jsonl \
  --warmup-tokens 7 --format official --gpu-cache-gib 4 --host-cache-gib 32 --predict 8
```

The last generated ID has not been fed back, so it adds no route records. Leave `--warmup-tokens` at zero to score all processed positions from empty caches. Replay reads one line at a time and requires layers 3…63 exactly once per position, eight distinct IDs 0…255, and consecutive positions for one sequence. It rejects reduced fixtures, missing layers and truncated tokens. The separate native `--trace` diagnostic is for reduced fixtures.

Budgets here are **net routed-cache bytes**, after reserving fixed weights, state, staging and headroom. Two independent, initially empty global LRUs may hold duplicate copies. GPU is checked first; only GPU misses consult/update host. All eight demands in a layer see the starting residency, then hits are promoted and misses admitted in trace order. A miss may be computed from staging even with zero slots. Replay reports GPU hits, host hits, disk misses, all-eight-GPU layer events, useful bytes on each link and per-layer counts. These disk bytes model a cache below host with no other page cache.

`--predict N` scores the first N recorded IDs from the **previous token at that same layer**, before the current demands update either cache. It reports precision, nonresident candidates that would have helped, and extra bytes for unused candidates. It is a shadow prediction: **no prefetch is issued and demand hit rates do not change**. ID order is the trace order, not necessarily sorted by router probability. Compare against no prediction on identical chronological traces; keep fitted profiles disjoint from evaluation prompts. Record model/revision, format, batch, context, workload, cold/warm state, measured host/SSD/DMA traffic, TTFT, decode latency and peak memory before making a speed or locality claim.
