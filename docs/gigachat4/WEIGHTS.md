# Reading GigaChat 4 weights

`FileWeights(dir, cache_bytes, backend)` opens a local checkpoint. It reads the
index and shard headers at startup. During inference it reads an embedding row,
a small tensor, or a tile of matrix rows from disk. It does not convert the
model, map the shards, or allocate a buffer the size of a shard.

## Names and format

The [pinned FP8 index](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/model.safetensors.index.json)
lists **95,616 tensor names across 45 shards**, totaling 719,702,848,992 payload
bytes. Lookups use the original name, without a translation table. For example:

| Request | Example name |
|---|---|
| Embedding row / output projection | `model.embed_tokens.weight` / `lm_head.weight` |
| Gated norm | `model.layers.0.input_layernorm.gate_down_projection.weight` |
| GDN / DSA | `model.layers.0.self_attn.conv1d.weight` / `model.layers.3.self_attn.indexer.wq_b.weight` |
| Router / routed expert | `model.layers.3.mlp.gate.e_score_correction_bias` / `model.layers.63.mlp.experts.255.down_proj.weight` |
| Shared expert | `model.layers.3.mlp.shared_experts.up_proj.weight` |

The index also contains 49 names under `model.layers.64.` for the extra
prediction layer. Indexing those names does not run speculative decoding.
Unindexed files, including a separate drafter, are not opened. For local
fixtures an unsharded `model.safetensors` is also accepted without an index.

Safetensors uses little-endian, row-major storage. The dense types are `BF16`,
`F32` (FP32), and IEEE `F16`. FP8 `F8_E4M3` is **E4M3FN**, not FNUZ: it has
signed zeros, subnormals down to 2⁻⁹, maximum magnitude 448, and NaNs at
`0x7f` and `0xff`. The reader also accepts the explicit `F8_E4M3FN` spelling.

For a matrix `W` of shape `[out,in]`, the FP8 scale tensor is named
`W_name + "_scale_inv"`: e.g. `up_proj.weight_scale_inv`. It must have shape
`[ceil(out/128), ceil(in/128)]`. Use the index to locate it: **46,309 of the
47,046 scale grids are in a different shard from their weights**. The last
blocks can be partial; neither weights nor scales have padding elements.

```
W[r,c] = decode_E4M3FN(raw[r*in+c]) * scale_inv[r/128, c/128]
```

The divisions in the subscripts are integer divisions. Despite the name,
**multiply** by the stored inverse scale; do not divide by it or transpose the
grid. For example, an up projection `[2048,7168]` has a `[16,56]` grid; a down
projection `[7168,2048]` has a `[56,16]` grid. All 47,046 grids in the pinned
headers use `F32`. The reader also permits dense BF16/F16 scales.

The [pinned config](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/config.json)
specifies `fp8`, `e4m3`, `[128,128]`, and dynamic activations. When a
`quantization_config` is present the reader checks its weight format and block
size. It reads the dtype of each tensor from its header; it does not infer
quantization from `ignored_layers`. CPU matvec uses FP32 inputs, dequantized
weights and FP32 accumulation. It does not emulate dynamic FP8 activation
rounding and is not a bitwise reference for an FP8 GPU GEMM.

## Memory, cache and backend contract

- `read(name, shape)` requires an exact shape and at most 64 MiB of decoded
  floats. `row(name, rows, cols, index)` reads only that row and its scales.
- `linear(name, x, out)` requires shape `[out,x.size()]`. It targets 16 MiB of
  raw weights per tile. Tile starts are multiples of 128 rows; only the last
  tile can end inside a block. If one block is larger than the target it is
  allowed up to **64 MiB**. A matrix too wide for that limit is rejected. The
  input and output vectors have separate limits of 64 MiB each.
- The optional synchronous `MatvecBackend::multiply` receives the raw bytes
  unchanged, a canonical dtype string (`F8_E4M3`, `BF16`, `F16`, `F32`), the
  tile dimensions, the whole input vector, and only that tile's scale rows as
  FP32. Index scales from local row zero; the scale vector is empty for dense
  tiles. Return exactly one float per tile row. The key includes a reader ID,
  the tensor name and `:rows=<first>:<end>` (exclusive end). Keys are stable
  within a reader and distinct across readers. The input buffers are borrowed
  for the call. A backend that retains weights must copy them.
- Before reading any host weights or scales, `linear` calls the backend's
  optional `multiply_cached` with the same key and the **current** input.
  A resident tile can return the result immediately with no host cache lookup
  or file read. A miss follows the normal host and `multiply` path.
- `cache_bytes=0` streams every request. A positive budget caches exact
  **raw byte ranges** (including scale ranges), in least-recently-used order.
  A range larger than the budget bypasses the cache. The limit covers resident
  payload bytes, not the index, allocator overhead, OS file cache, active tile,
  decoded scales, activations, or backend allocations. There is also a limit of
  65,536 cache entries.
- `stats()` reports shard/tensor counts, successful payload read operations
  and bytes, largest such read, wall seconds in successful payload seeks/reads
  (`payload_read_seconds`), host cache hits/misses, evictions, and current/peak
  cached payload bytes. A budget bypass counts as a host miss. The read timer
  excludes header parsing, allocation, dequantization and backend execution.
  The OS may satisfy reads from memory: bytes and seconds are not physical SSD
  measurements. One reader serves one calling thread; keep files unchanged.

### Optional Linux expert read-ahead

`prefetch_expert("model.layers.3.mlp.experts.55.")` advises the kernel that the
three projection ranges (`gate_proj`, `up_proj`, `down_proj`) and their scale
ranges may be needed. It uses `posix_fadvise(POSIX_FADV_WILLNEED)` on the
corresponding open shards. It requests at most **64 MiB total** per call.
If ranges do not fit, scale advice is capped and the rest is shared among
the projections. Only the prefix supplied by the caller is considered.

The reader never starts read-ahead on its own. The runner's `--prefetch` flag
opts in after routing to look ahead one **already selected** expert. Advice makes no
synchronous payload read in the reader, does not fill or promote entries in
its LRU, and does not copy anything to the GPU. The kernel may ignore the hint
or submit storage I/O while handling it; its page cache is outside the host
LRU budget. There is no throughput claim.

`prefetch_supported` indicates whether this Linux advice API was compiled in.
Elsewhere the call does nothing. `prefetch_requests` and `prefetch_bytes` count
attempts and requested bytes, including failed hints; `prefetch_errors`
counts failures. These are not payload-read or OS cache-hit counters.

## Validation

The reader checks index membership against each shard, rejects duplicate keys,
checks shape products and dtype sizes, and verifies every byte extent against
the actual file length. Overlaps, holes, unused trailing bytes, short reads,
missing/malformed scale grids, and mismatched requested shapes are errors.
File offsets are 64-bit. Metadata limits include 32 MiB per index/header,
128 MiB of total shard headers, 200,000 tensors, 256 shards and rank ≤ 8.
Only regular files and simple shard filenames are accepted; local Hugging Face
cache symlinks are supported. Use a trusted, immutable model directory. These
checks do not verify a cryptographic payload hash or model quality.

**Official metadata check:** 12,276,816 bytes of header prefixes and JSON were
retrieved by HTTP Range from all 45 shards at revision `2baf8b0`. The test
opened those exact headers with sparse placeholders set to the published file
lengths and the original index. All 95,616 names, byte extents and scale grids
validated: 47,046 `F8_E4M3`, 47,046 `F32` and 1,524 `BF16` tensors. No
official tensor payload was downloaded or used for a numerical result.

Run `ctest -R '^gigachat4_weights$' --output-on-failure` in the GigaChat 4
test build. The offline test generates small checkpoints with independently
checked values for all four dtypes, split-shard scales, partial blocks, backend
tile keys and data, no host reads on a resident-backend hit with a new input,
bounded read-ahead without a synchronous payload read, LRU promotion/eviction,
and malformed inputs. On POSIX it
also reads the last embedding row from a sparse fixture beyond byte 2³².
To validate the headers of an existing checkpoint without running inference,
pass its directory to `gigachat4_weights_test`.
