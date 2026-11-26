# Checking the official 708B checkpoint layout

`tools/gigachat4/check_layout.py` checks the [official FP8 checkpoint at
`2baf8b0`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697).
It reads metadata, not weights. It does not need a GPU, PyTorch or Transformers.
The optional row comparison below is the only mode that requests weight bytes.

## Run offline

From the repository root, with Python 3.10 or later:

```sh
python3 tools/gigachat4/check_layout.py \
  --config "$META/config.json" --index "$META/model.safetensors.index.json" \
  --headers "$META/headers"
```

Use the original bytes of `config.json` and `model.safetensors.index.json`;
the checker pins both by SHA-256. Omit `--headers` for the `names_only` check.
That check verifies the complete name set and the index byte total, but reports
**zero headers checked**: an index does not contain a tensor's shape or dtype.

The header directory must contain all 45 files named
`model-00001-of-00045.safetensors.json` through `model-00045-of-00045.safetensors.json`.
Each can be a plain safetensors header object or the memory tool's cache record
`[payload_start, shard_file_size, header_object]`. Here `payload_start` includes
the eight-byte length prefix. A plain header does not attest to the size of
any local shard file. The checker never opens such files.

## Fetch metadata only

```sh
META=./giga4-metadata  # must not exist yet
python3 tools/gigachat4/check_layout.py --hf --save-metadata "$META"
```

The destination contains only JSON and is published after verification. Omit
`--save-metadata` to keep nothing. `--hf --headers "$META/headers"` reuses the
45 local headers and requests only the pinned config and index.

For a fresh metadata run, the 92 requested ranges are:

| File | Requested bytes |
|---|---:|
| `config.json` | 4,337 |
| `model.safetensors.index.json` | 9,157,175 |
| 45 eight-byte prefixes and the exact header after each | 12,276,816 |
| **Total response body bytes** | **21,438,328** |

Every request requires HTTP **206**, an exact `Content-Range`, an unchanged
total file size across reads, identity encoding and, if present, an exact
`Content-Length`. A `200` response is closed without reading its body. The
checker follows only HTTPS Hugging Face redirects, without consuming redirect
bodies. Each shard prefix must match the known header length before a second
request is sent. The largest header body is 6,227,160 bytes. A server or proxy
that ignores ranges causes an error; there is no full-file fallback.

## What was verified

The local check passed on the pinned config, index and all 45 cached headers:

| Scope | Names | BF16 | FP8 E4M3 | FP32 inverse scales | Tensor bytes described |
|---|---:|---:|---:|---:|---:|
| Globals and base decoder, layers 0–63 | 95,567 | 1,487 | 47,040 | 47,040 | 714,789,875,712 |
| Optional MTP, layer 64 | 49 | 37 | 6 | 6 | 4,912,973,280 |
| **Index total** | **95,616** | **1,524** | **47,046** | **47,046** | **719,702,848,992** |

- Expected names, shapes and dtypes come from `memory_budget.tensor_schema()`,
  the same inventory used for memory sizing. This checker verifies each tensor
  against its assigned shard, including every routed expert and every FP8
  scale grid. The grid for an `[out,in]` weight is `[ceil(out/128),ceil(in/128)]`.
  For example, the DSA `kv_a_proj_with_mqa` weight `[576,7168]` needs `[5,56]`
  FP32 scales, including the partial last row block.
- Each byte span must have the right length. Spans must start at zero, have
  no gaps or overlaps, and fill the reported shard size when supplied.
  A pinned digest of the canonical headers also catches equal-sized tensors
  whose offsets were swapped. Duplicate JSON keys fail validation.
- The checker accounts for the MTP layer separately. Ordinary next-token
  decoding does not need it. The separate DFlash checkpoint is outside the
  official base index and is never requested.
- This checks the **description** of the checkpoint, not payload hashes,
  numerical parity, generation quality, or the RAM needed to run the model.

## Optional: compare real rows against PyTorch

On little-endian Linux with `/proc`, build the small CPU probe with the same
`FileWeights` implementation as the decoder. Use a Python environment with
PyTorch and `torch.float8_e4m3fn` for this step:

```sh
c++ -std=c++17 -O2 -Iengine/src/gigachat4 -Iengine/third_party/strata-llama/vendor \
  engine/src/gigachat4/weights.cpp engine/tests/gigachat4/layout_projection.cpp \
  -o /tmp/giga4-layout-projection
python3 tools/gigachat4/check_layout.py --hf --headers "$META/headers" \
  --sample-projections 28896 --projection-runner /tmp/giga4-layout-projection
```

This opt-in grants an explicit **payload-only** budget. The plan needs 28,896
bytes; the hard ceiling is 32,768. It runs only after the full layout check.
The three additional exact ranges are layer 3's `kv_a_proj_with_mqa.weight`
rows 574–575 (14,336 bytes), their FP32 scale row (224 bytes), and BF16
`indexer.weights_proj.weight` row 63 (14,336 bytes). Each spans the original
7,168 input columns. The probe reads an anonymous in-memory fixture; no
weight bytes are saved to disk.

The native CPU projections and PyTorch's FP32 matvec use the same input,
`x[c] = ((c % 17) - 8) / 8`. PyTorch decodes FP8 and BF16, broadcasts each
FP32 scale over 128 columns, and checks all three outputs (`atol=rtol=1e-4`).
This exercises real weight decoding and scale placement for these rows only;
it does not test activation quantization, CUDA or a full decoder forward.

Measured on Linux x86-64 with PyTorch `2.10.0+cpu`: all three outputs passed,
maximum absolute error **2.384185791015625e-7**. This run reused the 45 header
records and read 9,190,408 HTTP body bytes (the two JSON files plus the three
payload ranges). A separate HTTP check fetched and decoded the first shard's
prefix and header with the same bounded reader.

## Tests

```sh
python3 tools/gigachat4/test_check_layout.py -v
# Also exercise the full official cache with no network:
GIGA4_LAYOUT_METADATA="$META" python3 tools/gigachat4/test_check_layout.py -v
```

Set `GIGA4_LAYOUT_HEADERS` if the header directory is elsewhere. The negative
tests cover transposed scales, wrong dtypes, missing or misplaced names,
overlapping offsets, malformed ranges, ignored ranges and oversized prefixes.
