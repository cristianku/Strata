# GigaChat4 CPU numerical parity

`tools/gigachat4/reference_parity.py` runs the **official HF model code** at
[`ai-sage/GigaChat4-708B-A41B` revision `2baf8b0`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697)
on a tiny, seeded FP32 checkpoint and, optionally, a BF16/block-FP8 copy.
It never downloads or opens the original model weights. This checks the
forward equations, cache and weight decoding on synthetic data. It does not
measure model quality, hardware throughput or dynamic FP8 kernel accuracy.

## Reproduce on a Linux CPU

From the repository root, choose a working directory outside the checkout.
The only files fetched from HF are two Python files, the config and the
**index** (about 9 MB). The harness verifies their SHA-256 before importing code.

```sh
WORK=$(mktemp -d)
REF=$WORK/upstream
MODEL=$WORK/tiny
REV=2baf8b0cf71f1528c14e7a274877d9c8fbf22697
mkdir -p "$REF"
for f in config.json modeling_gigachat4.py configuration_gigachat4.py model.safetensors.index.json; do
    curl -fLsS --retry 2 "https://huggingface.co/ai-sage/GigaChat4-708B-A41B/resolve/$REV/$f" -o "$REF/$f" || exit 1
done

python3 -m venv "$WORK/venv"
PY=$WORK/venv/bin/python
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
"$PY" -m pip install 'transformers==5.8.1' 'safetensors==0.7.0'
"$PY" tools/gigachat4/reference_parity.py --source "$REF" --model "$MODEL" \
    --make-fixture --reference-trace "$WORK/hf.json"
```

`--make-fixture` requires an empty or new directory. Omit it when reusing the
fixture. Each run reads the saved tensors back via the safetensors index and
strictly loads them into the HF model. No tokenizer, CUDA, FLA or Triton is needed.

To compare a built executable, run:

```sh
"$PY" tools/gigachat4/reference_parity.py --source "$REF" --model "$MODEL" \
    --runner /absolute/path/to/runner
```

The default prompt is `1,5,8,13,21,34,8`, with four greedy tokens. Use
`--tokens 1,2,3 --generate 4` to choose another case. The harness passes
`--model`, `--tokens`, `--generate`, `--ignore-eos` and `--trace` to the runner.
For an existing trace pass `--compare-trace FILE` instead of `--runner`.

## What a pass checks

- Four blocks in checkpoint order: three Gated DeltaNet blocks with dense
  SwiGLU MLPs, then latent sparse attention with a shared expert and two of
  eight routed experts. The config keeps both MLA scaling factors, interleaved
  main RoPE, non-interleaved indexer RoPE, YaRN factor 32, pre/post gated
  norms, short convolution, router bias and final logit softcap.
- There are two safetensors shards with **official parameter names**, random
  learned scales and a nonzero random routing correction bias. Quantization
  and the auxiliary next-token prediction layer are disabled. Only the
  four decoder blocks and the ordinary LM head are in scope.
- Every HF forward checks actual eager DSA attention probabilities: no future
  position receives attention and exactly `min(2, eligible keys)` remain.
  The default prompt exceeds both the indexer budget and the convolution window.
- Three independent HF runs: incremental cached decoding; batched prompt prefill
  followed by cached generation; one uncached causal prefill over the prompt
  and generated inputs. All logits, post-residual layer outputs, attention
  outputs before post-normalization, routed expert IDs and greedy IDs are compared.
  Cached prefill also checks the recurrent state where HF exposes it.
- C++ is compared against **every step** of HF incremental decoding. Dense
  layers have `routes: []` entries; MoE IDs are compared without regard to
  order. The per-layer `recurrent` vectors flatten `[value_head, key_dim,
  value_dim]`, and are empty for DSA. A trace includes all prompt tokens
  and all generated tokens except the last (it has not yet been fed back).
- Numeric bounds are `abs(actual - ref) <= atol + rtol * abs(ref)`; HF vs HF
  uses `3e-5/1e-5` and C++ vs HF uses `1e-4/2e-5` (absolute/relative).
  Shapes, finite values, routes and greedy IDs must match. The tool exits
  nonzero on failure and prints peak absolute errors on success.

## Reference measurements

On Linux x86-64, Python 3.13.5, PyTorch 2.10.0+cpu (one thread),
Transformers 5.8.1 and safetensors 0.7.0, two fresh runs produced byte-identical
config, index, shards and HF trace. The fixture contains 128 FP32 tensors
(288,256 payload bytes). With the default prompt, the MoE visited seven of
eight experts and both HF comparisons generated `[53, 35, 40, 50]`.

| HF path vs incremental (10 steps) | Max abs attention error | Max abs block error | Max abs logit error | Max abs recurrent error |
|---|---:|---:|---:|---:|
| Batched prompt + cached decode | 7.33e-6 | 2.10e-5 | 6.20e-6 | 3.07e-6 |
| One uncached causal pass | 7.39e-6 | 2.52e-5 | 5.72e-6 | No state returned |

The short prompt `1,2,3` also passed both comparisons; four generated tokens
were `[63, 0, 43, 59]`. These are token IDs for random weights, not a text-quality test.

## BF16/block-FP8 storage

Derive a second indexed checkpoint **from the saved FP32 checkpoint**. On
every run, the harness reads both from disk and checks that their weights
are related by exactly the conversion below.

```sh
MIXED=$WORK/tiny-mixed
"$PY" tools/gigachat4/reference_parity.py --source "$REF" --model "$MODEL" \
    --mixed-model "$MIXED" --make-mixed-fixture \
    --mixed-reference-trace "$WORK/hf-mixed.json" --drift-report "$WORK/drift.json"

# Reuse both checkpoints; run and compare the native CPU executable on each.
"$PY" tools/gigachat4/reference_parity.py --source "$REF" --model "$MODEL" \
    --mixed-model "$MIXED" --runner /absolute/path/to/strata-gigachat4
```

Use `--runner-arg=--host-cache-mib --runner-arg=8` to pass extra arguments to
both invocations. To compare existing traces instead, use `--compare-trace
FP32.json --mixed-compare-trace MIXED.json`.

The pinned official weight index determines which matrices have a
`weight_scale_inv` companion. For each 128×128 block (including partial
blocks), set the FP32 scale to `max(abs(block))/448`, or 1 for an all-zero
block. Store `clamp(weight/scale, -448, 448)` as `torch.float8_e4m3fn`.
All 92 unscaled tensors are BF16 in the official headers, including the
router correction bias, norms and GDN time constants. The resulting name/dtype
map is checked against a digest of those official headers. There are **36 FP8
matrices, 36 FP32 scale grids and 92 BF16 tensors** (114,064 payload bytes).
Every matrix in this reduced checkpoint fits one scale block; this end-to-end
case does not cover indexing across multiple blocks.

Two different comparisons are reported:

1. **Parity, same stored weights.** Decode the mixed checkpoint back to FP32
   (`fp8.float() * stored_scale`; `bf16.float()` otherwise). Run the official
   class on CPU using those exact values, and compare its per-step outputs
   against the native runner reading the mixed files. The usual trace fields
   and FP32 tolerances apply. Neither side needs a dynamic activation FP8 kernel.
2. **Storage drift, different weights.** Feed the same FP32-model token
   trajectory to both HF models, then report logit differences, argmax changes
   and changed expert sets. Separately report both freely generated continuations.
   The drift is descriptive, not an acceptance threshold.

**Measured HF results (default prompt, same CPU environment as above):** two
independent conversions produced identical shards, index, config, mixed trace
and drift report. For the dequantized model, max absolute HF logit differences
were `3.81e-6` (cached prefill vs incremental) and `5.84e-6` (uncached vs
incremental); the corresponding routes and generated IDs matched. On 640
logits at the same 10 input positions, mixed-vs-FP32 RMS drift was `0.307919`
and max absolute drift was `1.48714`. Three of ten argmax IDs changed; the
expert set changed at step 9, layer 3 (`[1,2]` → `[0,6]`, zero-based).
Free greedy continuations were `[53,35,40,50]` (FP32) and `[58,8,44,59]`
(mixed). These measurements concern random synthetic weights only.
