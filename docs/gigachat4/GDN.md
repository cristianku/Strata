# GigaChat4 Gated DeltaNet

The delta update is shared with Strata's Flash-Next GDN. **The value-to-key head mapping is not.** For GigaChat4, value heads 0 and 1 both use key head 0, then heads 2 and 3 use key head 1. Passing the checkpoint projections directly to the Flash-Next kernel would produce the wrong result even with the right tensor sizes.

## One GigaChat4 step

Source: the [official FP8 checkpoint, revision `2baf8b0`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697). Dimensions below come from `config.json` and the module declarations, names from `model.safetensors.index.json`. The reference equations are in `GigaChat4GatedDeltaNetWrapper`, `torch_causal_conv1d_update`, `l2norm`, `torch_recurrent_gated_delta_rule` and `GigaChat4GatedDeltaRMSNorm` in `modeling_gigachat4.py`.

| Under `model.layers.<n>.self_attn.` | File shape | Interpretation |
|---|---:|---|
| `in_proj_qkvz.weight` | `[24576,7168]` | 32 key-head groups of 768 rows: `[Q:128, K:128, V:256, Z:256]` |
| `in_proj_ba.weight` | `[128,7168]` | 32 groups of four rows: `[b0,b1,a0,a1]` |
| `conv1d.weight` | `[16384,1,4]` | Depthwise, no bias. Channels are **all Q, all K, all V**, not projection order. |
| `A_log`, `dt_bias` | `[64]` each | One log decay parameter and one time-step bias per value head |
| `norm.weight` | `[128]` | Per-value-channel scale shared across all 64 heads |
| `out_proj.weight` | `[7168,8192]` | Input is the flattened gated output of 64 value heads |

For each Q/K/V channel, append the raw projected value to its causal window; multiply oldest-to-newest by taps 0–3, sum, then apply SiLU. Start with zero history. Z, b and a bypass the convolution. For each key head, L2-normalize Q and K separately as `x / sqrt(sum(x²) + 1e-6)`. Divide the normalized Q by `sqrt(Dk)`. Do not normalize V.

For each value head `h`, take Q/K from `floor(h / (Hv/Hk))`. The following is per token; `S` has key rows and value columns:

```text
beta  = sigmoid(b)
decay = exp(-exp(A_log[h]) * softplus(a + dt_bias[h]))
D     = decay * S
delta = beta * (v - Dᵀ k)
S     = D + k deltaᵀ
o     = Sᵀ q
y     = (o / sqrt(mean(o²) + output_eps)) * norm.weight * sigmoid(z) * gate_scale
```

Mean and norm are per value head. The checkpoint's `output_eps` is `1e-6` and `gate_scale` is `2.0` **outside** the sigmoid. The C++ step receives the decoder's already-normalized input and returns the projected attention output before the decoder's separate post-attention GatedNorm and residual.

`LayerState` stores FP32 raw conv history `[C,taps]` and the FP32 recurrent matrix `[Hv,Dk,Dv]`. At the published dimensions, this is 65,536 + 1,048,576 floats per layer: 4.25 MiB, or 204 MiB for all 48 GDN layers of one sequence, independent of context length. These are calculated sizes for GDN state only. A fresh state is all zeros. Prefill advances this same state one token at a time. The step updates it in two passes over contiguous matrix rows, using one `Dv`-element correction vector, without a second state matrix or a repeated Q/K buffer. All three linear projections pass through `Weights::linear`; the GDN code never materializes their matrices.

## Static comparison with Flash-Next

This comparison is from code inspection at Strata `c23477403`. The sources are [compiled geometry](../../engine/include/strata/core/geometry.hpp), [layer wiring](../../engine/src/core/layer.cpp), [native preprocessing](../../engine/src/kernels/cuda/native_gdn_preprocess.cu), [native CUDA recurrence](../../engine/src/kernels/cuda/native_gdn.cu) and [legacy CUDA recurrence](../../engine/src/kernels/cuda/gdn.cu).

| Contract | Strata Flash-Next decode | GigaChat4-708B |
|---|---|---|
| Hidden; key/value heads; key/value width | 2560; 16/48; 128/128 | 7168; 32/64; 128/128 |
| Conv channels; flattened V | 10240; 6144 | 16384; 8192 |
| Projection order | Separate GGUF `attn_qkv`, `attn_gate` (Z), `ssm_alpha`, `ssm_beta`, `ssm_out` | Two group-packed inputs and one output, with rows as above |
| Conv and cache | Four causal taps, then SiLU. Native decode retains the last **three** raw samples per channel. | Same operator. Native GigaChat4 retains the official **four** samples; on the next step the oldest is dropped. |
| Head sharing | `h % Hk` for these CUDA kernels | `h / (Hv/Hk)` (integer division), matching `repeat_interleave` |
| Input gates | Sigmoid beta; log decay = `ssm_a * softplus(alpha + dt)` | Same algebra only if `ssm_a = -exp(A_log)`. Raw `A_log` is not `ssm_a`. |
| Recurrent layout | `[Dk,Hv,Dv]` in the native and legacy decode kernels | `[Hv,Dk,Dv]`, matching the PyTorch state |
| Update and query scale | Decay old state, correct predicted V, read updated state. Native CUDA scales after the readout; legacy CUDA uses pre-scaled Q. | Same update. Official fallback scales Q before the readout. Moving the scalar is algebraically valid, not bit-identical in FP32. |
| Output norm | Headwise RMS, shared gamma, `sigmoid(z)` | Headwise RMS, shared gamma, **`2 * sigmoid(z)`** |

The [split kernel](../../engine/src/split/src/gdn_core.cuh) has another ABI: its recurrent layout is head-major, but its inputs are flat `[Q,K,V,Z]` and `[alpha,beta]`, it uses a supplied head map, and its convolution history is parity-buffered. A matching state layout alone does not make this a checkpoint-compatible path.

## Reproduce the checks

From the repository root (`REF` is the local pinned source directory):

```sh
g++ -std=c++17 -O3 -Iengine/third_party/strata-llama/vendor -Iengine/src/gigachat4 \
  engine/src/gigachat4/gdn.cpp engine/tests/gigachat4/gdn_test.cpp -o /tmp/gigachat4-gdn-test
/tmp/gigachat4-gdn-test "$REF/config.json"
# Optional: regenerate the test constants using only CPU PyTorch and the pinned Python source.
OMP_NUM_THREADS=1 python3 engine/tests/gigachat4/gdn_oracle.py "$REF"
```

The tiny FP32 test compares six projected token outputs and the final recurrent state to the official chunk implementation. Its oracle generator also checks the official single-token cached path against that chunk. Two key heads, four value heads, unequal key/value widths, distinct convolution taps and a zero input inside the sequence exercise the ordering and retention rules. Output tolerance is `2e-6` absolute; final state tolerance is `1e-6`. Every raw history entry and a cache snapshot/resume are checked exactly. A separate two-token test uses the full checkpoint geometry with synthetic projections and checks cache allocation and reuse. These tests do not run the 708B weights or establish FP8, FLA or GPU parity.
