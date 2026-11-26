# GigaChat4 sparse attention

GigaChat4 selects **individual causal token positions**, then runs multi-head latent attention (MLA) over those positions. Its indexer does not pool tokens. The 708B checkpoint uses this block in 16 decoder layers: 3, 7, …, 63 (zero-based). Each query may use up to 2,048 keys, including its own if selected.

Sources: the [checkpoint config](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/config.json), [attention, indexer, norms and rotary code](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/modeling_gigachat4.py), and [`_compute_yarn_parameters` in Transformers 5.8.1](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/modeling_rope_utils.py#L327-L459). Version 5.8.1 is recorded in this checkpoint's `config.json`. The dimensions below were also checked against the FP8 checkpoint's Safetensors **headers**.

## What one token computes

`x` is the decoder's already-normalized 7,168-channel input. File matrices are `[out,in]`. All matrix projections, including the small norm gates, call `Weights::linear`; the attention code does not read projection matrices into host vectors.

| Stage | 708B contract |
|---|---|
| Query | `q_a_proj` → 1,536, GatedNorm, then multiply by `sqrt(7168/1536)`. Both `q_b_proj` → `64 × 192` and `indexer.wq_b` → `64 × 128` consume **this scaled latent**. |
| Key and value | `kv_a_proj_with_mqa` → `[512 latent, 64 rotary]`. GatedNorm and `sqrt(7168/512)` apply **only to the latent**. `kv_b_proj` → `64 × [128 non-rotary K, 128 V]`. The same 64-channel rotary key serves every head. |
| GatedNorm | First compute `z = gamma * x / sqrt(mean(x²) + 1e-6)`. Return `z * 2 * sigmoid(Wdown · SiLU(Wup · z))`. The gate bottleneck has 16 channels, including on reduced configs. |
| Indexer key | `indexer.wk(x)` → 128, then **plain RMSNorm** with its own gamma and epsilon (default `1e-6`); no centering, bias, or model-wide norm gate. Rotate the **first** 64 channels and cache the result. |
| Index score | For each eligible key `j`, compute `sum_h((weights_proj(x)[h] / sqrt(64)) * ReLU(dot(index_q[h], index_k[j]))) / sqrt(128)`. Head weights are signed. Select the highest scores **after** this reduction, once for all MLA heads. The index scores determine a mask; they are not added to the attention logits. |
| Attention and output | Per head, use `[128 non-rotary, 64 rotary]` for Q/K. Take softmax of the scaled dot products on selected keys and mix V. Flatten `64 × 128`, multiply elementwise by `sigmoid(gate_proj(x))`, then apply `o_proj` → 7,168. The decoder owns the post-attention norm and residual. |

## RoPE: same frequencies, different pairs

MLA's `rope_interleave=true` reads adjacent input pairs `(x[2i], x[2i+1])`. The indexer **always** reads split-half pairs `(x[i], x[i+D/2])`. Both write all real components followed by all imaginary components. For either pair `(a,b)`, write `a*cos - b*sin` and `b*cos + a*sin`. Leaving MLA in adjacent-pair order would change the cache layout even if a particular dot product happened to agree.

For rotary width `D=64`, base `100000`, extension `F=32`, original length `32768`, the correction band is **pairs 14 through 24**. Transformers computes the boundaries from `D * log(original_length / (2πβ)) / (2 log(base))` using floor for β=32 and ceil for β=1. The pair ramp `r` goes from 0 to 1 over that band, clamped outside it. With `f=base^(2i/D)`, the inverse frequency is `(1-r)/f + r/(F*f)`. Multiply that FP32 frequency by the absolute position before taking sine/cosine.

There are two distinct magnitude factors. For the published `mscale=mscale_all_dim=1`, the Transformers rotary magnitude is the ratio `m(F,1)/m(F,1) = 1`, where `m(F,s)=1+0.1*s*ln(F)` for `F>1`. The model separately multiplies the **entire** MLA logit by `m(F,1)² / sqrt(192) ≈ 0.13086080`. Multiplying the rotary subspace by `m(F,1)` instead gives the wrong relative weight to the non-rotary subspace. Reduced configs with unequal factors also exercise the non-unit rotary magnitude.

## Cache, causality and cost

`dsa_step` accepts contiguous positions starting at zero, with one `LayerState` per layer and sequence. A token sees its own projected row plus the previous rows, never a later token in a prefill batch. A row becomes part of the state only after its output projection succeeds. The usable length is bounded by `max_position_embeddings` and by `max_context` if a smaller limit is supplied in the native config. Exceeding the limit fails; it does not silently convert full-history attention to a sliding window.

| Per-token FP32 cache in one 708B DSA layer | Floats |
|---|---:|
| `keys`: all non-rotary K, then **one** rotated tail | `64 × 128 + 64` |
| `values`: all V heads | `64 × 128` |
| `index_keys`: one rotated index key | `128` |
| **Total** | **16,576 (66,304 bytes)** |

This is an expanded K/V CPU path, not a compressed-latent cache. Storage grows on demand, geometrically, up to the context limit. At 4,096 tokens the **used payload** in all 16 DSA layers is about 4.05 GiB per sequence. At the advertised million-token limit it would be about 1.01 TiB, before weights, recurrent state, scratch or allocator overhead. These are calculated sizes, not measured memory use. Top-k limits the per-query attention work, **not** this retained history.

For the first 2,048 eligible keys no selection can change the mask. Native code still caches the index key but skips query/score projections for that step. After that, it scans the index history, retaining at most 2,048 candidates in a heap, and runs per-head softmax only on selected keys in chronological order. No sequence-by-sequence score matrix is allocated. Exact ties at the cutoff prefer the earlier position; PyTorch's `topk` does not guarantee a portable ordering for tied entries.

## Static reuse findings

These are comparisons of source contracts at Strata `c23477403`, not GPU measurements:

* [Flash-Next QSA CUDA indexer](../../engine/src/kernels/cuda/native_qsa_indexer.cu): the 128-channel RMS and split-half rotation on the first 64 channels look similar. But it rounds raw keys through FP16, **averages four tokens**, and rotates the pooled key at the block's starting position. Its compiled geometry has four index query heads. GigaChat4 normalizes/rotates every raw key separately, at its own position, and uses 64 index query heads.
* [Hy4-Flash HIP indexer](../../engine/src/hy4flash/run/hy4_indexer.hip.h): the weighted sum of ReLU dot products is useful. Its keys, however, are **gated pools of four**; it uses 16 heads, rotates the *last* 64 channels and includes a current group in selection. Those are not GigaChat4's selection or cache contracts.
* [Vendored DeepSeek 3.2 graph](../../engine/third_party/strata-llama/src/models/deepseek32.cpp): it has per-position index scoring and causal top-k, making that reduction a closer match. It normalizes index keys with **LayerNorm plus bias**, applies a Hadamard transform, uses ordinary RMS on the query latent, and loads split per-head `wk_b` / `wv_b`. GigaChat4's published indexer uses RMS without bias or Hadamard; its MLA uses gated latents, explicit alphas, a packed KV projection, and an output gate. Reusing the whole graph by renaming tensors would therefore change the model.
* [Split engine rotary helper](../../engine/src/split/src/rope.hpp): the frequency ramp is similar. Its simple YaRN constructor directly scales cos/sin by `1+0.1 ln(F)`. That is not the 708B rotary magnitude. Nor is multi-stream M-RoPE interleaving the main attention's even/odd **input channel** permutation.

## Test evidence

From the repo root:

```sh
g++ -std=c++17 -O2 -Iengine/third_party/strata-llama/vendor -Iengine/src/gigachat4 \
  engine/src/gigachat4/dsa.cpp engine/tests/gigachat4/dsa_test.cpp -o /tmp/gigachat4-dsa-test
/tmp/gigachat4-dsa-test
```

The C++ test uses FP32 outputs from the pinned Python `GigaChat4Attention.eval()` with CPU PyTorch 2.10.0 and Transformers 5.8.1. Both weights and inputs are deterministic binary fractions (`sample()` in the test). We compared one eager, nine-token forward with an explicit causal mask against nine cached steps in Python before recording its outputs. Native sequential steps match the batched outputs within `2e-5` absolute. With top-k 2, the reference selects positions `[0,4]` at query 7 and `[4,8]` at query 8: this distinguishes selection from a sliding window and from forced inclusion of self. The tests also compare both post-RoPE cache layouts, a reduced dense MLA variant, published-width rotation at position 32,768, and a two-token synthetic projection-shape check at the full 708B dimensions. See [PARITY.md](PARITY.md) for the reproducible whole-decoder comparison.

The standalone test passed on Linux x86-64 with GCC 14.2, including an AddressSanitizer/UndefinedBehaviorSanitizer build. No 708B weight payload, long-context inference or GPU attention benchmark was used for these checks.
