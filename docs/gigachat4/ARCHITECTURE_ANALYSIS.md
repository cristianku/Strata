# GigaChat4-708B-A41B: architecture contract

This is an opt-in engine experiment, not a model in the installer catalog.
The reference is the [official FP8 repository at `2baf8b0`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697).
Read `config.json`, `model.safetensors.index.json`, `configuration_gigachat4.py`,
`modeling_gigachat4.py`, `generation_config.json` and `chat_template.jinja`
together. Configuration defaults do not necessarily describe these weights.
Strata comparisons below are source inspection at `c23477403`, not hardware results.

| Path | Contract of the released checkpoint | Implication for Strata |
|---|---|---|
| File layout | Safetensors `[out,in]` matrices, individual named gate/up/down per expert. 95,616 names; 719,702,848,992 tensor bytes in the main index. The separate DFlash file is not part of that index. | The existing GGUF / `.strata` pack mapping is not interchangeable. Use bounded row tiles and read embeddings one row at a time. |
| Decoder | Width 7,168; 64 base layers. Full attention at 3, 7, …, 63. Dense FFNs in layers 0–2 (width 18,432); MoE in 3–63. Attention type and FFN type are independent. | Opt-in shape engine, following the separate `strata-inkling` / `strata-hy4` build pattern. Never send this graph through the Flash-Next default. |
| Routing | `GigaChat4TopkRouter`: FP32 dot, sigmoid, select 8 of 256 using **score + correction bias**. Weight each selected expert using its **original score**, normalize by selected sum + 1e-20, multiply by 2.5. `n_group=topk_group=1`; group masking selects everything. | `native_router.cu` specializes softmax, 512 → 10. Changing only `k` would silently change results. The new router uses the lower ID on exact score ties; PyTorch top-k does not guarantee tie indices. |
| Experts | One separate, always-on shared expert, **no extra sigmoid**. Routed and shared width 2,048. Clip gate above +10, clip up to ±10, multiply `SiLU(gate) * up`, apply down. Add the shared output exactly once. | Projection storage/cache can be shared across types. A fused unbounded SwiGLU or gating the shared expert would change the math. |
| Norm and residuals | Each attention and MLP branch gets GatedNorm **before and after**, then residual addition. GatedNorm = weighted RMSNorm with epsilon 1e-6, multiplied by `2 * sigmoid(Wdown SiLU(Wup norm))`. Rank 16, no biases. Final norm also gated. | Plain RMSNorm and the GDN per-head gate cannot implement this on their own. |
| GDN | 48 layers; 32 Q/K and 64 V heads, K=V=128; depthwise causal conv width 4, learned decay, delta update with FP32 state. | Strata's GDN primitives are candidates for optimization, not a compatible checkpoint binding. See [GDN](GDN.md). |
| DSA | 16 sparse MLA layers; Q rank 1,536, KV rank 512, 64 heads, 128 non-rotary + 64 rotary Q/K, 128 V. Learned 64-head, 128-wide indexer selects at most 2,048 causal positions. | QSA's block scoring is not the same indexer. Projecting full K/V and keeping index keys reproduces eager semantics at substantial context cost. See [DSA](DSA.md). |
| Output | Untied `[128256,7168]` LM head; `30 * tanh(logits / 30)` before sampling. BOS 1; EOS/PAD 2. | No embedding tying or default Qwen prompt. The test runner uses greedy decoding, not a claim about sampling quality. |
| Inference | Official `use_cache=false` by default; with cache the model carries conv + recurrent state and attention + index history. An ordinary next-token call uses base layers, not the optional MTP/DFlash drafters. | Sequential causal prefill is a baseline for cached decode; no speculative branch, batching, prompt/prefix reuse or model alias. |
| Numeric formats | Released FP8 E4M3 with block `[128,128]` FP32 `weight_scale_inv`, dynamic activations in the quantization config, plus mixed unquantized tensors. | Weight-only dequantized FP32 accumulation is possible on SM70. This is **not** bitwise parity with a native dynamically quantized FP8 activation GEMM. BF16 storage is not a requirement for native BF16 arithmetic. |
| Tokenization | Model-owned tokenizer and Jinja template: default system turn, distinct reasoning/instant generation suffixes, role and message delimiters, tool schema rendering. | Use the local template before tokenization, or pass token IDs for numerical tests. See [tokenizer](TOKENIZER.md). |

**Licenses.** Strata `LICENSE` is MIT (2026 Niko1221 and the Strata contributors).
The pinned [model license](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/2baf8b0cf71f1528c14e7a274877d9c8fbf22697/LICENSE.txt)
is MIT (2026 Giga). Retain applicable copyright and permission notices when
redistributing copies or substantial portions. No checkpoint, converted weights,
tokenizer vocabulary, or official Python modules are included here. The
vendored JSON reader retains its existing third-party notice.
