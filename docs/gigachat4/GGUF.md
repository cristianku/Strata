# GigaChat4-708B Q4_K_M: what Strata can read

**One routed expert projection works offline.** The [standalone CPU probe](../../tools/gigachat4/gguf_expert.cpp)
can extract the original packed matrix, decode one block, or multiply the matrix by an FP32 vector.
It does not run a layer or generate tokens. `strata-gigachat4` still uses [safetensors FileWeights](WEIGHTS.md);
the app, launcher, engine defaults and Qwen architecture checks are unchanged.

## What is in the nine files?

Inventory of [ai-sage's Q4_K_M, revision `4b705bf`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B-GGUF/tree/4b705bf01252a02ba5dabd6be197171744f1b55d/GigaChat4-708B-A41B-Q4_K_M).
Read all nine GGUF v3 tensor directories via bounded HTTP 206 ranges, checking exact `Content-Range`
and the file length. **6,239,586 header/directory bytes and one 144-byte weight block** were read from
the shards. The 2,018 tensor descriptors are unique by name; sizes computed from their types and
dimensions cover every shard's data section exactly. The bindings below account for **every
GGUF descriptor and all 48,570 native non-scale weight names**: names and dimensions were
cross-checked against the pinned FP8 safetensors headers, including the MTP layer. The
other 47,046 FP8 names are scale grids; they have no GGUF counterparts. No full shard,
expert or model download.

| Storage in the official package | Tensor count | Payload bytes |
|---|---:|---:|
| Q4_K (256 values / 144 bytes) | 689 | 335,423,471,616 |
| Q6_K (256 / 210) | 87 | 93,250,590,720 |
| Q5_0 (32 / 22; MLA key matrices) | 17 | 49,020,928 |
| F32 | 1,225 | 714,831,360 |
| **Payload** | **2,018** | **429,437,914,624** |

**On disk: 429,444,154,336 bytes = 429.444 GB = 399.951 GiB.** This includes 6,239,712 bytes of
aligned headers. The payload divides into **414,946,689,024** bytes of routed experts, **13,036,968,960**
of other trunk/global weights (including shared experts), and **1,454,256,640** of appended MTP
weights (44 `blk.64.*` tensors). Ordinary inference has 64 layers; `gigachat4.block_count=65`
includes `nextn_predict_layers=1`. The GGUF card identifies the **BF16** release as its base.
The FP8 [reference index/config](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697)
is a name/shape reference, not a byte-for-byte source for this quant.

For comparison, 707,721,990,400 trunk coefficients × 4 bits gives **353,860,995,200 bytes**:
no scales, mixed types, MTP or file metadata. The [sizing tool](../../tools/gigachat4/memory_budget.py)
estimates **398,442,996,224** for the hypothetical trunk with Q4_K routed experts and its `matrices`
fixed-weight policy, or **413,739,380,736** if smaller/non-routed tensors retain source storage (`source`).
Neither estimate is the official package. At top-8 across all 61 MoE layers, a cold read of just the
chosen routed matrices would be **12,967,084,032 bytes per token** (`8/256` of routed bytes).
That's a byte count, not measured disk traffic, memory use or speed.

## Routed experts: a stack, not a transpose

Strata requests `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` in `[out,in]` order.
The [reference converter](https://github.com/ggml-org/llama.cpp/blob/0541e50e2f/conversion/deepseek.py#L411-L459)
stacks in ascending expert-ID order along a new outer axis. GGUF **lists the fastest dimension first**.
Thus `[in,out,256]` describes C-order `[256,out,in]` bytes. There is no matrix transpose and no
gate/up fusion. `Q4_K_M` is the **recipe**, not a ggml tensor type.

| Projection | Directory tensor under `blk.L.` | GGUF axes | Observed type | Row bytes | One expert's matrix bytes |
|---|---|---|---|---:|---:|
| gate / up | `ffn_gate_exps.weight` / `ffn_up_exps.weight` | `[7168,2048,256]` | Q4_K in all 61 layers | 4,032 | 8,257,536 |
| down | `ffn_down_exps.weight` | `[2048,7168,256]` | Q4_K in 32 layers | 1,152 | 8,257,536 |
| down | same | same | Q6_K in 29 layers | 1,680 | 12,042,240 |

For expert `E`, output row `r`, input block `b` (all zero-based):

```text
row_bytes = (in / 256) * bytes_per_block
file_byte = shard.data_start + tensor.offset + (E*out + r)*row_bytes + b*bytes_per_block
```

Read the type per tensor. Both K-quants carry their scales in each block; **never** apply a
safetensors `weight_scale_inv`. The selected gate/up/down triplet occupies 24,772,608 or
28,557,312 bytes. A single layer can straddle a shard boundary. To choose the file for the probe:

| Shard | First routed matrix | Last routed matrix |
|---:|---|---|
| 1 | 3 down | 9 down |
| 2 | 9 gate | 16 down |
| 3 | 16 gate | 23 gate |
| 4 | 23 up | 30 up |
| 5 | 31 down | 37 up |
| 6 | 38 down | 45 down |
| 7 | 45 gate | 52 down |
| 8 | 52 gate | 58 up |
| 9 | 59 down | 63 up |

Within each layer, file order for the routed matrices is down, gate, up.

## Rest of the native weight map (inspection, not execution)

In this section, native stems are relative to `model.layers.L.`, GGUF stems to `blk.L.`. Add
`.weight` unless a full name or exception is shown. Two-dimensional weights preserve row
semantics even when their listed dimensions are reversed.

| Native | GGUF | Interpretation |
|---|---|---|
| `model.embed_tokens` / `lm_head` (global) | `token_embd` / `output` | Untied `[7168,128256]`, Q4_K / Q6_K. |
| `mlp.{gate,up,down}_proj` | `ffn_{gate,up,down}` | Dense layers 0–2, also MTP layer 64. Gate/up Q4_K, down Q6_K. |
| `mlp.shared_experts.{gate,up,down}_proj` | `ffn_{gate,up,down}_shexp` | A separate 2-D expert, not index 256. Gate/up Q4_K, down follows the layer's routed down type. |
| `mlp.gate.weight` / `mlp.gate.e_score_correction_bias` | `ffn_gate_inp.weight` / `exp_probs_b.bias` | FP32 `[7168,256]` and `[256]`; biased *selection*, unbiased sigmoid *weights*. |

**Gated norms.** Each pair of stems below expands to **three** tensors. Native suffixes
`.weight`, `.gate_up_projection.weight`, `.gate_down_projection.weight` correspond to GGUF
`.weight`, `_gate_up.weight`, `_gate_down.weight`. All three are F32 in this package. For width
`w`, GGUF axes are `[w]`, `[w,16]`, `[16,w]`. Apply RMS and gamma, then multiply the normalized
vector by `2 * sigmoid(down(SiLU(up(normalized))))`. No implicit `+1` on gamma.

| Native stem | GGUF stem | Where used |
|---|---|---|
| `input_layernorm` | `attn_norm` | Before attention |
| `post_self_attn_layernorm` | `post_attention_norm` | After attention, before the residual add |
| `post_attention_layernorm` | `ffn_norm` | Before the MLP; **not** `post_attention_norm` |
| `post_feedforward_layernorm` | `post_ffw_norm` | After the MLP, before the residual add |
| `self_attn.q_a_layernorm` / `self_attn.kv_a_layernorm` | `attn_q_a_norm` / `attn_kv_a_norm` | MLA latents, widths 1536 / 512 |
| `model.norm` (global) | `output_norm` | Before LM head |

**GDN** (48 recurrent layers). See the [native grouped layout](GDN.md), upstream
[QKVZ split](https://github.com/ggml-org/llama.cpp/blob/0541e50e2f/conversion/base.py#L83-L98)
and [GigaChat4 conversion](https://github.com/ggml-org/llama.cpp/blob/0541e50e2f/conversion/gigachat4.py#L181-L212).

| Native under `self_attn.` | GGUF | Required adaptation |
|---|---|---|
| `in_proj_qkvz.weight` | `attn_qkv.weight` `[7168,16384]` (24 Q4_K, 24 Q6_K), `attn_gate.weight` `[7168,8192]` (Q4_K) | GGUF outputs **all Q, all K, all V**, plus Z separately. For each key head `h=0..31` gather Q `[128h:+128]`, K `[4096+128h:+128]`, V `[8192+256h:+256]` from QKV; append Z `[256h:+256]` from the gate to form that native 768-row group. Here `[start:+n]` means `n` consecutive rows. |
| `in_proj_ba.weight` | `ssm_beta.weight`, `ssm_alpha.weight`, each `[7168,64]` Q4_K | Interleave projected pairs for each key head as `beta[2h:2h+2], alpha[2h:2h+2]`. Concatenating the two full outputs would give the wrong gates. |
| `conv1d.weight` | `ssm_conv1d.weight` `[4,16384]` F32 | Restore singleton native axis `[16384,1,4]`; channel order already all Q/K/V. |
| `A_log`, `dt_bias` | `ssm_a` (no suffix), `ssm_dt.bias`, both `[64]` F32 | **`ssm_a = -exp(A_log)`**, not a log. Use directly as the multiplier of `softplus(alpha+dt)` in log-decay, or adapt back before calling a native function that exponentiates `A_log`. |
| `norm.weight`, `out_proj.weight` | `ssm_norm.weight` `[128]` F32, `ssm_out.weight` `[8192,7168]` Q4_K | Headwise output RMS then `2*sigmoid(z)`; not a three-tensor GatedNorm. Value head `v` shares key head `v/2` (integer division). |

**DSA / MLA** (16 trunk layers, plus the MTP attention weights). `self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,o_proj,gate_proj}`
→ `{attn_q_a,attn_q_b,attn_kv_a_mqa,attn_output,attn_gate}`, all Q4_K. In a DSA layer
`attn_gate` is the attention-output gate, **not** GDN Z. For `self_attn.indexer.{wq_b,wk,weights_proj,k_norm}`
the bindings are `indexer.{attn_q_b,attn_k,proj,k_norm}`; the first three are Q4_K and the last
is an ordinary `[128]` F32 norm.

The real **intra-matrix transpose** is in `self_attn.kv_b_proj.weight` (`[16384,512]`).
The converter views it as `[64,256,512]`, splits 128 K rows and 128 V rows **per head**, then
transposes K's last two axes only. `attn_k_b.weight` thus has GGUF axes `[128,512,64]` (Q5_0);
`attn_v_b.weight` has `[512,128,64]` (Q4_K). To reproduce the native expanded projection,
transpose decoded K per head and alternate the K and V row groups, or change the matvec
contract to operate on the separate factors. Concatenating the packed bytes is incorrect.
`attention.head_count_kv=1`, `key_length=576`, `value_length=512` describe the **latent cache**;
`head_count=64`, `key_length_mla=192`, `value_length_mla=128` describe expanded attention.

**MTP.** `blk.64` is dense, non-recurrent and has its own MLA/indexer tensors.
`model.layers.64.{eh_proj,embed_tokens,enorm,hnorm,shared_head.head,shared_head.norm}` map
to `blk.64.nextn.{eh_proj,embed_tokens,enorm,hnorm,shared_head_head,shared_head_norm}`;
the same three-tensor expansion applies to each gated norm. These tensors are not needed by the probe.

## What the headers tell us, and what they don't

The first shard has 70 metadata entries, including `general.architecture=gigachat4`,
`general.file_type=15`, `general.quantization_version=2`, rank-16 norm gates, the 65-element
recurrent mask, a 64-element trunk indexer mask, the clamp arrays (10), final logit cap (30),
router (`256`, `8`, one group, scale `2.5`, normalization, sigmoid), SSM sizes,
YaRN parameters, token IDs, BPE tokens/merges and the chat template. Later shards carry
**only** `split.no`, `split.count`, `split.tensors.count`.

There is no missing directory or unknown storage type in this inventory. What isn't serialized
as independent keys includes the two sigmoid gate multipliers, pre/post norm and residual
equations, GDN packing/head repeat convention, MLA scaling and interleaved RoPE semantics,
and the indexer's separate norm epsilon. These are fixed by the pinned
[converter checks](https://github.com/ggml-org/llama.cpp/blob/0541e50e2f/conversion/gigachat4.py#L73-L129)
and [architecture code](https://github.com/ggml-org/llama.cpp/blob/0541e50e2f/src/models/gigachat4.cpp),
not by a generic GGUF type table. A native binding still needs those semantics, the mapping
above, tokenizer/template binding, and numerical checks against a reference.
No FP8 scale grids, calibration corpus, per-tensor quantization error, or numerical oracle logits
are supplied by the directories. Header identity is **not** a checksum of the weight payload.

## Run the isolated probe

Linux, 64-bit little-endian, C/C++17 compilers; no GPU, Python package or network needed. From
the repository root, choose a build directory outside the checkout:

```sh
B=$(mktemp -d)
H=engine/third_party/strata-llama/vendor/hash
cc -O2 -I"$H" -c "$H/sha256/sha256.c" -o "$B/sha256.o"
c++ -std=c++17 -O2 -Iengine/include -I"$H/.." tools/gigachat4/gguf_expert.cpp \
    engine/src/quant/types.cpp engine/src/quant/dequant.cpp "$H/hash.cpp" "$B/sha256.o" -o "$B/gguf-expert"

S="$MODEL_DIR/GigaChat4-708B-A41B-Q4_K_M-00001-of-00009.gguf"
"$B/gguf-expert" "$S" 3 173 gate pack "$B/gate.q4k"
"$B/gguf-expert" "$S" 3 173 gate block 517 7 "$B/block.f32"
# x.f32: exactly 7168 finite little-endian float32 values (2048 for down).
"$B/gguf-expert" "$S" 3 173 gate matvec x.f32 "$B/y.f32"
python3 tools/gigachat4/check_gguf_expert.py "$B/gguf-expert" "$S"
```

Use a trusted, immutable local file with its original shard name. Only the containing shard is
needed. The probe pins its length and SHA-256 of the header/directory before opening Strata's
read-only `GgufFile` mapping. It checks the expected stack shape, type and byte extent before
addressing a slice. Outputs must be new files. `pack` writes exactly one expert matrix, still in
original K-block format, **without a container header**. `block` writes 256 FP32 values. `matvec`
decodes one row at a time with Strata's `quant::dequant_row`, accumulates in FP32 and writes
`out` little-endian floats. It is a weight-only CPU reference. Strata also has Q4_K/Q6_K × Q8_K
[dot kernels](../../engine/include/strata/quant/dot.hpp); choosing those would quantize the
activations and requires a separate parity check. No routed GPU kernel or cache is wired here.

**Checks performed.** All 183 routed stack lookups, including the last block of expert 255,
succeeded on authentic headers with sparse placeholders at the real file lengths. The offline
test above inserts synthetic nonzero Q4_K and Q6_K blocks into a temporary sparse shard and
checks packed bytes, last-expert/row/column matvec, rejected indices/header and exclusive output.
It needs a filesystem with large sparse-file support; it does not copy a 48 GB shard.

For the one actual block checked, shard 1 range **7,167,424,464–7,167,424,607** is layer 3,
expert 173, gate, row 517, block 7. SHA-256 of the 144 packed bytes:
`7ba27c76825be5b859311e20e6a142fdb3128a78c908b9755c7d4ba5ad5b3ec4`.
All 256 decoded FP32 values matched the separate [gguf-py NumPy decoder](../../engine/third_party/strata-gguf-py/gguf/quants.py)
byte-for-byte (`block.f32` SHA-256 `935e63004a17ea2ae32d83ed3e83d66a86eca2e359b6367ec707384cedf372d3`).
No complete official projection output, FP8 comparison, end-to-end generation, throughput or
quality result was measured.
