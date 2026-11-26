# GigaChat4 development status

**Experimental, source build only.** The 708B checkpoint's decoder graph runs
on reduced fixtures in Strata. A full checkpoint has not generated a token on
a V100 in the testing reported here. Do not select an existing Qwen pack or
advertise this model as a supported `strata launch` choice.

### Try the native runner

```sh
cmake -S engine/src/gigachat4 -B build/gigachat4 -DCMAKE_BUILD_TYPE=Release
cmake --build build/gigachat4 --parallel 4
ctest --test-dir build/gigachat4 --output-on-failure
build/gigachat4/strata-gigachat4 --model "$MODEL_DIR" --tokens 1,5,8 --generate 4
```

`MODEL_DIR` is either a local compatible safetensors checkpoint or the synthetic
fixture created by [the parity harness](PARITY.md). The runner does not download
files. To build from the main engine tree, add `-DSTRATA_BUILD_GIGACHAT4=ON`
and build target `strata-gigachat4`. For the actual tokenizer and chat template
use [the offline text frontend](TOKENIZER.md). For CUDA use the
[SM70 build instructions](HARDWARE_COMPATIBILITY.md) and pass `--cuda`.

### Evidence

| Gate | Outcome |
|---|---|
| Checkpoint structure | All 95,616 names in 45 real headers checked against names, shapes, byte spans, dtypes and scale grids. Three sampled real projection rows passed native-vs-PyTorch comparison (max absolute error `2.38e-7`). [Layout](LAYOUT.md) |
| Reduced model | Four-layer F32 and mixed FP8/BF16 fixtures ran through native GDN, DSA, MoE and output head. Ten steps per case. All selected expert sets and all generated tokens matched the official CPU reference using the same weights. Max absolute logit errors `7.53e-6` and `7.39e-6`. The mixed comparison uses dequantized weights with FP32 compute. [Parity](PARITY.md) |
| Caching | Range-bounded file reads; host and device residency budgets; direct device-hit path; optional file read-ahead for the next **already selected** expert. Tests exercise eviction, cache bypass, partial FP8 blocks, offsets above 4 GiB and truncated reads. [Weights](WEIGHTS.md) |
| Build and isolation | Standalone CPU, main-tree CPU and CUDA 12.8 SM70 compiled. On the CPU machine all four C++ component tests passed; the device-only test reported skip. Existing `model_family_test` and `gguf_geometry_test` passed and produced identical binaries with `STRATA_BUILD_GIGACHAT4` on and off. |
| Performance | [Tiny CPU measurements](BENCHMARKS.md) and a reproducible device protocol; no full-checkpoint throughput or quality claims. |
| Optional copy pipeline | `--cuda --overlap-copies` enables two staging slots on eligible nonresident projections. Compiled for SM70; actual concurrency and speedup require a V100 timeline. |
| Four-bit GGUF | [Standalone CPU probe](GGUF.md) locates a single routed projection in the separate 429.44 GB GGUF release. One real Q4_K block matched the independent decoder. This format is not yet an end-to-end input for `strata-gigachat4`. |

### Remaining gates

1. Obtain the target inventory and run the reduced FP8/SM70 parity test on the
   actual V100, with and without GPU residency. Check pinned allocation, peak
   memory and device events, including transfer boundaries and scale grids.
2. Verify storage on the model volume. The 45 source files require **719.72 GB**
   decimal before optional draft files or any conversion workspace. Test one
   short full-size prompt with a reference trajectory before profiling.
3. Quantify differences from the original dynamic FP8 activation kernels and
   BF16 intermediate rounding; this branch's weight decoding is not an
   implementation of those exact activation numerics. Investigate larger
   prefill batches, GPU GDN/DSA, compact latent KV, lower-bit expert packs and
   layer-aware residency only against matched correctness and timing runs.
4. Establish a same-hardware practical decode rate, repeatability and target
   context limit before proposing a catalog or serving integration. Speculative
   MTP/DFlash, padded multi-sequence batches, and text sampling beyond greedy
   decoding are outside the qualified path.

[Architecture](ARCHITECTURE_ANALYSIS.md) · [Implementation plan](IMPLEMENTATION_PLAN.md) ·
[Memory and locality](MEMORY_ANALYSIS.md) · [Hardware](HARDWARE_COMPATIBILITY.md) ·
[Benchmarks](BENCHMARKS.md)
