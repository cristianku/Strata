# GigaChat4 CUDA weight projections

`--cuda` moves **matrix × vector projections** to one NVIDIA GPU. The checkpoint stays in its original file format. Attention state, routing, nonlinear operations and token selection stay on the CPU. The default is still the CPU path.

## Build and test on a Volta machine

Use a CUDA 12 toolkit with an SM70 compiler target, a compatible host compiler and an NVIDIA driver. From the repository root:

```sh
cmake -S engine/src/gigachat4 -B build/giga4-sm70 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_BUILD_TESTS=ON
cmake --build build/giga4-sm70 --target strata-gigachat4 gigachat4_cuda_test
ctest --test-dir build/giga4-sm70 -R '^gigachat4_cuda$' -V
```

The test uses synthetic data. It compares all 256 E4M3FN encodings against an independent CPU formula, including the sign of zero, subnormals, ±448 and both NaNs. It also compares GPU products for all four dtypes, partial 128×128 scale blocks, cached tiles, uncached tiles spanning multiple transfers, and a row wider than the staging buffer. For the experimental overlap path it compares serial and two-stream outputs bit for bit, exercises both slots repeatedly across partial scale blocks, and checks the counts and bytes of the final outstanding tiles. Exit **77** means the device tests were skipped. Compiling an SM70 binary or passing the host decoder check does not establish device parity.

**Validation recorded 2026-11-26:** NVCC 12.8.93 compiled the backend and test with `-O3 -arch=sm_70` on Linux x86-64 (GCC 14, glibc 2.36 compile headers). The linked test completed the host check for all 256 encodings, then exited 77 because CUDA detected no GPU. Device results and V100 throughput were not measured in this run.

On the separate V100 target, the operator compiled `7d3944b` with NVCC
12.9.86. The same device test exited 0 and the reduced mixed checkpoint
passed against HF with and without residency. [Device results](BENCHMARKS.md).

With a local checkpoint and valid token IDs, for example:

```sh
build/giga4-sm70/strata-gigachat4 --model "$MODEL_DIR" --tokens 1,5,8 \
  --generate 4 --cuda --gpu-cache-mib 512
```

`512` is an example budget, not a measured hardware recommendation. `CUDA_VISIBLE_DEVICES` can select the GPU. For a reduced end-to-end comparison, use the fixture and `--compare-trace` procedure in [PARITY.md](PARITY.md) and produce the trace with the CUDA runner. The projection test alone does not test model generation or throughput.

## Computation

For a row `r` and column `c` of an FP8 tile, the kernel uses:

```text
w = decode_e4m3fn(raw[r, c]) * weight_scale_inv[r / 128, c / 128]
```

The checkpoint's `weight_scale_inv` is a **multiplier** here. The scale grid is row-major, with both dimensions rounded up. The loader starts FP8 row ranges on 128-row boundaries. BF16 uses Strata's bit expansion; F16 uses half-to-float conversion; F32 is read directly. FP8 uses integer bit operations. One warp computes each row with FP32 products, accumulation and a shuffle reduction. No native FP8/BF16 instructions or tensor cores are required. This does not emulate FP8 activation quantization. Floating-point reduction order can differ from the scalar CPU path.

## Memory and timing

- **Resident weights:** `--gpu-cache-mib` bounds the sum of raw weights, FP32 scales and alignment padding in the device LRU cache. One key identifies an immutable reader, tensor and row range. Before reading the weights or scales, the loader asks the backend to multiply from the resident copy. A hit avoids both host weight reads and weight DMA; only the input and output vectors cross the bus. On a miss the loader supplies the raw tile. Host-cache hits and storage reads are counted separately.
- **Streaming:** A tile larger than the budget bypasses the cache without evicting other entries. Zero budget streams every tile. Reusable device scratch holds at most 1 MiB of raw data per upload plus that slice's scales and alignment. Very wide rows are split at 128-column boundaries, with partial sums combined on the device. No expanded FP32 weight matrix is allocated.
- **Other allocations:** By default one 1 MiB pinned host buffer stages all DMA. Reusable device input and output buffers hold FP32 vectors sized for the largest call so far. These, stream scratch, allocator overhead and the CUDA context are outside the residency budget. Leave VRAM for them. A failed CUDA allocation or launch is reported as an error.
- **Synchronization:** The default path uses one explicit nonblocking stream and waits after each timed copy and kernel. Both modes return owning CPU vectors and keep no borrowed reader pointers.

### Experimental overlap (`--cuda --overlap-copies`)

The factory defaults to `make_cuda_backend(cache_bytes, false)`. With overlap explicitly enabled, a nonresident matrix that needs multiple whole-row sub-tiles can upload the next sub-tile while the preceding projection runs. Resident projections, small transfers, and rows too wide to pack with their scales stay on the serial path. Wide-row partial sums keep their original order.

On the first eligible call, the backend adds a nonblocking copy stream, per-slot timing events, a second 1 MiB pinned buffer, and a second 1 MiB device scratch buffer. It reserves at least 1 MiB in the first device scratch buffer too. Each pipeline slot fits both weights and scales in 1 MiB; serial fallback can grow the first scratch slightly beyond that for scales. Slot allocations are reused until the backend is destroyed. Unused overlap mode adds no pinned or device scratch allocation.

The compute stream waits for the slot's H→D event before reading its weights. Before reusing either buffer in a slot, the host waits for that slot's kernel-completion event and collects both timings. The final slots also retire before result download. An increase in `overlap_tiles` confirms this schedule was used; it does **not** prove the GPU ran a copy and kernel concurrently. Use a device timeline to check concurrency and compare end-to-end wall time on the same GPU, model and prompt. No speedup is claimed.

`metrics.cuda` in the runner output is cumulative for the backend's lifetime:

| Fields | Meaning |
| --- | --- |
| `cache_hits`, `cache_misses`, `cache_evictions` | One hit or miss per completed loader tile; an unsuccessful probe followed by an upload counts only once. A bypass counts as one miss. |
| `expert_cache_hits`, `expert_cache_misses` | Subset for routed `.experts.` tensor keys; excludes the shared expert. |
| `cache_budget_bytes`, `cache_bytes`, `peak_cache_bytes`, `cache_entries` | Configured budget, current and peak charged residency, and current tile count. |
| `streamed_tiles`, `kernel_launches` | Streamed sub-tiles and all projection launches, respectively. |
| `overlap_enabled`, `overlap_calls`, `overlap_tiles` | Opt-in flag, completed multi-tile projections using the two-stream path, and sub-tiles retired on it. `overlap_tiles` is a subset of `streamed_tiles`. |
| `pinned_bytes`, `stream_scratch_bytes`, `input_buffer_bytes`, `output_buffer_bytes` | Current payload allocation sizes outside the LRU, in bytes. |
| `h2d_bytes`, `d2h_bytes`, `weight_h2d_bytes` | DMA payload; `weight_h2d_bytes` is the weight-and-scale subset of H→D. Padding is never transferred. |
| `kernel_ms`, `h2d_ms`, `d2h_ms` | CUDA-event milliseconds for completed kernels and copies. Excludes CPU staging copies, file I/O and other host work. With two streams these intervals can overlap; their sum is not wall time. |
| `device` | CUDA device ordinal visible to this process. |
