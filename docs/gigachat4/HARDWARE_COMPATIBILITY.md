# What is verified on Volta?

**The binary builds for real SM70. A device run is a separate gate.**
The development machine is an x86-64 Linux CPU environment; it exposes no
NVIDIA device. Its RAM and disk are not the inference server's capacity.

| Capability | Decision and evidence |
|---|---|
| ISA and toolkit | CUDA 12.8.93 compiled both `strata-gigachat4` and the device tests with `CMAKE_CUDA_ARCHITECTURES=70`. `cuobjdump -lelf` on the executable listed `strata-gigachat4.1.sm_70.cubin`. CUDA 13 is not the Volta build target. |
| FP16 tensor cores | A V100 has FP16 tensor cores with FP32 accumulation. Strata already uses those for suitable dense batched GEMMs. This baseline performs single-token GEMV with FP32 arithmetic. Do not infer tensor-core utilization from the build architecture. |
| BF16 and FP8 | Storage only. The kernel expands BF16 bits, decodes FP8 E4M3FN and multiplies by the stored block scale. No SM80 BF16 MMA, SM89 FP8 operations or `cp.async` instructions are needed. Nor does the path narrow BF16 to FP16 and silently overflow. |
| Allocation and pinning | GPU cache is explicitly byte-bounded; use zero to test the streaming-only path. Pageable file data is copied to a bounded pinned buffer for DMA. Input/output vectors, scratch, the CUDA context and allocator usage must fit **in addition to** cache residency. Allocation failures are surfaced. |
| PCIe and I/O | Explicit streams and CUDA events count DMA payload and time. The effective link width/generation and storage device must be checked on the target. Local payload reads can be satisfied from OS page cache; those bytes are not NVMe bandwidth. |
| Sequence storage | GDN, causal conv, sparse-attention cache and activations other than the current projection live on the host. See [memory](MEMORY_ANALYSIS.md) before choosing a context. The checkpoint's 1M training context is not a qualified runtime limit. |

### Reproduce the build on a supported CUDA 12 host

```sh
cmake -S engine/src/gigachat4 -B build/giga4-sm70 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_BUILD_TESTS=ON
cmake --build build/giga4-sm70 --parallel 4
cuobjdump -lelf build/giga4-sm70/strata-gigachat4
ctest --test-dir build/giga4-sm70 --output-on-failure
```

Local cross-compilation used GCC 14.2 and Debian 12 glibc headers for NVCC
on a Debian 13 build host. CUDA 12.8's math headers conflict with Debian
13's newer libc declarations; compiling a CUDA canary against those unmodified
host headers failed. Using the older headers as an isolated host-compiler
sysroot resolved this. No global changes to glibc or the CUDA headers are required.
On a machine with no driver/device, `gigachat4_cuda_test` reports skip (77),
not a pass.

### Target inventory and qualification

The target inventory has been requested from the server operator. Until it
is available, VRAM size/free, negotiated PCIe link, usable NVMe space,
sustained read rate, available RAM, memlock limit and driver version are
**unverified**. Do not assume the 16 GB or 32 GB V100 variant, or x16 under load.
For the operator's run, capture `nvidia-smi -q`, `nvcc --version`,
`nvcc --list-gpu-code`, `free -b`, the cgroup memory limit, `ulimit -l`,
`lsblk -o NAME,MODEL,TRAN,SIZE,ROTA` and `df -B1` for the model volume.
First run the small device tests and the mixed fixture in
[PARITY.md](PARITY.md), then measure [cold/warm inference](BENCH_PROTOCOL.md).
No driver installation, clock adjustment or cache flush is part of this check.
