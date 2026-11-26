# GigaChat4 measurements

**A single V100 passed the reduced-model correctness test. There is no measured
full-708B token rate.** The two experiments below use synthetic weights. The
[memory analysis](MEMORY_ANALYSIS.md) gives separate full-model transfer bounds.

### Measured: streaming vs residency on one V100

The target operator built commit `7d3944b78209b0fbccb478df7fefba087953aeb4`
with CUDA 12.9.86 for SM70 and ran the [mixed-fixture comparison](PARITY.md)
on **one Tesla V100 PCIe 32 GB**. CTest passed 5/5 without skips; the standalone
CUDA test also exited 0, including its serial vs two-stream projection check.

The two native inference runs used the same seven input IDs, 1 MiB host cache,
four greedy output IDs, and a per-step diagnostic trace. The operator's results:

| GPU weight cache | Streamed sub-tiles | Tile hits / misses | Weight + scale bytes H→D | TTFT incl. load (ms) | Prefill (ms) | Decode, 3 steps (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 0 MiB | 756 | 0 / 756 | 973,512 | 188.21 | 38.39 | 15.87 |
| 1 MiB | 0 | 668 / 88 | 104,056 | 181.91 | 31.70 | 11.44 |

Both runs selected the same experts as the HF oracle at all ten positions and
generated `[58, 8, 44, 59]`. Maximum absolute logit error against HF was
`6.95e-6`. The two device traces were byte-identical (SHA-256
`30a594d3eb22319052fa1b8ff879ba4291863a5b20706005855b6784643bfaea`).
With cache enabled, 104,056 weight bytes remained resident. These are weight
transfers, not total PCIe traffic, physical NVMe reads or total process VRAM.
There was only one timed run per setting; the byte counts verify cache behavior,
but the times are not a full-model benchmark or evidence of an overlap speedup.

### Measured: small mixed checkpoint, CPU

Linux x86-64, Intel Xeon Gold 6240, GCC 14.2 Release, single native sequence,
no GPU. Four layers, width 32, two of eight routed experts, seven prompt IDs,
32 generated IDs (31 timed decode steps). Synthetic fixture: seed 70841, mixed
FP8/BF16 with 114,064 payload bytes, prepared as in [PARITY.md](PARITY.md).
No attention/state trace or prefetch. Each row is a fresh process. `fincore`
reported 139,264 resident page bytes for 132,424 bytes of shard files
before **every** run, including the first. This was not a disk-cold experiment.

| Host cache MiB | Run | TTFT incl. load, ms | Prefill tok/s | Decode tok/s | Payload read time, ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2.569 | 5,095 | 5,315 | 2.727 |
| 0 | 2 | 2.528 | 5,139 | 5,313 | 2.730 |
| 0 | 3 | 2.514 | 5,164 | 5,202 | 2.793 |
| 0 | 4 | 2.773 | 4,784 | 5,124 | 2.816 |
| 1 | 1 | 2.248 | 6,747 | 9,075 | 0.131 |
| 1 | 2 | 2.195 | 6,882 | 8,985 | 0.130 |
| 1 | 3 | 2.216 | 6,671 | 8,963 | 0.125 |
| 1 | 4 | 2.146 | 6,997 | 9,051 | 0.125 |

With no application cache, each run made 4,858 logical payload reads totaling
3,823,144 bytes. With 1 MiB, each made 186 reads totaling 111,440 bytes
(4,672 host **range** hits). All eight runs produced the same IDs. Peak child
RSS (`wait4`, includes launch footprint) was 17,448,960 B and 17,428,480 B in
the respective series. These sub-millisecond differences should not be used to
size a real deployment. None of these counters measures physical NVMe reads.

Reproduce, using the CPU binary and the mixed fixture:

```sh
for mib in 0 1; do
    python3 tools/gigachat4/benchmark.py --runner "$RUNNER" --model "$MIXED" \
        --model-id mixed-seed-70841 --tokens 1,5,8,13,21,34,8 --generate 32 \
        --host-cache-mib "$mib" --repeat 4 --page-cache --out "bench-$mib.json"
done
```

Measured binary SHA-256: `79907b1d57985bcff1b35298b4bbbd1f43570d12a066909a633328faf9586a8e`.
The 32 generated IDs (one byte per ID in this fixture) hash to
`7ba6b8ce4b568c4ffe99dfe634440fafc90c2c230d22543dff82b9d36a6e5703`.

### Full-model qualification

| Metric | How to obtain it | Status |
|---|---|---|
| TTFT, prefill and generation rate | Native phase timers; use exact prompts and separate first process from warm decode | Tiny CPU and V100 only |
| Peak host RAM | Per-child high-water mark; report model-volume file cache separately | Tiny CPU run only |
| GPU utilization / peak VRAM | Sample specified physical GPU and compute PID; sampled peak is a lower bound | Not measured |
| PCIe | Native H→D/D→H bytes and event time; report link negotiation and any independent bus counters separately | Tiny V100 weight transfers measured; no full-size bus measurement |
| NVMe | Selected device sector delta and wall interval with competing activity recorded | Not measured |
| Routed expert cache hit ratio | `expert_cache_hits / (expert_cache_hits + expert_cache_misses)` on device tiles, not all cached tensors | Not measured on full-size routes |
| CUDA kernel vs transfer time | CUDA event counters plus host read seconds; do not equate summed event time with wall time when overlapping | Not measured |

Use the [benchmark protocol](BENCH_PROTOCOL.md). An all-cold, disk-streamed
token is likely I/O-limited: at an **assumed** 3 GB/s, the **routed** FP8 bytes
alone take at least 7.166 s. If the same weights hit in RAM but miss in VRAM,
the analogous bound at an **assumed** 12 GB/s PCIe is 1.791 s. Neither
identifies the actual bottleneck on the target: measure cache hits, fixed
weight traffic, CPU attention and GPU kernel time before drawing that conclusion.
