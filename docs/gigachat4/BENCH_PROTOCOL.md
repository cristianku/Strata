# GigaChat4 local process benchmark

Use [`tools/gigachat4/benchmark.py`](../../tools/gigachat4/benchmark.py) to measure one
native runner, one local model directory and one fixed list of input token IDs.
Each run starts a new process, generates greedily with `--ignore-eos`, records the
output IDs, then exits. No server, downloads, root privileges or cache flushes.
This protocol is `gigachat4-linux-process-v1`, not the standard Qwen benchmark.

## Run it

Run from the checkout on Linux with Python 3.9 or later. The runner and model must already
exist. `RUNNER` and `MODEL` below are paths to **your** binary and immutable
model or parity fixture. For example:

```sh
python3 tools/gigachat4/benchmark.py \
  --runner "$RUNNER" --model "$MODEL" \
  --tokens 1,5,8,13,21,34,8 --generate 32 --repeat 4 \
  --host-cache-mib 0 --model-id "$MODEL_ID" --build-id "$BUILD_ID" \
  --output cpu.json
```

`--repeat` counts **all** launches, including the first. For CUDA, use a CUDA
build and add `--cuda --gpu-cache-mib N` with the desired budget in MiB.
`--host-cache-mib N` applies on either backend. Omitting a budget leaves it to
the executable. Cache budgets are not limits on total RAM or VRAM.

`--prefetch` is off by default. On either backend, it passes the native
`--prefetch` flag: the runner advises the Linux kernel (`POSIX_FADV_WILLNEED`)
about file ranges for the next routed expert. The kernel decides whether to
read those pages ahead. This does not populate the runner's host weight cache
or copy weights to CUDA. Under `metrics.weights`, `prefetch_enabled` and
`prefetch_supported` distinguish a request from backend support;
`prefetch_requests`, `prefetch_bytes_advised` and `prefetch_errors` describe
the advice, not bytes actually read from storage. Keep both cache budgets fixed
when comparing prefetch off and on.

Optional observations (add to the same command):

| Option | What to supply |
| --- | --- |
| `--disk-device nvme0n1` | One kernel device or partition name from `/sys/class/block`. Select the device actually serving the shards. If the backing device is hidden or ambiguous, omit it. |
| `--page-cache` | Use the already installed `fincore` to observe residency of top-level `*.safetensors` before and after each process. |
| `--gpu-id GPU-…` | Use the already installed `nvidia-smi` for **one** GPU UUID or index. This only selects the monitor; it does not set the runner's CUDA device. Match the physical card, especially with `CUDA_VISIBLE_DEVICES`. |
| `--sample-ms 500` | Target interval for live process-I/O samples and GPU queries. GPU queries are sequential and may take longer. Keep the setting and probes the same in both arms of a comparison. |

Without `--output`, stdout is the JSON report. Runner diagnostics go to stderr.
Each entry in `runs` has the generated IDs and backend `metrics`, plus the
observations below. A runner failure, invalid JSON, or a token/step count that
does not match the command stops the series, retains the failed attempt and
returns a nonzero exit code. No failed run is silently retried.

Run the offline wrapper tests without a model or GPU:
`python3 -m unittest discover -s tools/gigachat4 -p 'test_benchmark.py' -v`.

## Keep the comparison reproducible

1. Fix the binary, model, token IDs and output length. Supply the source
   revision/build options and the model revision or fixture ID. The report
   hashes the executable and wrapper **before** timing. It does not hash or
   pre-read weights. If you verify model digests, do it separately and record
   when: reading every shard changes the starting cache state.
2. Record the CPU/GPU model, RAM/VRAM, storage stack, filesystem, effective
   container limits, driver and competing workloads. The report includes the
   command, Linux kernel, CPU affinity and selected environment variables.
   To compare CPU and CUDA, or two cache budgets, change only the intended
   setting. Alternate configuration order across series if startup matters.
3. Keep the **first process** separate from **repeat processes**. Every process
   starts with empty application weight and sequence caches. A repeat may
   benefit from the previous process's Linux file cache, but it does not reuse
   its RAM or GPU expert cache. The first process can also start with resident
   weights. No run is automatically marked disk-cold, fully warm or steady-state.
   Keep residency snapshots and device reads with any cold/warm claim. To
   measure warm **in-process** decode, use the backend's decode phase; a new
   launch is not another request against a resident server.
4. Report individual runs; for a repeat median, report how many repeats it
   covers and exclude the first run. Check `generated_identical_across_successful_runs`
   and compare IDs between configurations. Identical IDs are a repeatability
   check, not a parity proof. Label synthetic fixtures as such; their rates
   are not full-checkpoint performance.

## Interpret the counters

`null` means unobserved, not applicable or no defined rate. Zero is a value
reported by a source. Disabled probes and failed reads have a reason or
diagnostic; an empty GPU sample list means no completed in-window query.
Backend-specific fields that the executable did not emit are not inferred.

| JSON field (under `runs[]`) | Measurement domain |
| --- | --- |
| `metrics.load_seconds` | Runner steady clock: start of model setup through decoder construction. Does not imply every weight is resident. |
| `metrics.prefill_seconds`, `prefill_tokens_per_second` | All prompt steps through selection of the first output, if requested. Rate = `prompt_tokens / prefill_seconds`. |
| `metrics.ttft_seconds` | Runner start of model setup to first greedy ID ready, **including load**. Not client-visible latency: stdout contains one result after decode. `--generate 0` is prefill only, with no TTFT. |
| `metrics.decode_seconds`, `decode_tokens_per_second` | Backend time **after** the first ID. Rate = `decode_steps / decode_seconds`; for `N` generated IDs there are `max(N − 1, 0)` decode steps. No rate for zero steps or a zero timer. |
| `process_wall_seconds` | Wrapper monotonic clock from just before spawn to observed process exit. Includes startup, output and teardown. File-cache probes are outside this interval. |
| `child_peak_rss_bytes` | Per-PID Linux `wait4().ru_maxrss` high-water mark, converted from KiB to bytes. This covers the process lifetime, including any inherited pre-`exec` footprint, which can matter on tiny fixtures. It is not a sampled RSS, sum of peaks, model-only allocation, or VRAM. `child_input_blocks`, `child_major_faults`, `child_minor_faults` are final kernel counters from the same call. |
| `metrics.weights` | Backend logical payload operations and bytes (`payload_read_ops`, `logical_payload_bytes_read`), host-cache hits/misses/evictions, and sizes. Payload reads exclude index and header reads and can be satisfied by the OS cache. |
| `child_io` | `/proc/PID/io`: `rchar` is bytes returned by read-like calls from any file descriptor; `syscr` counts such calls; `read_bytes` is storage I/O the kernel accounts to the task. The last is neither model-only nor a block-device measurement. `coverage: final` is read after exit, before reaping. If that is denied, `sampled_lower_bound` gives the last readable live snapshot and its timestamp; even a zero there can miss later I/O. |
| `device_io` | Difference of the specified device's `/sys/class/block/NAME/stat` snapshots bracketing the child. `read_bytes = read_sectors_512 × 512`, regardless of drive sector size. `read_time_ms` is the **sum** of read-request time; concurrent requests overlap. `read_bytes_per_second` uses `window_seconds` (wall time), not that sum. All users of this device contribute. A mounted volume, partition, RAID, VM device and physical disk are different domains; do not add parent and child device counts. |
| `shard_page_cache_before`, `shard_page_cache_after` | `fincore` totals resident file pages (in bytes) and file length across the listed shard paths. Page counts can include slack in a last page. These are point-in-time residency observations, not bytes served by the cache or a hit ratio. They exclude other files, device caches and application caches. Do not subtract device reads from logical reads to infer cache hits. |
| `metrics.cuda` | Backend counters, independent of GPU polling. `h2d_bytes` counts all instrumented host-to-device copies; `weight_h2d_bytes` (weights and scales) is a **subset**. `d2h_bytes` is the reverse direction. `h2d_ms`, `d2h_ms`, `kernel_ms` are cumulative CUDA-event times on the instrumented stream, not CPU timers. `h2d_bytes_per_event_second` divides H2D bytes by H2D event time only. It excludes CPU staging copies and is not measured PCIe bus load, disk throughput, or end-to-end bandwidth. Cache allocations/hits here are not total process VRAM. |
| `gpu` | Timestamped `nvidia-smi` samples. `device_util_pct` and `device_used_mib` are **device-wide**. The utilization mean is the arithmetic mean of available samples of the driver's rolling utilization reading. `process_used_mib` matches both device UUID and the runner PID in the compute-process list. An invisible PID (including a namespace mismatch), unsupported field or missing sample yields `null`, not a device-wide fallback. Sample maxima are lower bounds on peaks; short runs may have no samples. |

GPU and disk activity from other processes, filesystem readahead, virtualized
storage and the observer itself can affect these measurements. Keep those limits
with the numbers. A run without device counters can establish logical read volume;
it cannot establish how many bytes came from an SSD.
