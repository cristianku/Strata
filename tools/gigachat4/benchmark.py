#!/usr/bin/env python3
"""Measure a local GigaChat4 runner on Linux. See docs/gigachat4/BENCH_PROTOCOL.md."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import subprocess
import sys
import threading
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sec_rate(count, seconds):
    return count / seconds if seconds > 0 and count > 0 else None


def parse_result(text, prompt_count, generate):
    def reject_constant(value):
        raise ValueError(f"invalid JSON number: {value}")

    def finite_float(text):
        value = float(text)
        return value if math.isfinite(value) else reject_constant(text)

    data = json.loads(text, parse_float=finite_float, parse_constant=reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        raise ValueError("runner must emit one JSON object containing metrics")
    ids = data.get("generated")
    if (not isinstance(ids, list) or len(ids) != generate
            or any(type(i) is not int or i < 0 for i in ids)):
        raise ValueError(f"expected {generate} generated token IDs with --ignore-eos")
    m = {"cuda": None, "weights": None, **data["metrics"]}
    for key in ("load_seconds", "ttft_seconds", "prefill_seconds", "decode_seconds"):
        if key == "ttft_seconds" and not generate:
            if key not in m or m[key] is not None:
                raise ValueError("expected null ttft_seconds for prefill-only run")
            continue
        val = m.get(key)
        if type(val) not in (float, int) or not math.isfinite(val) or val < 0:
            raise ValueError(f"missing/invalid backend timing: {key}")
    for key, expected in (("prompt_tokens", prompt_count), ("decode_steps", max(0, generate - 1))):
        if type(m.get(key)) is not int or m[key] != expected:
            raise ValueError(f"expected backend {key}={expected}, got {m.get(key)!r}")
    cuda = m["cuda"] if m["cuda"] is not None else {}
    if not isinstance(cuda, dict):
        raise ValueError("expected a CUDA metrics object or null")
    n, ms = cuda.get("h2d_bytes"), cuda.get("h2d_ms")
    return {
        "generated": ids, "metrics": m,
        "backend_extra": {k: v for k, v in data.items() if k not in ("generated", "metrics")},
        "prefill_tokens_per_second": sec_rate(m["prompt_tokens"], m["prefill_seconds"]),
        "decode_tokens_per_second": sec_rate(m["decode_steps"], m["decode_seconds"]),
        "h2d_bytes_per_event_second": (
            n / (ms * 0.001) if type(n) is int and n >= 0 and type(ms) in (int, float) and ms > 0 else None
        ),
    }


def proc_io(pid):
    try:
        counters = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
        return {"rchar": int(counters["rchar"]), "read_bytes": int(counters["read_bytes"]),
                "syscr": int(counters["syscr"])}
    except (OSError, KeyError, ValueError) as exc:
        return {"rchar": None, "read_bytes": None, "syscr": None, "reason": str(exc)}


def capture_stdout(child, start, interval_ms):
    # Some kernels deny access to zombie /proc/PID/io in an unprivileged
    # container. Keep a timestamped live lower bound as a fallback.
    chunks, live = [], None
    poller = select.poll()
    poller.register(child.stdout, select.POLLIN)
    while True:
        snapshot = proc_io(child.pid)
        if snapshot["rchar"] is not None:
            live = {**snapshot, "observed_seconds": time.monotonic() - start}
        if poller.poll(interval_ms):
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                return b"".join(chunks), live
            chunks.append(chunk)


def disk_snapshot(name):
    if not name:
        return None
    try:
        stats = Path("/sys/class/block", name, "stat").read_text().split()
        return time.monotonic(), int(stats[2]), int(stats[3])
    except (OSError, ValueError, IndexError):
        return None


def disk_delta(name, before, after):
    out = {"device": name, "read_sectors_512": None, "read_bytes": None,
           "read_time_ms": None, "window_seconds": None, "read_bytes_per_second": None}
    if before is None or after is None:
        out["reason"] = "no device selected" if not name else "device stat unavailable"
        return out
    seconds, sectors, read_ms = (b - a for a, b in zip(before, after))
    if sectors < 0 or read_ms < 0 or seconds <= 0:
        out["reason"] = "counter reset or invalid interval"
        return out
    return {**out, "read_sectors_512": sectors, "read_bytes": sectors * 512,
            "read_time_ms": read_ms, "window_seconds": seconds,
            "read_bytes_per_second": sectors * 512 / seconds}


def file_cache(files):
    missing = {"resident_bytes": None, "file_bytes": None}
    if files is None:
        return {**missing, "reason": "page-cache probe disabled"}
    if not files:
        return {**missing, "reason": "no top-level .safetensors files"}
    try:
        # mincore-based versions can return all-resident for files the caller
        # neither owns nor may write. Do not promote such an answer to evidence.
        if any(p.stat().st_uid != os.geteuid() and not os.access(p, os.W_OK) for p in files):
            return {**missing, "reason": "cannot verify residency for unowned, non-writable files"}
        result = subprocess.run(["fincore", "-brno", "RES,SIZE", "--", *files],
                                capture_output=True, text=True, check=True, timeout=120)
        rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        if len(rows) != len(files) or any(len(r) != 2 or min(r) < 0 for r in rows):
            raise ValueError("invalid fincore output")
        return {"resident_bytes": sum(r[0] for r in rows), "file_bytes": sum(r[1] for r in rows)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {**missing, "reason": str(exc)}


def query_smi(device, query):
    result = subprocess.run(["nvidia-smi", "-i", device, query, "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=3)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def smi_number(text):
    try:
        value = int(text.strip())
        return value if value >= 0 else None
    except ValueError:  # "N/A", "[Not Supported]", etc.
        return None


class GpuSampler:
    """Independent device and PID queries; no device selection changes in the child."""

    def __init__(self, device, interval, pid, start):
        self.device = device
        self.interval = interval
        self.pid = str(pid)
        self.start = start
        self.stop = threading.Event()
        self.samples = []
        self.errors = set()
        self.thread = None

    def begin(self):
        if self.device is None:
            self.errors.add("GPU sampling disabled")
        elif not shutil.which("nvidia-smi"):
            self.errors.add("nvidia-smi unavailable")
        else:
            self.thread = threading.Thread(target=self.collect, daemon=True)
            self.thread.start()

    def collect(self):
        while not self.stop.is_set():
            t = time.monotonic()
            try:
                rows = query_smi(self.device, "--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used")
                if len(rows) != 1 or len(rows[0]) != 5:
                    raise ValueError("expected one selected GPU")
                uuid, name, driver, util, mem = rows[0]
                self.samples.append((time.monotonic(), {
                    "uuid": uuid, "name": name, "driver": driver,
                    "device_util_pct": smi_number(util), "device_used_mib": smi_number(mem),
                }))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.errors.add(str(exc))
                return
            if self.stop.is_set():
                return
            try:
                rows = query_smi(self.device, "--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
                values = [smi_number(mem) for gpu, pid, mem in rows if gpu == uuid and pid == self.pid]
                self.samples.append((time.monotonic(), {
                    "process_used_mib": values[0] if len(values) == 1 else None,
                }))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.errors.add(str(exc))
            self.stop.wait(max(0, self.interval - (time.monotonic() - t)))

    def finish(self, end):
        self.stop.set()
        if self.thread:
            self.thread.join()
        # Discard queries completed after the observed child exit, even if they
        # began while the process was alive (important on tiny fixtures).
        samples = [{"end_seconds": t - self.start, **row} for t, row in self.samples if t <= end]
        util, dev_mem, pid_mem = (
            [s[k] for s in samples if s.get(k) is not None]
            for k in ("device_util_pct", "device_used_mib", "process_used_mib")
        )
        return {
            "device_selector": self.device, "interval_seconds": self.interval,
            "mean_sampled_device_util_pct": sum(util) / len(util) if util else None,
            "max_sampled_device_util_pct": max(util, default=None),
            "max_sampled_device_used_mib": max(dev_mem, default=None),
            "max_sampled_process_used_mib": max(pid_mem, default=None),
            "samples": samples, "diagnostics": sorted(self.errors),
        }


def run_once(args, command, index, files):
    cache_before = file_cache(files)
    disk_before = disk_snapshot(args.disk_device)
    start = time.monotonic()
    # Only stdout is captured; the runner's diagnostics stay on stderr. There
    # is no shared RUSAGE_CHILDREN high-water mark and no pipe for stderr to fill.
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE) as child:
        sampler = GpuSampler(args.gpu_id, args.sample_ms / 1000, child.pid, start)
        try:
            sampler.begin()
            text, live_io = capture_stdout(child, start, args.sample_ms)
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            end = time.monotonic()
            sampler.stop.set()
            disk_after = disk_snapshot(args.disk_device)
            io = proc_io(child.pid)
            if io["rchar"] is not None:
                io["coverage"] = "final"
            elif live_io is not None:
                io = {**live_io, "coverage": "sampled_lower_bound", "final_unavailable": io["reason"]}
            else:
                io["coverage"] = "unavailable"
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        except BaseException:
            sampler.stop.set()
            child.kill()
            raise
    gpu = sampler.finish(end)
    run = {
        "run": index, "order": "first" if index == 1 else "repeat",
        "process_cache_at_start": "fresh process", "pid": child.pid, "exit_code": child.returncode,
        "process_wall_seconds": end - start, "child_peak_rss_bytes": usage.ru_maxrss * 1024,
        "child_io": io, "child_input_blocks": usage.ru_inblock,
        "child_major_faults": usage.ru_majflt, "child_minor_faults": usage.ru_minflt,
        "device_io": disk_delta(args.disk_device, disk_before, disk_after),
        "shard_page_cache_before": cache_before, "shard_page_cache_after": file_cache(files),
        "gpu": gpu, "generated": None, "metrics": None,
    }
    if child.returncode:
        run["error"] = f"runner exited with {child.returncode}; see stderr"
    else:
        try:
            run.update(parse_result(text, len(args.tokens.split(",")), args.generate))
        except (ValueError, OverflowError) as exc:
            run["error"] = f"invalid runner result: {exc}"
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runner", "--exe", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokens", required=True, help="comma-separated nonnegative decimal token IDs")
    p.add_argument("--generate", required=True, type=int, help="number of tokens (0 = prefill only)")
    p.add_argument("--repeat", "--runs", default=2, type=int, help="total process launches, including first")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--gpu-cache-mib", type=int)
    p.add_argument("--host-cache-mib", type=int)
    p.add_argument("--prefetch", action="store_true", help="ask the runner to advise kernel file readahead")
    p.add_argument("--overlap-copies", action="store_true", help="use two CUDA streams for eligible weight tiles")
    p.add_argument("--gpu-id", help="optional nvidia-smi selector for monitoring only (one index or UUID)")
    p.add_argument("--sample-ms", type=int, default=500, help="target GPU and live proc I/O sample interval")
    p.add_argument("--disk-device", help="optional kernel device or partition name in /sys/class/block")
    p.add_argument("--page-cache", action="store_true", help="probe .safetensors residency using fincore")
    p.add_argument("--model-id", help="immutable model revision or synthetic fixture ID supplied by operator")
    p.add_argument("--build-id", help="source revision/build description supplied by operator")
    p.add_argument("--output", "--out", type=Path, help="write report to this file instead of stdout")
    args = p.parse_args()
    if not sys.platform.startswith("linux"):
        p.error("Linux is required")
    args.runner = args.runner.resolve()
    args.model = args.model.resolve()
    if not args.runner.is_file() or not os.access(args.runner, os.X_OK) or not args.model.is_dir():
        p.error("--runner must be executable; --model must be a local directory")
    tokens = [t.strip() for t in args.tokens.split(",")]
    if not all(t.isascii() and t.isdecimal() for t in tokens):
        p.error("--tokens requires a nonempty CSV of nonnegative decimal IDs")
    args.tokens = ",".join(str(int(t)) for t in tokens)
    if args.generate < 0 or min(args.repeat, args.sample_ms) <= 0:
        p.error("--generate must be nonnegative; --repeat and --sample-ms must be positive")
    if args.gpu_cache_mib and not args.cuda:
        p.error("--gpu-cache-mib requires --cuda")
    if args.overlap_copies and not args.cuda:
        p.error("--overlap-copies requires --cuda")
    if args.disk_device and ("/" in args.disk_device or args.disk_device in (".", "..")):
        p.error("--disk-device takes a kernel device name, e.g. nvme0n1")
    command = [str(args.runner), "--model", str(args.model), "--tokens", args.tokens,
               "--generate", str(args.generate), "--ignore-eos"]
    if args.cuda:
        command.append("--cuda")
    if args.overlap_copies:
        command.append("--overlap-copies")
    if args.prefetch:
        command.append("--prefetch")
    for flag, n in (("--gpu-cache-mib", args.gpu_cache_mib), ("--host-cache-mib", args.host_cache_mib)):
        if n is not None:
            if n < 0:
                p.error(f"{flag} must be nonnegative")
            command.extend([flag, str(n)])
    files = sorted(args.model.glob("*.safetensors")) if args.page_cache else None
    report = {
        "protocol": "gigachat4-linux-process-v1", "started_utc": utc_now(),
        "command": command, "runner_sha256": digest(args.runner), "wrapper_sha256": digest(Path(__file__)),
        "model_id": args.model_id, "build_id": args.build_id, "requested_runs": args.repeat,
        "kernel": platform.release(), "arch": platform.machine(), "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "page_cache_probe_files": None if files is None else [p.name for p in files],
        "measurement_domains": {
            "metrics_and_backend_extra": "runner JSON; backend timers, logical reads, CUDA counters if emitted",
            "h2d_bytes_per_event_second": "backend H2D bytes / CUDA H2D event time; not PCIe bus utilization",
            "process_wall_seconds": "monotonic clock around subprocess launch through observed exit",
            "child_peak_rss_bytes": "Linux wait4(pid).ru_maxrss (KiB converted to bytes)",
            "child_io": "/proc/PID/io; final if readable, else live lower bound; rchar/syscr logical",
            "child_input_blocks_and_faults": "wait4(pid) final kernel counters; not a physical device trace",
            "device_io": "selected sysfs block device, all users; sectors=512 B; read_time_ms is cumulative",
            "shard_page_cache": "optional fincore before/after; resident file pages, not bytes served from cache",
            "gpu": "optional nvidia-smi samples: device-wide utilization/VRAM; compute PID VRAM separately",
        },
        "runs": [],
    }
    for i in range(1, args.repeat + 1):
        try:
            run = run_once(args, command, i, files)
        except OSError as exc:
            run = {"run": i, "error": str(exc), "generated": None, "metrics": None}
        report["runs"].append(run)
        if "error" in run:
            print(run["error"], file=sys.stderr)
            break
    successful = [r["generated"] for r in report["runs"] if "error" not in r]
    report["generated_identical_across_successful_runs"] = (
        all(s == successful[0] for s in successful[1:]) if len(successful) >= 2 else None
    )
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0 if len(successful) == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
