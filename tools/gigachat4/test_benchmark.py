"""Offline regression tests for the Linux process benchmark. No engine, GPU or model required."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark


# Runs as a real child of the wrapper. Report the actual argv and peak RSS so
# that checking the report doesn't just check the wrapper's copy of the command.
RUNNER = """
import argparse, json, os, resource, sys
from pathlib import Path

p = argparse.ArgumentParser()
for name in ('model', 'tokens', 'generate'):
    p.add_argument('--' + name, required=True)
for name in ('host-cache-mib', 'gpu-cache-mib'):
    p.add_argument('--' + name, type=int)
for name in ('ignore-eos', 'cuda', 'prefetch', 'overlap-copies'):
    p.add_argument('--' + name, action='store_true')
a = p.parse_args()
assert a.ignore_eos
model = Path(a.model)
n = int(a.generate)
mode = os.environ['BENCH_FIXTURE_MODE']
if mode == 'exit':
    sys.exit(7)
if mode == 'invalid_json':
    print('no JSON result')
    sys.exit(0)
if mode == 'rss' and not (model / 'seen').exists():
    (model / 'seen').touch()
    pages = bytearray(32 << 20)
    pages[::4096] = b'x' * (len(pages) // 4096)
m = {'load_seconds': 1, 'prefill_seconds': 2, 'ttft_seconds': 3 if n else None,
     'decode_seconds': 4 if n > 1 else 0, 'decode_steps': max(0, n - 1),
     'prompt_tokens': len(a.tokens.split(',')),
     'weights': {'prefetch_enabled': a.prefetch, 'prefetch_bytes_advised': 4096 if a.prefetch else 0,
                 'logical_payload_bytes_read': len((model / 'tiny.safetensors').read_bytes())}}
if a.cuda:
    m['cuda'] = {'h2d_bytes': 1024, 'h2d_ms': 0.25, 'kernel_ms': 0}
if mode == 'bad_steps':
    m['decode_steps'] += 1
if mode == 'bad_timer':
    m['decode_seconds'] = float('nan')
print(json.dumps({'generated': [16] * (0 if mode == 'short' else n), 'metrics': m,
                  'argv': sys.argv[1:], 'pid': os.getpid(),
                  'rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux wait4 and /proc")
class BenchmarkCLI(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model = Path(tmp.name, "fixture with spaces")
        self.model.mkdir()
        (self.model / "tiny.safetensors").write_bytes(b"fixture" * 64)
        self.runner = Path(tmp.name, "runner")
        self.runner.write_text(f"#!{sys.executable}\n{RUNNER}", encoding="utf-8")
        self.runner.chmod(0o755)

    def bench(self, *options, n=3, mode="ok", status=0):
        result = subprocess.run(
            [sys.executable, benchmark.__file__, "--runner", self.runner, "--model", self.model,
             "--tokens", "1,02,3", "--generate", str(n), *options],
            env={**os.environ, "BENCH_FIXTURE_MODE": mode}, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, status, result.stderr)
        return json.loads(result.stdout)

    def test_repeats_and_per_child_peak(self):
        report = self.bench("--repeat", "2", mode="rss")
        a, b = report["runs"]
        self.assertEqual([a["order"], b["order"]], ["first", "repeat"])
        self.assertTrue(report["generated_identical_across_successful_runs"])
        self.assertNotEqual(a["pid"], b["pid"])
        for run in (a, b):
            extra = run["backend_extra"]
            self.assertEqual(extra["argv"], ["--model", str(self.model), "--tokens", "1,2,3",
                                             "--generate", "3", "--ignore-eos"])
            self.assertEqual(run["pid"], extra["pid"])
            self.assertGreaterEqual(run["child_peak_rss_bytes"], extra["rss_kib"] * 1024)
            self.assertEqual(run["generated"], [16, 16, 16])
            self.assertEqual(run["prefill_tokens_per_second"], 1.5)
            self.assertEqual(run["decode_tokens_per_second"], 0.5)  # two post-prefill steps, not three
            self.assertFalse(run["metrics"]["weights"]["prefetch_enabled"])
            self.assertIsNone(run["metrics"]["cuda"])
            self.assertIsNone(run["device_io"]["read_bytes"])
        # A cumulative RUSAGE_CHILDREN peak would incorrectly repeat the larger value.
        self.assertGreater(a["child_peak_rss_bytes"], b["child_peak_rss_bytes"] + (8 << 20))

    def test_prefetch_is_independent_of_cuda_and_host_cache(self):
        for options in ([], ["--cuda", "--host-cache-mib", "0", "--gpu-cache-mib", "8", "--overlap-copies"]):
            with self.subTest(options=options):
                report = self.bench("--repeat", "1", "--prefetch", *options, n=1)
                run = report["runs"][0]
                argv = run["backend_extra"]["argv"]
                self.assertEqual(report["command"][1:], argv)
                self.assertEqual(argv.count("--prefetch"), 1)
                self.assertEqual(run["metrics"]["weights"]["prefetch_bytes_advised"], 4096)
                self.assertEqual(run["metrics"]["weights"]["logical_payload_bytes_read"], 448)
                self.assertIsNone(run["decode_tokens_per_second"])
                if options:
                    self.assertIn("--overlap-copies", argv)
                    self.assertEqual(argv[argv.index("--host-cache-mib") + 1], "0")
                    self.assertEqual(argv[argv.index("--gpu-cache-mib") + 1], "8")
                    self.assertEqual(run["metrics"]["cuda"]["kernel_ms"], 0)
                    self.assertEqual(run["h2d_bytes_per_event_second"], 4096000)
                else:
                    self.assertTrue(set(argv).isdisjoint({"--cuda", "--gpu-cache-mib", "--host-cache-mib"}))
                    self.assertIsNone(run["h2d_bytes_per_event_second"])

    def test_prefill_only(self):
        run, = self.bench("--repeat", "1", n=0)["runs"]
        self.assertEqual(run["generated"], [])
        self.assertIsNone(run["metrics"]["ttft_seconds"])
        self.assertIsNone(run["decode_tokens_per_second"])

    def test_failures_stop_repeats(self):
        for mode in ("exit", "short", "bad_steps", "bad_timer", "invalid_json"):
            with self.subTest(mode=mode):
                run, = self.bench("--repeat", "3", mode=mode, status=1)["runs"]
                self.assertIn("error", run)
                self.assertEqual(run["exit_code"], 7 if mode == "exit" else 0)
                self.assertIsNone(run["metrics"])


class Observations(unittest.TestCase):
    def test_disk_time_and_wall_time_have_distinct_domains(self):
        delta = benchmark.disk_delta("fixture", (10, 900, 300), (12, 916, 4300))
        self.assertEqual(delta["read_bytes"], 8192)
        self.assertEqual(delta["read_time_ms"], 4000)
        self.assertEqual(delta["read_bytes_per_second"], 4096)  # wall time = 2 seconds
        self.assertEqual(benchmark.disk_delta("fixture", (10, 0, 0), (11, 0, 0))["read_bytes"], 0)
        self.assertIsNone(benchmark.disk_delta("fixture", None, None)["read_bytes"])

    def test_gpu_does_not_attribute_other_processes_or_cards(self):
        for reading, expected in (("28", 28), ("0", 0), ("[N/A]", None), (None, None)):
            with self.subTest(reading=reading):
                sampler = benchmark.GpuSampler("GPU-A", 1, 42, time.monotonic())

                def query(device, fields):
                    self.assertEqual(device, "GPU-A")
                    if fields.startswith("--query-gpu="):
                        return [["GPU-A", "fixture", "driver", "0", "300"]]
                    rows = [["GPU-A", "987", "500"], ["GPU-B", "42", "700"]]
                    if reading is not None:
                        rows.append(["GPU-A", "42", reading])
                    sampler.stop.set()  # one full sample; no sleeps or scheduler races
                    return rows

                with patch.object(benchmark, "query_smi", side_effect=query):
                    sampler.collect()
                result = sampler.finish(time.monotonic())
                self.assertEqual(result["mean_sampled_device_util_pct"], 0)
                self.assertEqual(result["max_sampled_device_used_mib"], 300)
                self.assertEqual(result["max_sampled_process_used_mib"], expected)


if __name__ == "__main__":
    unittest.main()
