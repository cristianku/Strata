import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import memory_budget as m


def sequence(routes):
    for t, ids in enumerate(routes):
        for layer in range(3, 64):
            yield {"token": t, "layer": layer, "experts": list(ids)}


class HeaderBoundary(unittest.TestCase):
    shard = "model-00045-of-00045.safetensors"

    def test_full_or_wrong_range_response_never_read(self):
        for status, cr in [(200, ""), (206, "bytes 0-65535/999999"), (206, "bytes 8-15/999999")]:
            with self.subTest(status=status, cr=cr), patch.object(m, "build_opener") as op:
                r = MagicMock(status=status, headers={"Content-Range": cr})
                op.return_value.open.return_value = r
                with self.assertRaises(ValueError):
                    m.header_range(self.shard, 0, 7)
                r.read.assert_not_called()

    def test_absurd_prefix_aborts_before_second_request(self):
        with patch.object(m, "header_range", return_value=(struct.pack("<Q", 100 << 20), 1 << 34)) as get:
            with self.assertRaises(ValueError):
                m.fetch_header(self.shard)
            get.assert_called_once_with(self.shard, 0, 7)


class Budget(unittest.TestCase):
    def test_reconciles_official_index_and_router_buffer(self):
        b = m.budget("official", 0, 0, 1, 1, False, 0, 0)
        self.assertEqual(b["full_source_bytes"], 719702848992)
        self.assertEqual(b["groups"]["mtp"]["source_bytes"], 4912973280)
        self.assertEqual(sum(v["tensors"] for v in b["groups"].values()), 95616)
        self.assertEqual(b["base_repacked_bytes"], b["base_source_bytes"])
        self.assertEqual(b["base_params_excluding_router_buffers"], 707721974784)
        self.assertEqual(b["native_host_logical_state_bytes"], (204 << 20) + 1060864)

    def test_quantized_peak_and_state_scaling(self):
        b = m.budget("Q4_K", 16*m.GIB, 64*m.GIB, 8192, 1, True, 2*m.GIB, 8*m.GIB, "matrices")
        p = b["gpu_fixed"]
        self.assertTrue(p["within_assumed_budgets"])
        self.assertEqual((p["gpu_cache_slots"], p["host_cache_slots"]), (106, 2411))
        self.assertLessEqual(p["peak_budgeted_gpu_bytes"], 16*m.GIB)
        self.assertEqual(p["min_routed_pcie_bytes_per_token"], 9463136256)
        self.assertEqual(sum(m.state_bytes(1048576, 2, False).values()),
                         (644*m.GIB + 198*(1 << 20)) * 2)


class DemandReplay(unittest.TestCase):
    def test_native_route_file_and_warmed_prompt(self):
        # Exercise the documented default CLI path with the runner's one-line-per-layer schema.
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d, "routes.jsonl")
            trace.write_text("".join(json.dumps(r) + "\n" for r in sequence([range(8)] * 8)))
            out = json.loads(subprocess.check_output([
                sys.executable, m.__file__, "replay", str(trace), "--warmup-tokens", "7",
                "--format", "official", "--gpu-cache-gib", "24", "--host-cache-gib", "0", "--predict", "8"
            ]))
        self.assertEqual(out["tokens"], 1)
        self.assertEqual(out["demand_nvme_bytes"], 0)
        self.assertEqual(out["counts"]["gpu_hits"], 488)
        self.assertEqual(out["counts"]["predicted"], 488)

    def test_cold_hot_and_layer_namespaces(self):
        blob = m.expert_bytes("Q4_K")
        # Identical integer IDs at 61 distinct layers must cause 488 compulsory reads.
        for gs, hs, second in [(488, 0, "gpu_hits"), (0, 488, "host_hits"), (0, 0, "disk_misses")]:
            with self.subTest(tier=second):
                out = m.replay(sequence([range(8)] * 2), "Q4_K", gs*blob, hs*blob, 8)
                counts = out["counts"]
                self.assertEqual(counts[second], 976 if second == "disk_misses" else 488)
                self.assertEqual(counts["disk_misses"], 976 if second == "disk_misses" else 488)
                self.assertEqual(out["demand_nvme_bytes"], counts["disk_misses"] * blob)
                self.assertEqual(counts["predicted"], 488)  # no first-token "predictions"
                self.assertEqual(out["prediction_precision"], 1.0)

    def test_past_only_with_changing_routes(self):
        out = m.replay(sequence([range(8), range(8, 16), range(8, 16)]), "BF16", 0, 0, 4)
        self.assertEqual(out["prediction_precision"], 0.5)
        self.assertEqual(out["unneeded_prefetch_bytes"], 4 * 61 * m.expert_bytes("BF16"))
        self.assertEqual(out["counts"]["disk_misses"], 3 * 488)  # scoring did not prefetch

    def test_rejects_out_of_order_or_partial_routes(self):
        valid = list(sequence([range(8)]))
        for records in (valid[:-1], valid[::-1], [dict(valid[0], experts=[1]*8)], []):
            with self.assertRaises(ValueError):
                m.replay(records, "Q3_K", 0, 0)


if __name__ == "__main__":
    unittest.main()
