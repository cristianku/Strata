"""Offline regression tests for the 708B layout and bounded HTTP reads."""
import io
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_layout as ck


class Layout(unittest.TestCase):
    def test_full_fp8_pairing_and_partial_block(self):
        schema = dict(ck.tensor_schema())
        self.assertEqual(len(schema), 95616)
        self.assertEqual(sum(k.startswith(ck.MTP) for k in schema), 49)
        self.assertEqual(schema[ck.FP + "_scale_inv"], ("F32", (5, 56)))  # 576 output rows
        self.assertEqual(schema["model.layers.63.mlp.experts.255.down_proj.weight"], ("F8_E4M3", (7168, 2048)))
        for name, (dt, shape) in schema.items():
            if dt == "F8_E4M3":
                self.assertEqual(schema[name + "_scale_inv"], ("F32", tuple((n + 127) // 128 for n in shape)))
            elif name.endswith("_scale_inv"):
                self.assertEqual(schema[name.removesuffix("_scale_inv")][0], "F8_E4M3")
        self.assertNotIn("model.layers.64.mlp.experts.0.up_proj.weight", schema)

    def test_index_wrong_name_same_count(self):
        with self.assertRaisesRegex(ValueError, "missing .*original.*extra .*clone"):
            ck.check_names({"weight_map": {"clone": ck.SHARDS[0]}}, {"original": ("BF16", (1,))})

    def test_duplicate_json_keys_and_wrong_config(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            ck.parse(b'{"a":{"dtype":"F32","dtype":"BF16"}}')
        with self.assertRaisesRegex(ValueError, "not the original JSON"):
            ck.pinned_json("config.json", b'{}')

    def test_header_wrong_layout(self):
        from copy import deepcopy
        shard = ck.SHARDS[0]
        name, scale = ck.FP, ck.FP + "_scale_inv"
        schema = {name: ("F8_E4M3", (129, 3)), scale: ("F32", (2, 1))}
        header = {
            name: {"dtype": "F8_E4M3", "shape": [129, 3], "data_offsets": [0, 387]},
            scale: {"dtype": "F32", "shape": [2, 1], "data_offsets": [387, 395]},
        }
        for obj in (header, [ck.START[shard], ck.START[shard] + 395, header]):
            self.assertEqual(ck.check_header(shard, obj, schema.keys(), schema)[1], ck.START[shard] + 395)
        for key, field, bad, reason in [
            (scale, "shape", [1, 2], "shape"),  # same bytes, wrong scale axes
            (name, "shape", [3, 129], "shape"),
            (scale, "shape", [2, True], "shape"),
            (scale, "dtype", "BF16", "dtype"),
            (name, "data_offsets", [False, 387], "data_offsets"),
            (scale, "data_offsets", [386, 394], "overlap"),
            (scale, "data_offsets", [388, 396], "gap"),
        ]:
            with self.subTest(key=key, field=field, bad=bad), self.assertRaisesRegex(ValueError, reason):
                altered = deepcopy(header)
                altered[key][field] = bad
                ck.check_header(shard, altered, schema.keys(), schema)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            ck.check_header(shard, header, {name}, schema)
        with self.assertRaisesRegex(ValueError, "missing"):
            ck.check_header(shard, {name: header[name]}, schema.keys(), schema)
        with self.assertRaisesRegex(ValueError, "spans do not fill"):
            ck.check_header(shard, [ck.START[shard], ck.START[shard] + 396, header], schema.keys(), schema)


class Response(io.BytesIO):
    def __init__(self, data, status, **headers):
        super().__init__(data)
        self.status, self.headers, self.reads = status, headers, []

    def read(self, n=-1):
        assert n >= 0, "unbounded response read"
        self.reads.append(n)
        return super().read(n)


class NetworkBoundary(unittest.TestCase):
    def setUp(self):
        self.hf = ck.PinnedHF()
        self.hf.opener = MagicMock()
        self.shard = ck.SHARDS[0]

    def test_refuses_ignored_wrong_or_encoded_range_without_read(self):
        for status, cr, cl, ce in [
            (200, "bytes 0-7/99999", "8", "identity"),
            (206, "bytes 1-8/99999", "8", "identity"),
            (206, "bytes 0-7/*", "8", "identity"),
            (206, "bytes 0-7/99999", "99999", "identity"),
            (206, "bytes 0-7/99999", "8", "gzip"),
            (206, "bytes 0-7/7", "8", "identity"),
        ]:
            with self.subTest(status=status, cr=cr, cl=cl, ce=ce):
                resp = Response(b"unused", status, **{"Content-Range": cr, "Content-Length": cl, "Content-Encoding": ce})
                self.hf.opener.open.return_value = resp
                with self.assertRaises(ValueError):
                    self.hf.read(self.shard, 0, 8)
                self.assertEqual(resp.reads, [])
                self.assertTrue(resp.closed)

    def test_redirect_not_drained_and_range_repeated(self):
        redirect = Response(b"redirect body", 302)
        target = "https://cas-bridge.xethub.hf.co/test"
        err = HTTPError("https://huggingface.co/test", 302, "redirect", {"Location": target}, redirect)
        ok = Response(b"12345678unread", 206, **{"Content-Range": "bytes 0-7/99999"})
        self.hf.opener.open.side_effect = [err, ok]
        self.assertEqual(self.hf.read(self.shard, 0, 8), (b"12345678", 99999))
        self.assertEqual(redirect.reads, [])
        self.assertEqual(ok.reads, [8])
        requests = [call.args[0] for call in self.hf.opener.open.call_args_list]
        self.assertEqual([r.get_header("Range") for r in requests], ["bytes=0-7"] * 2)
        self.assertEqual(requests[-1].full_url, target)

    def test_no_arbitrary_files_or_header_ranges(self):
        for name, start, n in [("dflash.safetensors", 0, 8), (self.shard, ck.START[self.shard], 1),
                               (self.shard, 8, 1 << 30), ("config.json", 0, 10)]:
            with self.subTest(name=name, start=start, n=n), self.assertRaises(ValueError):
                self.hf.read(name, start, n)
        self.hf.opener.open.assert_not_called()

    def test_unpinned_header_length_refused_before_second_get(self):
        # A server may answer 206 but provide an unsafe length prefix.
        resp = Response(struct.pack("<Q", 1 << 30), 206, **{"Content-Range": "bytes 0-7/99999999999"})
        self.hf.opener.open.return_value = resp
        with self.assertRaisesRegex(ValueError, "header length"):
            self.hf.header(self.shard)
        self.hf.opener.open.assert_called_once()
        self.assertEqual(resp.reads, [8])

    def test_short_read_and_inconsistent_total(self):
        for total, data, expected_reads in [(99999, b"abc", [8]), (99998, b"12345678", [])]:
            resp = Response(data, 206, **{"Content-Range": "bytes 0-7/99999"})
            self.hf.opener.open.return_value = resp
            with self.assertRaises(ValueError):
                self.hf.read(self.shard, 0, 8, total=total)
            self.assertEqual(resp.reads, expected_reads)

    def test_payload_budget_checked_before_any_request(self):
        for bad_cap in (28895, 32769):
            with self.assertRaisesRegex(ValueError, "requires"):
                ck.sample_projections(self.hf, {}, bad_cap, Path("unused"))
        self.hf.opener.open.assert_not_called()


@unittest.skipUnless(os.environ.get("GIGA4_LAYOUT_METADATA"), "set GIGA4_LAYOUT_METADATA to saved metadata")
class OfficialFiles(unittest.TestCase):
    def test_all_pinned_headers(self):
        root = Path(os.environ["GIGA4_LAYOUT_METADATA"])
        ck.pinned_json("config.json", (root / "config.json").read_bytes())
        index = ck.pinned_json(ck.INDEX, (root / ck.INDEX).read_bytes())
        headers = Path(os.environ.get("GIGA4_LAYOUT_HEADERS", root / "headers"))
        result, locations = ck.validate(index, lambda s: ck.parse((headers / (s + ".json")).read_bytes()))
        self.assertEqual(result["base"]["tensor_bytes"], 714789875712)
        self.assertEqual(result["mtp"]["tensor_bytes"], 4912973280)
        self.assertEqual((result["headers_checked"], result["names_checked"]), (45, 95616))
        self.assertEqual(set(locations), {ck.FP, ck.FP + "_scale_inv", ck.BF})


if __name__ == "__main__":
    unittest.main()
