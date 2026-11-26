"""Offline tests. Set GIGACHAT4_TOKENIZER_DIR to also run the official vectors."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat


SIMPLE = [{"role": "user", "content": "Привет! Hello!"}]
THINKING_IDS = [1, 12319, 128000, 128001, 895, 128000, 52597, 370, 3, 52467, 3, 128001,
                558, 24039, 128000, 128014]
DIRECT_IDS = THINKING_IDS + [128015, 127825]
GENERATED = [12618, 102, 234, 128015, 127825, 52597, 370, 14, 3621, 3, 2]


def shim(folder, body):
    runner = Path(folder) / "native runner"
    runner.write_text(f"#!{sys.executable}\nimport json, sys\n" + body + "\n", encoding="utf-8")
    runner.chmod(0o755)
    return runner


class LocalContract(unittest.TestCase):
    def test_incomplete_checkpoint(self):
        with TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "missing tokenizer file"):
            chat.load_tokenizer(Path(d))

    @unittest.skipUnless(os.name == "posix", "executable script shim uses a POSIX shebang")
    def test_native_argv_and_failure(self):
        with TemporaryDirectory() as d:
            model = Path(d) / "checkpoint with spaces"
            runner = shim(d, 'print(json.dumps({"generated": [], "argv": sys.argv[1:]}))')
            result = chat.run_native(runner, model, [1, 12319], 7, True, 0, 512)
            self.assertEqual(result["argv"], ["--model", str(model), "--tokens", "1,12319", "--generate", "7",
                                             "--cuda", "--gpu-cache-mib", "0", "--host-cache-mib", "512"])
            for body, error in (("sys.exit(3)", "status 3"), ("print('not JSON')", "stdout")):
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, error):
                    chat.run_native(shim(d, body), model, [1], 7, False, None, None)


@unittest.skipUnless(os.environ.get("GIGACHAT4_TOKENIZER_DIR"), "set GIGACHAT4_TOKENIZER_DIR to official files")
class OfficialTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["GIGACHAT4_TOKENIZER_DIR"]).resolve()
        cls.t = chat.load_tokenizer(cls.model)

    def test_ru_en_with_and_without_template(self):
        for text, ids in (("Привет, мир!", [52597, 370, 14, 9073, 3]), ("Hello, world!", [16656, 14, 3621, 3])):
            self.assertEqual(self.t.encode(text, add_special_tokens=False), ids)
            self.assertEqual(chat.decode_result(self.t, {"generated": ids})["text"], text)
        self.assertEqual(chat.prompt_ids(self.t, SIMPLE, None, True), THINKING_IDS)
        self.assertEqual(chat.prompt_ids(self.t, SIMPLE, None, False), DIRECT_IDS)

    def test_tool_schema_calls_and_results(self):
        # Complete ID sequence digests from AutoTokenizer at the documented
        # revision. Covers text blocks, GCML, tojson(ensure_ascii=False),
        # tool result formatting, and reasoning retained/dropped in history.
        tools = [{"type": "function", "function": {
            "name": "weather", "description": "Погода", "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "array", "items": {"type": "integer"}},
            }, "required": ["city"]},
        }}]
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Погода?"}, {"type": "text", "text": "Weather?"}]},
            {"role": "assistant", "reasoning_content": "lookup", "content": None, "tool_calls": [{
                "id": "w1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Москва", "days": [1, 2]}},
            }]},
            {"role": "tool", "tool_call_id": "w1", "content": {"temp": 5, "label": "дождь"}},
        ]
        for reasoning, n, digest in (
            (True, 152, "614756ff4d609f3396a6ff73208d4cc2bc485aeec2e29df669b05fd46c28c528"),
            (False, 153, "aaefa5a86dc39acf10d67bf689f75dec929551d0c774a50e4d29a4425d200c23"),
        ):
            ids = chat.prompt_ids(self.t, messages, tools, reasoning)
            self.assertEqual(len(ids), n)
            self.assertEqual(hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(), digest)
        messages[-1]["tool_call_id"] = "wrong-id"
        with self.assertRaisesRegex(ValueError, "tool_call_id does not match"):
            chat.prompt_ids(self.t, messages, tools, True)

    def test_invalid_ids_are_not_silently_dropped(self):
        # The output head is padded beyond the tokenizer vocabulary.
        for bad in ({}, [], {"generated": "3"}, {"generated": [True]}, {"generated": [128016]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "generated"):
                chat.decode_result(self.t, bad)

    @unittest.skipUnless(os.name == "posix", "executable script shim uses a POSIX shebang")
    def test_full_cli_decodes_generated_only(self):
        with TemporaryDirectory() as d:
            runner = shim(d, f'print(json.dumps({{"generated": {GENERATED!r}, "argv": sys.argv[1:]}}))')
            p = subprocess.run(
                [sys.executable, str(Path(chat.__file__)), "--model", str(self.model), "--runner", str(runner),
                 "--generate", str(len(GENERATED))],
                input=json.dumps({"messages": SIMPLE, "reasoning": False, "tools": []}),
                encoding="utf-8", stdout=subprocess.PIPE, check=True,
            )
        result = json.loads(p.stdout)
        self.assertEqual(result["argv"], ["--model", str(self.model), "--tokens", ",".join(map(str, DIRECT_IDS)),
                                         "--generate", str(len(GENERATED))])
        self.assertEqual(result["generated"], GENERATED)
        self.assertEqual(result["text"], "🦉</think>\n\nПривет, world!")


if __name__ == "__main__":
    unittest.main()
