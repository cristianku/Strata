# GigaChat4: send a text prompt to the native runner

`tools/gigachat4/chat.py` turns a chat history into token IDs using the checkpoint's own tokenizer and chat template. It can print the IDs or pass them to `strata-gigachat4` and decode the reply. It is a standalone, opt-in tool.

## Run a request

From the repository root, install the tokenizer dependencies in a separate environment. PyTorch is not required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/gigachat4/requirements-chat.txt
```

For an offline install, supply the dependency wheels locally with `pip install --no-index --find-links /path/to/wheels -r tools/gigachat4/requirements-chat.txt`.

Set `MODEL` to a **local directory**, not a Hub ID. The frontend reads `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. It uses the installed Transformers backend and its sandboxed template renderer. It does not import checkpoint Python or download model files.

```sh
MODEL=/path/to/GigaChat4-708B-A41B
RUNNER=/path/to/strata-gigachat4
MESSAGES='[{"role":"user","content":"Привет! Hello!"}]'

# No weights or executable needed. Prints {"tokens":[...]}.
python tools/gigachat4/chat.py --model "$MODEL" --messages "$MESSAGES" --tokenize-only

# The runner now also needs the local model config, index and weight shards.
python tools/gigachat4/chat.py --model "$MODEL" --messages "$MESSAGES" \
  --runner "$RUNNER" --no-reasoning --generate 32
```

`--messages` accepts inline JSON, a UTF-8 JSON file, or `-` for stdin (the default). A request object can carry the whole conversation, reasoning setting and tool list together:

```sh
python tools/gigachat4/chat.py --model "$MODEL" --runner "$RUNNER" --generate 32 <<'JSON'
{
  "messages": [{"role": "user", "content": "Какая погода в Москве?"}],
  "reasoning": false,
  "tools": [{"type": "function", "function": {
    "name": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}]
}
JSON
```

`--reasoning`, `--no-reasoning` and `--tools` override the matching request fields. Reasoning defaults to on. Tool definitions and previous calls/results are rendered as prompt text; tools are not executed. Use JSON **objects** for historical tool-call arguments, as the official template requires.

With a CUDA build, pass `--cuda`. `--gpu-cache-mib N` and `--host-cache-mib N` forward the supplied budgets to the executable. Omitted flags leave the native defaults in place. The prompt is one comma-separated `--tokens` argument: the operating system's command-line length limit applies. The frontend does not truncate the conversation to fit.

## What comes back

Stdout is one JSON object. `generated` and `metrics` come from the runner. `text` decodes **only the generated IDs**, in one pass so that UTF-8 characters split across tokens survive. A final EOS (`2`) is omitted from `text` but kept in `generated`. Whitespace, `</think>` and GCML tool markup are preserved. With reasoning on, the opening `<think>` is already in the prompt, so it is not prepended to the continuation. Native diagnostics go to stderr.

## Check the official tokenizer without weights

Verified against [`ai-sage/GigaChat4-708B-A41B` at `2baf8b0cf71f1528c14e7a274877d9c8fbf22697`](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/tree/2baf8b0cf71f1528c14e7a274877d9c8fbf22697), with Python 3.13.5, Transformers 5.8.1, Tokenizers 0.22.2 and Jinja2 3.1.6.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `tokenizer.json` | 10,681,085 | `359786e8a6ef4574d560d34c870ebb129120a0353b4ddd3960c36a50808d72f3` |
| `tokenizer_config.json` | 281 | `e7c784db68ecb8b8ac90e0a0f9e8bdd4bfe1d033fb4fec59aeec4968cc88b41c` |
| `chat_template.jinja` | 44,706 | `3446d26c36c28cc5af809c109d9764669c381c63ab5d84acafb55702ce65c5ef` |

The tokenizer defines 128,016 IDs (0–128,015); the model's output head has 128,256 rows. An output ID with no tokenizer entry is reported as an error rather than silently lost.

Point `GIGACHAT4_TOKENIZER_DIR` at a local copy of the three files. If needed, this **manual** download fetches just those files outside the checkout:

```sh
export GIGACHAT4_TOKENIZER_DIR="$(mktemp -d)"
base=https://huggingface.co/ai-sage/GigaChat4-708B-A41B/resolve/2baf8b0cf71f1528c14e7a274877d9c8fbf22697
for file in tokenizer.json tokenizer_config.json chat_template.jinja; do
  curl -fLsS "$base/$file" -o "$GIGACHAT4_TOKENIZER_DIR/$file" || exit 1
done
HF_HUB_OFFLINE=1 python tools/gigachat4/test_chat.py -v
```

The reference vectors were obtained separately with `AutoTokenizer.from_pretrained(local_dir, local_files_only=True, trust_remote_code=False)`. Plain `Привет, мир!` encodes to `[52597, 370, 14, 9073, 3]`; `Hello, world!` to `[16656, 14, 3621, 3]` (no added BOS). The bilingual chat example above takes **16 prompt IDs** with reasoning on and **18** with reasoning off. A four-message tool conversation matched all **152/153** reference IDs (on/off), including a Cyrillic JSON result and GCML arguments. The CLI tests also check native argv, process errors, split UTF-8 and the final EOS.

In a separate native CPU smoke test, a four-layer **synthetic** fixture with its embedding and output tables widened to the checkpoint vocabulary accepted the 18-ID prompt. The runner returned `[16656,16656,9073,9073]`, decoded as `HelloHello мир мир`. This exercises the text-to-native round trip; it is not a result from the 708B weights or a CUDA test.
