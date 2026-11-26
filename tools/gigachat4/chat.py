#!/usr/bin/env python3
"""Run one GigaChat4 chat request using a local tokenizer and native executable."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def read_json(source):
    if source == "-":
        return json.load(sys.stdin.buffer)
    if source.lstrip().startswith(("[", "{")):
        return json.loads(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def load_tokenizer(model):
    if not model.is_dir():
        raise ValueError(f"--model must be a local checkpoint directory: {model}")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (model / name).is_file():
            raise ValueError(f"missing tokenizer file: {model / name}")

    from transformers import TokenizersBackend

    # Use the installed backend directly. No Hub lookup, model config, or
    # checkpoint-provided Python class participates in loading these files.
    config = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    return TokenizersBackend(
        tokenizer_file=str(model / "tokenizer.json"),
        bos_token=config["bos_token"],
        eos_token=config["eos_token"],
        chat_template=(model / "chat_template.jinja").read_text(encoding="utf-8"),
    )


def prompt_ids(tokenizer, messages, tools, reasoning):
    from jinja2 import TemplateError

    if not isinstance(messages, list) or not messages or not all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages
    ):
        raise ValueError("messages must be a nonempty array of objects with a role")
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise ValueError("tools must be a JSON array of tool definitions")
    if not isinstance(reasoning, bool):
        raise ValueError("reasoning must be true or false")
    try:
        # apply_chat_template does not add a second BOS after rendering.
        return tokenizer.apply_chat_template(
            messages, tools=tools, reasoning=reasoning, add_generation_prompt=True,
            tokenize=True, return_dict=False,
        )
    except TemplateError as exc:
        raise ValueError(f"chat template: {exc}") from exc


def decode_result(tokenizer, result):
    generated = result.get("generated") if isinstance(result, dict) else None
    if not isinstance(generated, list) or any(
        type(i) is not int or not 0 <= i < 2**32 or tokenizer.backend_tokenizer.id_to_token(i) is None
        for i in generated
    ):
        raise ValueError("native output must contain a 'generated' array of valid tokenizer IDs")
    # Strip only the terminal EOS. </think> and GCML delimiters are part of
    # the continuation; decoding one ID at a time would corrupt split UTF-8.
    ids = generated[:-1] if generated and generated[-1] == tokenizer.eos_token_id else generated
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {**result, "text": text}


def run_native(runner, model, tokens, generate, cuda, gpu_cache_mib, host_cache_mib):
    cmd = [str(runner.resolve()), "--model", str(model), "--tokens", ",".join(map(str, tokens)),
           "--generate", str(generate)]
    if cuda:
        cmd.append("--cuda")
    for flag, value in (("--gpu-cache-mib", gpu_cache_mib), ("--host-cache-mib", host_cache_mib)):
        if value is not None:
            cmd.extend([flag, str(value)])
    child = subprocess.run(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    if child.returncode:
        raise ValueError(f"native runner exited with status {child.returncode}")
    try:
        return json.loads(child.stdout)
    except ValueError as exc:
        raise ValueError("native runner did not write a JSON object on stdout") from exc


def run(args):
    request = read_json(args.messages)
    if isinstance(request, list):
        request = {"messages": request}
    if not isinstance(request, dict):
        raise ValueError("expected a messages array or a chat request object")
    tools = read_json(args.tools) if args.tools is not None else request.get("tools")
    reasoning = args.reasoning if args.reasoning is not None else request.get("reasoning", True)
    model = args.model.resolve()
    tokenizer = load_tokenizer(model)
    tokens = prompt_ids(tokenizer, request.get("messages"), tools, reasoning)
    if args.tokenize_only:
        return {"tokens": tokens}
    result = run_native(args.runner, model, tokens, args.generate, args.cuda, args.gpu_cache_mib, args.host_cache_mib)
    return decode_result(tokenizer, result)


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="offline checkpoint directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runner", type=Path, help="path to the native GigaChat4 executable")
    mode.add_argument("--tokenize-only", action="store_true", help="print prompt token IDs without running inference")
    parser.add_argument("--messages", default="-", help="messages array or request object: JSON, file, or - (stdin)")
    parser.add_argument("--tools", help="override tool definitions: JSON array, file, or -")
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, help="override template reasoning (default: on)")
    parser.add_argument("--generate", type=nonnegative, default=64, help="maximum new tokens (default: 64)")
    parser.add_argument("--cuda", action="store_true", help="pass --cuda to the runner")
    parser.add_argument("--gpu-cache-mib", type=nonnegative)
    parser.add_argument("--host-cache-mib", type=nonnegative)
    args = parser.parse_args()
    if args.messages == args.tools == "-":
        parser.error("only one JSON argument can read stdin")
    try:
        result = run(args)
    except ImportError as exc:
        parser.exit(1, f"chat: install tools/gigachat4/requirements-chat.txt ({exc})\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"chat: {exc}\n")
    sys.stdout.buffer.write((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
