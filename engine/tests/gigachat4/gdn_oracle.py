"""Regenerate gdn_test.cpp's tiny FP32 oracle. Needs CPU PyTorch, no model weights.

Pass the directory with the pinned modeling_gigachat4.py. Execute only its
pure-PyTorch GDN definitions: no model loading, Transformers, FLA or GPU.
"""

import ast
import hashlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def reference(path):
    src = path.read_bytes()
    if hashlib.sha256(src).hexdigest() != "b3ebfbb2e18cb90f638887699a720a9894c28afa2aea58b9c77fa6249e2d7c46":
        raise ValueError("modeling_gigachat4.py does not match the oracle revision")
    names = {
        "GigaChat4GatedDeltaRMSNorm", "torch_causal_conv1d_update", "l2norm",
        "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
        "_apply_linear_attention_padding_mask", "_build_gated_delta_norm", "GigaChat4GatedDeltaNetWrapper",
    }
    tree = ast.parse(src)
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    env = dict(torch=torch, nn=nn, F=F, logger=SimpleNamespace(warning_once=logging.warning),
               ACT2FN={"silu": F.silu},
               chunk_gated_delta_rule=None, fused_recurrent_gated_delta_rule=None, causal_conv1d_fn=None,
               causal_conv1d_update=None, FusedRMSNormGated=None, is_fast_path_available=False)
    import __future__
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), env)
    return env["GigaChat4GatedDeltaNetWrapper"]


def samples(rows, cols, seed):
    i, j = torch.arange(rows)[:, None], torch.arange(cols)
    return (((i * 13 + j * 7 + i * j * 3 + seed * (i + 2 * j + 1)) % 29) - 14).float() / 16


def emit(name, t):
    flat = t.flatten().tolist()
    print(f"const Vec {name} = {{")
    for i in range(0, len(flat), 5):
        print("    " + ", ".join(f"{n:.9e}f" for n in flat[i:i + 5]) + ",")
    print("};")


@torch.no_grad()
def main():
    cls = reference(Path(sys.argv[1]) / "modeling_gigachat4.py")
    config = SimpleNamespace(hidden_size=5, linear_num_value_heads=4, linear_num_key_heads=2,
                             linear_key_head_dim=3, linear_value_head_dim=2, linear_conv_kernel_dim=4,
                             hidden_act="silu", rms_norm_eps=1e-6, linear_attn_o_norm_eps=1/256,
                             linear_gating_type="gated_rmsnorm_sigmoid", linear_sigmoid_gate_scale=2.0)
    m = cls(config, 0).eval()
    for linear, seed in [(m.in_proj_qkvz, 1), (m.in_proj_ba, 2), (m.out_proj, 3)]:
        linear.weight.copy_(samples(*linear.weight.shape, seed))
    m.conv1d.weight.copy_(samples(20, 4, 4)[:, None, :])
    m.dt_bias.copy_(samples(1, 4, 5)[0])
    m.A_log.copy_(samples(1, 4, 6)[0])
    m.norm.weight.copy_(torch.tensor([1.25, -0.75]))
    x = samples(6, 5, 7)[None]
    x[:, 0] /= 16
    x[:, 3] = 0

    def cache():
        return SimpleNamespace(conv_states=[None], recurrent_states=[None], has_previous_state=False)
    c = cache()
    chunk = m(x, cache_params=c)
    d = cache()
    steps = []
    for t in range(x.shape[1]):
        steps.append(m(x[:, t:t+1], cache_params=d, cache_position=torch.tensor([t])))
        d.has_previous_state = True
    torch.testing.assert_close(chunk, torch.cat(steps, dim=1), atol=1e-6, rtol=1e-5)
    for a, b in [(c.conv_states, d.conv_states), (c.recurrent_states, d.recurrent_states)]:
        torch.testing.assert_close(a[0], b[0], atol=1e-6, rtol=1e-5)
    emit("expected_output", chunk)
    emit("expected_recurrent", c.recurrent_states[0])


if __name__ == "__main__":
    main()
