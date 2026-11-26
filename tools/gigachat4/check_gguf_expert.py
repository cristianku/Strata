"""Offline probe check. Needs only shard 1's authentic directory, never its weights."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("probe", type=Path)
    cli.add_argument("shard1", type=Path, help="official shard 1 or its first 6,122,781 bytes")
    opts = cli.parse_args()
    probe = opts.probe.resolve(strict=True)
    with opts.shard1.open("rb") as src:
        header = src.read(6_122_781)
    if len(header) != 6_122_781:
        raise ValueError("short shard 1 header")
    # No official weights in this fixture. Use a filesystem supporting large sparse files.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard = root / "GigaChat4-708B-A41B-Q4_K_M-00001-of-00009.gguf"
        with shard.open("wb") as f:
            f.write(header)
            f.truncate(48_721_510_688)

        def run(proj, *args, ok=True):
            p = subprocess.run([probe, shard, "3", "255", proj, *map(str, args)],
                               capture_output=True, text=True)
            if (p.returncode == 0) != ok:
                raise AssertionError(f"unexpected status {p.returncode}: {p.stderr}")

        for proj, rows, cols, tensor_offset, block, expected in (
            # Q4_K: d=1, min=0, first group scale=1; each low nibble is 3.
            ("gate", 2048, 7168, 5_730_662_400,
             struct.pack("<ee", 1, 0) + b"\x01" + bytes(11) + b"\x03"*128, 3.),
            # Q6_K: all low/high bits zero = -32; scale=1; d=0.5.
            ("down", 7168, 2048, 2_635_806_720,
             bytes(192) + b"\x01"*16 + struct.pack("<e", .5), -16.),
        ):
            # Last expert, last row and last block. Adjacent rows/experts stay zero.
            stride = rows * (cols//256) * len(block)
            with shard.open("r+b") as f:
                f.seek(6_122_784 + tensor_offset + 256*stride - len(block))
                f.write(block)
            target = root / f"{proj}.bin"
            run(proj, "pack", target)
            assert target.stat().st_size == stride
            with target.open("rb") as f:
                assert not any(f.read(stride - len(block))) and f.read() == block
            run(proj, "block", rows-1, cols//256-1, target, ok=False)  # exclusive output
            assert target.stat().st_size == stride
            target.unlink()
            run(proj, "block", rows, 0, target, ok=False)
            run(proj, "block", rows-1, cols//256, target, ok=False)
            assert not target.exists()
            inp = root / "x.f32"
            inp.write_bytes(bytes((cols-256)*4) + struct.pack("<f", 1.) + bytes(255*4))
            run(proj, "matvec", inp, target)
            assert target.read_bytes() == bytes((rows-1)*4) + struct.pack("<f", expected)
            target.unlink()
        # A different header at the correct official size must be refused before parsing.
        with shard.open("r+b") as f:
            f.seek(8)
            f.write(b"\xff"*8)
        run("up", "pack", root/"bad", ok=False)
        assert not (root/"bad").exists()
    print("PASS: Q4_K and Q6_K last-expert slicing/matvec, range and header checks")


if __name__ == "__main__":
    main()
