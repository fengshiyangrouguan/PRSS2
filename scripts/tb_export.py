#!/usr/bin/env python3
"""Export TensorBoard scalar curves from an event file to PNG plots.

Pure-stdlib TFRecord/Event wire-format reader (no tensorboard dependency),
then matplotlib renders train/val curves.

Usage:
    python scripts/tb_export.py --logdir outputs/.../tb --outdir outputs/.../tb_png
"""

import argparse
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- event parsing


def read_varint(buf: bytes, i: int):
    shift = 0
    val = 0
    while True:
        byte = buf[i]
        i += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, i
        shift += 7


def skip_field(buf: bytes, i: int, wire: int) -> int:
    if wire == 0:
        _, i = read_varint(buf, i)
    elif wire == 1:
        i += 8
    elif wire == 2:
        ln, i = read_varint(buf, i)
        i += ln
    elif wire == 5:
        i += 4
    else:
        raise ValueError(f"unsupported wire type {wire}")
    return i


def parse_event(data: bytes):
    """Return (step, {tag: simple_value}) from a serialized Event proto."""
    i = 0
    step = None
    summary = None
    while i < len(data):
        tag_num, i = read_varint(data, i)
        field, wire = tag_num >> 3, tag_num & 7
        if field == 1 and wire == 1:
            i += 8  # wall_time
        elif field == 2:
            step, i = read_varint(data, i)
        elif field == 5 and wire == 2:
            # Event.summary (field 5 per tensorboard event.proto)
            ln, i = read_varint(data, i)
            summary = data[i:i + ln]
            i += ln
        else:
            i = skip_field(data, i, wire)
    vals = {}
    if summary:
        i = 0
        while i < len(summary):
            tag_num, i = read_varint(summary, i)
            field, wire = tag_num >> 3, tag_num & 7
            if field == 1 and wire == 2:
                ln, i = read_varint(summary, i)
                val = data  # placeholder; real buffer below
                val = summary[i:i + ln]
                i += ln
                vi = 0
                vtag, vval = None, None
                while vi < len(val):
                    t2, vi = read_varint(val, vi)
                    f2, w2 = t2 >> 3, t2 & 7
                    if f2 == 2 and w2 == 2:
                        ln2, vi = read_varint(val, vi)
                        vtag = val[vi:vi + ln2].decode("utf-8", "replace")
                        vi += ln2
                    elif f2 == 3 and w2 == 5:
                        vval = struct.unpack("<f", val[vi:vi + 4])[0]
                        vi += 4
                    else:
                        vi = skip_field(val, vi, w2)
                if vtag and vval is not None:
                    vals[vtag] = vval
            else:
                i = skip_field(summary, i, wire)
    return step, vals


def read_events(path: Path):
    """Yield (step, {tag: value}) for every summary event in the file."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            length = struct.unpack("<Q", hdr)[0]
            f.read(4)  # crc32c of length
            if length == 0:
                continue
            data = f.read(length)
            f.read(4)  # crc32c of data
            step, vals = parse_event(data)
            if step is not None and vals:
                yield step, vals


def collect(logdir: Path):
    """Merge all event files in logdir into {tag: [(step, value), ...]}."""
    series = {}
    for ev in sorted(logdir.glob("events.out.tfevents.*")):
        for step, vals in read_events(ev):
            for tag, value in vals.items():
                series.setdefault(tag, []).append((step, value))
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


# ----------------------------------------------------------------- plotting


def plot_series(series, outdir: Path, epochs_per_step: float = 1.0):
    """Render one PNG per tag (val curves) plus one combined train-loss PNG."""
    outdir.mkdir(parents=True, exist_ok=True)
    val_tags = [t for t in series if t.startswith("val/") or t.startswith("test/")]
    train_tags = [t for t in series if t.startswith("train/")]

    # per-tag val curves
    for tag in val_tags:
        pts = series[tag]
        xs = [s * epochs_per_step for s, _ in pts]
        ys = [v for _, v in pts]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, "-o", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        name = tag.replace("/", "__")
        fig.savefig(outdir / f"{name}.png", dpi=150)
        plt.close(fig)

    # combined train losses
    if train_tags:
        fig, ax = plt.subplots(figsize=(7, 4))
        for tag in train_tags:
            pts = series[tag]
            xs = [s * epochs_per_step for s, _ in pts]
            ys = [v for _, v in pts]
            ax.plot(xs, ys, label=tag, lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("train losses")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "train_losses.png", dpi=150)
        plt.close(fig)

    return [p.name for p in outdir.glob("*.png")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True, help="TensorBoard event dir")
    p.add_argument("--outdir", required=True, help="PNG output dir")
    p.add_argument("--epochs-per-step", type=float, default=1.0,
                   help="multiplier converting global_step to epoch for x-axis")
    args = p.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    series = collect(logdir)
    if not series:
        print(f"no scalar events found in {logdir}")
        return
    for tag, pts in sorted(series.items()):
        n = len(pts)
        print(f"{tag:32s} {n:5d} points  last={pts[-1][1]:.6g}")
    pngs = plot_series(series, outdir, args.epochs_per_step)
    print(f"\nwrote {len(pngs)} PNGs to {outdir}:")
    for name in pngs:
        print(f"  {name}")


if __name__ == "__main__":
    main()
