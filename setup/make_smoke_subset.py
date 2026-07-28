#!/usr/bin/env python3
"""
Carve a small smoke-test dataset out of a big PyG InMemoryDataset.

Why: the shipped graph datasets are tens of GB in a single `processed/data.pt`, and
`InMemoryDataset` loads that file whole - so a laptop cannot open one, and copying one
off the cluster to try a config is minutes of transfer for a run that needs 200 events.

Run this ON THE CLUSTER, next to the big file, then copy the ~30 MB result:

    python setup/make_smoke_subset.py SRC_DATASET_DIR DST_DATASET_DIR --n 200
    python setup/make_smoke_subset.py SRC_DATASET_DIR --inspect

SRC/DST are dataset *directories* (the ones a config's `graph_folder_path` points at);
the file itself is read from and written to `<dir>/processed/data.pt`.

Two properties make this cheap and portable:

  * `torch.load(..., mmap=True)` maps the file instead of reading it, so only the pages
    belonging to the events actually taken are ever touched - carving 200 events out of
    a 39 GB file costs a fraction of a second and no meaningful memory.
  * it works directly on PyG's collated `(data_dict, slices_dict, cls)` representation,
    so it needs torch ONLY - no torch_geometric on the machine doing the carving, which
    matters on a login node.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

# Keys collated along dim 1 rather than dim 0. PyG concatenates "*index*" attributes on
# their last dimension (edge_index is [2, num_edges]), everything else on the first.
_CAT_DIM_1 = ("edge_index",)


def _cat_dim(key: str) -> int:
    return 1 if "index" in key and key in _CAT_DIM_1 else 0


def _load(path: Path):
    # str(), not Path: torch < 2.6 raises "f must be a string filename in order to use
    # mmap argument" - and the cluster containers are exactly that old.
    obj = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(obj, tuple) or len(obj) < 2:
        raise SystemExit(f"{path}: not a collated InMemoryDataset file (got {type(obj)})")
    data, slices = obj[0], obj[1]
    rest = obj[2:]
    return data, slices, rest


def _num_graphs(slices: dict) -> int:
    return len(next(iter(slices.values()))) - 1


def describe(src: Path) -> None:
    data, slices, _ = _load(src / "processed" / "data.pt")
    n = _num_graphs(slices)
    print(f"{src.name}\n  graphs: {n}")
    for key, value in data.items():
        if torch.is_tensor(value):
            per = value.shape[_cat_dim(key)] / max(n, 1)
            print(f"  {key:12s} {tuple(value.shape)} {str(value.dtype):14s} ~{per:.1f}/graph")
    y = data.get("y")
    if torch.is_tensor(y) and y.numel() <= n * 4:
        uniq = torch.unique(y)
        if uniq.numel() <= 10:
            print(f"  y values: {uniq.tolist()}  (labels, e.g. PDG codes)")


def subset(src: Path, dst: Path, n: int, start: int = 0) -> None:
    src_file = src / "processed" / "data.pt"
    data, slices, rest = _load(src_file)

    total = _num_graphs(slices)
    if start + n > total:
        raise SystemExit(f"asked for graphs [{start}:{start + n}] but source has {total}")

    out_data, out_slices = {}, {}
    for key, value in data.items():
        if not torch.is_tensor(value) or key not in slices:
            out_data[key] = value          # non-collated attribute, keep as is
            continue
        dim = _cat_dim(key)
        offsets = slices[key]
        lo, hi = int(offsets[start]), int(offsets[start + n])
        # .clone() materialises off the mmap: only these pages are read.
        out_data[key] = value.narrow(dim, lo, hi - lo).clone()
        out_slices[key] = (offsets[start:start + n + 1] - offsets[start]).clone()

    (dst / "processed").mkdir(parents=True, exist_ok=True)
    torch.save((out_data, out_slices, *rest), dst / "processed" / "data.pt")

    # InMemoryDataset checks for these and warns (or reprocesses) when they are absent.
    for extra in ("pre_filter.pt", "pre_transform.pt"):
        if (src / "processed" / extra).exists():
            shutil.copy2(src / "processed" / extra, dst / "processed" / extra)

    size_mb = (dst / "processed" / "data.pt").stat().st_size / 1e6
    print(f"wrote {n} graphs ({start}..{start + n - 1} of {total}) -> {dst}  [{size_mb:.1f} MB]")
    describe(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="source dataset directory")
    parser.add_argument("dst", type=Path, nargs="?", help="destination dataset directory")
    parser.add_argument("--n", type=int, default=200, help="number of graphs (default 200)")
    parser.add_argument("--start", type=int, default=0, help="index of the first graph")
    parser.add_argument("--inspect", action="store_true", help="describe the source and exit")
    args = parser.parse_args()

    if args.inspect:
        describe(args.src)
        return
    if args.dst is None:
        parser.error("dst is required unless --inspect is given")
    subset(args.src, args.dst, args.n, args.start)


if __name__ == "__main__":
    main()
