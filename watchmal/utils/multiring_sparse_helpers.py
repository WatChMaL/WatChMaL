# -*- coding: utf-8 -*-
"""
Helpers for MultiRing sparse-CNN training and test engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from watchmal.dataset.multiring.sparse_cnn import VoxelGridConfig


def save_loss(out_path: Path, ytr: List[float], yva: List[float], ylabel: str, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(ytr), len(yva))
    if n <= 0:
        return
    x = list(range(1, n + 1))
    plt.figure()
    plt.plot(x, ytr[:n], label="train")
    plt.plot(x, yva[:n], label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def collate_sparse(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    coords_list, feats_list, meta_list = [], [], []
    for b, s in enumerate(samples):
        c_np = np.asarray(s["coords"], dtype=np.int32).copy()
        c_np[:, 0] = b
        coords_list.append(torch.from_numpy(c_np))

        f_np = np.asarray(s["feats"], dtype=np.float32)
        feats_list.append(torch.from_numpy(f_np))

        m = dict(s.get("meta", {}))  # shallow copy so we don't mutate the cached sample

        v = m.get("voxel_parent_frac")
        if v is not None and not torch.is_tensor(v):
            m["voxel_parent_frac"] = torch.from_numpy(np.asarray(v, dtype=np.float32))

        v = m.get("voxel_parent_id")
        if v is not None and not torch.is_tensor(v):
            m["voxel_parent_id"] = torch.from_numpy(np.asarray(v, dtype=np.int64))

        v = m.get("voxel_charge_pe")
        if v is not None and not torch.is_tensor(v):
            m["voxel_charge_pe"] = torch.from_numpy(np.asarray(v, dtype=np.float32)).view(-1)

        meta_list.append(m)

    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "meta": meta_list,
    }



def worker_init_fn(_):
    mp.set_sharing_strategy("file_system")
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class LoaderCfg:
    batch_size: int = 1
    num_workers: int = 0
    split: float = 0.9
    sampler_config: Optional[Any] = None
    cache_refresh_epochs: Optional[int] = None

@dataclass
class TestLoaderCfg:
    batch_size: int = 1
    num_workers: int = 0
    indices_path: Optional[str] = None
    max_events: Optional[int] = None


def import_from_path(path: str):
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Invalid import path: {path}")
    mod_name, obj_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)

def load_train_cfg(train_output_dir: str) -> DictConfig:
    p = Path(str(train_output_dir)) / ".hydra" / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Training config not found: {p}")
    return OmegaConf.load(str(p))


def to_container(x):
    if x is None:
        return {}
    if isinstance(x, DictConfig):
        return OmegaConf.to_container(x, resolve=True) or {}
    if isinstance(x, dict):
        return dict(x)
    return {}

def deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def build_model_from_train_output(
    train_output_dir: str,
    device: Optional[str] = None,
    ckpt_name: str = "MultiRingSegEngine_MultiRingModel_BEST.pth",
) -> torch.nn.Module:
    train_cfg = load_train_cfg(train_output_dir)
    model: torch.nn.Module = instantiate(train_cfg.model)
    if device is not None:
        model = model.to(torch.device(device))

    ckpt = Path(train_output_dir) / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_dataset_from_train_output(
    train_output_dir: str,
    dataset_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    train_cfg = load_train_cfg(train_output_dir)
    ds_cfg = train_cfg.data.dataset
    target_path = getattr(ds_cfg, "target_3d", None)
    params = to_container(getattr(ds_cfg, "params", None))
    overrides = to_container(dataset_overrides)
    deep_update(params, overrides)

    params["augmentation"] = False
    params["add_meta"] = True
    params["min_charge"] = 0.01

    if "grid" in params and isinstance(params["grid"], dict):
        # Legacy compat: old (4D) train configs nested time_bin_ns under grid;
        # VoxelGridConfig does not accept it, so drop it when restoring those runs.
        params["grid"].pop("time_bin_ns", None)
        params["grid"] = VoxelGridConfig(**params["grid"])
    ds_cls = import_from_path(str(target_path))
    return ds_cls(**params)
