# -*- coding: utf-8 -*-
"""
Engine for the MultiRing segmentation model.

MultiRingSegEngine subclasses BaseEngine and reuses its configuration,
distributed-reduce and state-init helpers. Only the segmentation-specific
pieces live here: the sparse voxel data pipeline, the per-voxel set-loss
training/validation loop, checkpointing with scheduler/epoch state, and the
(optional) diagnostics hook.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import os
import sys

os.environ["SPCONV_ALGO"] = "native"

import numpy as np
from datetime import datetime
import torch
import torch.nn.utils as nn_utils
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from hydra.utils import instantiate

from watchmal.engine.base_engine import BaseEngine
from watchmal.dataset.multiring.sparse_cnn import VoxelGridConfig, HyperKSparseCNN3D
from watchmal.dataset.samplers.batch_file_sampler import BatchFileSampler
from watchmal.utils.logging_utils_caverns import setup_logging
from watchmal.utils.banner import show_training_banner
from watchmal.utils.multiring_sparse_helpers import (
    LoaderCfg,
    TestLoaderCfg,
    build_dataset_from_train_output,
    build_model_from_train_output,
    collate_sparse,
    save_loss,
    to_container,
    worker_init_fn,
)

mp.set_sharing_strategy("file_system")

log = setup_logging(__name__)


def _load_run_diagnostics():
    """
    Lazily import run_diagnostics from the optional `diagnostic_multiring` submodule.

    Importing it at module load would make the whole engine unimportable when the
    (private) submodule is not checked out, so we defer it to the point where
    diagnostics are actually requested and raise a clear, actionable error otherwise.
    """
    repo_root = Path(__file__).resolve().parents[3]
    diag_src = repo_root / "submodules" / "diagnostic_multiring" / "src"
    if str(diag_src) not in sys.path:
        sys.path.insert(0, str(diag_src))
    try:
        from diagnostic_multiring.wrapper_diagnostic import run_diagnostics
    except ImportError as e:
        raise ImportError(
            "Diagnostics were requested but the 'diagnostic_multiring' submodule is not "
            "available. Run `git submodule update --init submodules/diagnostic_multiring`, "
            "or disable diagnostics by setting the task's `diagnostic.enabled: []`."
        ) from e
    return run_diagnostics


class MultiRingSegEngine(BaseEngine):
    CKPT_NAME = "MultiRingSegEngine_MultiRingModel_BEST.pth"

    def __init__(
        self,
        target_key,
        model,
        rank,
        device,
        dump_path,
        wandb_run=None,
        dataset=None,
        train_output_dir: Optional[str] = None,
        dataset_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target_key=target_key,
            model=model,
            rank=rank,
            device=device,
            dump_path=dump_path,
            wandb_run=wandb_run,
            dataset=dataset,
        )
        self.is_pyg = False

        # Segmentation-specific state
        self.data_config = None
        self.early_stopping = None
        self.history = {
            "train": {"loss": [], "loss_ce": [], "loss_dice": []},
            "val": {"loss": [], "loss_ce": [], "loss_dice": []},
        }
        self.val_subset = None
        self.test_subset = None
        self.cache_refresh_epochs = None
        # Diagnostics are configured per-task: train()/test() receive the `diagnostic`
        # block and store it here. enabled/out_dir are read from it (single source).
        self._diagnostic_cfg = None

        self.train_output_dir = str(train_output_dir) if train_output_dir is not None else None
        self.train_output_dir_p = Path(self.train_output_dir) if self.train_output_dir is not None else None
        self.dataset_overrides = to_container(dataset_overrides)

        log.info(f"File sharing strategy set to {mp.get_sharing_strategy()}")

    def configure_early_stopping(self, early_stopping_config):
        self.early_stopping = instantiate(early_stopping_config)

    # ------------------------------------------------------------------ #
    # Dataset / data loaders
    # ------------------------------------------------------------------ #
    def configure_dataset(self, data_config):
        """
        Stash the data config so configure_data_loaders can build the sparse dataset.

        The sparse pipeline builds its dataset (and train/val split) inside
        configure_data_loaders rather than from a precomputed split file, so here we
        only record the config and any target names.
        """
        self.data_config = data_config
        self.is_pyg = False
        ds_cfg = data_config.dataset
        if hasattr(ds_cfg, "target_names"):
            self.target_names = list(ds_cfg.target_names)

    def configure_data_loaders(self, loaders_config):
        loaders_cfg = OmegaConf.to_container(loaders_config, resolve=True) or {}

        if "train" in loaders_cfg or "validation" in loaders_cfg:
            self._configure_train_val_data_loaders(self.data_config, loaders_cfg)

        if "test" in loaders_cfg:
            self._configure_test_data_loader(loaders_cfg)

    def _configure_train_val_data_loaders(self, data_config, loaders_cfg):
        ds_cfg = data_config.dataset
        params = OmegaConf.to_container(ds_cfg.params, resolve=True) or {}
        if "grid" in params and isinstance(params["grid"], dict):
            params["grid"] = VoxelGridConfig(**params["grid"])
        if params.get("seed") is not None:
            log.warning(
                f"Dataset params still set a `seed`; it is ignored. Randomness now comes "
                f"from the top-level `seed` ({self.seed}). Remove the key from the config."
            )
        # Single source of randomness: the run-wide seed, so file selection and the
        # train/val split below are reproducible and identical on every rank.
        params["seed"] = self.seed
        full_ds = HyperKSparseCNN3D(**params)

        train_cfg = LoaderCfg(**(loaders_cfg.get("train", {}) or {}))
        val_cfg = LoaderCfg(**(loaders_cfg.get("validation", {}) or {}))

        self.cache_refresh_epochs = train_cfg.cache_refresh_epochs
        log.info(f"Configured data loaders with train/val split: {train_cfg.split}/{1 - train_cfg.split}")

        n = len(full_ds)
        k = int(n * train_cfg.split)
        # Dedicated generator on the un-offset run seed: the train/val split has to come
        # out identical on every rank, so it must not read the global RNG, which carries
        # a per-rank offset during training.
        split_generator = torch.Generator()
        split_generator.manual_seed(self.seed)
        idx = torch.randperm(n, generator=split_generator).tolist()
        train_idx, val_idx = (idx[:k], idx[k:]) if k < n else (idx, idx[:0])

        d_train = Subset(full_ds, train_idx)
        d_val = Subset(full_ds, val_idx)
        self.val_subset = d_val

        if self.is_distributed:
            train_sampler = DistributedSampler(
                d_train,
                num_replicas=self.n_gpus,
                rank=self.rank,
                shuffle=False,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                d_val,
                num_replicas=self.n_gpus,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

            self.data_loaders["train"] = DataLoader(
                d_train,
                batch_size=train_cfg.batch_size,
                num_workers=train_cfg.num_workers,
                shuffle=False,
                sampler=train_sampler,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2 if train_cfg.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_sparse,
            )
        else:
            bsampler = BatchFileSampler(
                subset=d_train,
                batch_size=train_cfg.batch_size,
                shuffle_files=False,
                shuffle_events_within_file=False,
                seed=self.seed,
                drop_last=False,
            )

            self.data_loaders["train"] = DataLoader(
                d_train,
                batch_sampler=bsampler,
                num_workers=train_cfg.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2 if train_cfg.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_sparse,
            )

            val_sampler = instantiate(val_cfg.sampler_config, indices=val_idx) if val_cfg.sampler_config is not None else None

        self.data_loaders["validation"] = DataLoader(
            d_val,
            batch_size=val_cfg.batch_size or train_cfg.batch_size,
            num_workers=val_cfg.num_workers,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            prefetch_factor=2 if val_cfg.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_sparse,
        )

    def _configure_test_data_loader(self, loaders_cfg) -> None:
        if self.rank != 0:
            return

        if self.train_output_dir is None:
            raise ValueError("test() needs train_output_dir in hydra_config.engine.params or hydra_config.engine")

        if self.model is None:
            self.model = build_model_from_train_output(
                train_output_dir=self.train_output_dir,
                device=str(self.device),
                ckpt_name=self.CKPT_NAME,
            )
            self.module = self.model
            self.is_distributed = False
            self.n_gpus = 1
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
            self.module = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

        if self.dataset is None:
            self.dataset = build_dataset_from_train_output(
                train_output_dir=self.train_output_dir,
                dataset_overrides=self.dataset_overrides,
            )

        full_ds = self.dataset
        test_cfg = TestLoaderCfg(**(loaders_cfg.get("test", {}) or {}))

        indices = list(range(len(full_ds)))
        log.info(f"[TEST] Full test dataset has {len(full_ds)} events")

        if test_cfg.indices_path:
            p = Path(test_cfg.indices_path)
            if not p.exists():
                raise FileNotFoundError(f"indices_path not found: {p}")
            if p.suffix == ".npy":
                indices = np.load(str(p)).astype(int).tolist()
            else:
                indices = [int(x) for x in p.read_text().strip().split()]

        if test_cfg.max_events is not None:
            indices = indices[: int(test_cfg.max_events)]

        self.test_subset = Subset(full_ds, indices)

        self.data_loaders["test"] = DataLoader(
            self.test_subset,
            batch_size=int(test_cfg.batch_size),
            num_workers=int(test_cfg.num_workers),
            shuffle=False,
            pin_memory=True,
            persistent_workers=(int(test_cfg.num_workers) > 0),
            collate_fn=collate_sparse,
        )

        log.info(f"[TEST] Train cfg : {self.train_output_dir_p / '.hydra' / 'config.yaml'}")
        log.info(f"[TEST] Train ckpt: {self._resolve_checkpoint_path()}")
        log.info(f"[TEST] Test out : {Path(os.getcwd()).resolve()}")
        log.info(f"[TEST] Test base_dir: {getattr(full_ds, 'base_dir', None)}")

    # ------------------------------------------------------------------ #
    # Batch / target handling
    # ------------------------------------------------------------------ #
    def _move_to_device(self, batch):
        return {
            "coords": batch["coords"].to(self.device, non_blocking=True),
            "feats": batch["feats"].to(self.device, non_blocking=True),
            "meta": batch["meta"],
        }

    def _extract_target_if_any_(self, batch: Dict[str, Any]) -> None:
        meta = batch.get("meta", None)
        self.target = None

        def _prepare_tensor(x: torch.Tensor) -> torch.Tensor:
            if self.target_key == "voxel_parent_frac":
                x = x.float()
            else:
                x = x.long()
            return x.to(self.device, non_blocking=True)

        if isinstance(meta, list):
            t_list = []
            for m in meta:
                if self.target_key in m:
                    val = m[self.target_key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.as_tensor(val)
                    t_list.append(val)
            if t_list:
                self.target = _prepare_tensor(torch.cat(t_list, dim=0))
        elif isinstance(meta, dict) and self.target_key in meta:
            val = meta[self.target_key]
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            self.target = _prepare_tensor(val)

    def forward(self, forward_type):
        batch = self.data
        out = self.model(batch)
        loss_dict = self.criterion(batch=batch, out=out)
        self.loss = loss_dict["loss"]
        outputs = {}
        return outputs, loss_dict

    def backward(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        nn_utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------ #
    # Diagnostics plotting helpers
    # ------------------------------------------------------------------ #
    def _get_run_img_dir(self) -> Path:
        out_dir = (self._diagnostic_cfg or {}).get("out_dir", "img")
        img_dir = Path(os.getcwd()) / out_dir
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def _append_epoch_history(self, split, metrics_mean):
        for k in ("loss", "loss_ce", "loss_dice"):
            if k in metrics_mean:
                self.history[split][k].append(float(metrics_mean[k]))

    def _plot_curves(self):
        if self.rank != 0:
            return
        img_dir = self._get_run_img_dir()
        save_loss(
            img_dir / "train_val_loss.png",
            self.history["train"]["loss"],
            self.history["val"]["loss"],
            ylabel="Loss (mean per epoch)",
            title="Train/Val Loss",
        )
        save_loss(
            img_dir / "train_val_ce.png",
            self.history["train"]["loss_ce"],
            self.history["val"]["loss_ce"],
            ylabel="Cross-Entropy (mean per epoch)",
            title="Train/Val Cross-Entropy",
        )
        save_loss(
            img_dir / "train_val_dice.png",
            self.history["train"]["loss_dice"],
            self.history["val"]["loss_dice"],
            ylabel="Dice penalty (mean per epoch)",
            title="Train/Val Dice penalty",
        )

    # ------------------------------------------------------------------ #
    # Train / validate
    # ------------------------------------------------------------------ #
    def sub_train(self, loader, val_interval):
        self.model.train()
        metrics_epoch_history = {"loss": 0.0}

        for step, train_data in enumerate(loader):
            batch = self._move_to_device(train_data)
            self.data = batch
            self._extract_target_if_any_(batch)
            outputs, metrics = self.forward(forward_type="train")
            self.loss = metrics["loss"]
            self.backward()
            metrics = {k: v.item() for k, v in metrics.items()}
            outputs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in outputs.items()}

            if self.rank == 0:
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"train_batch_" + k: v for k, v in outputs.items()} | {"train_batch_" + k: v for k, v in metrics.items()}
                    )
                for k in metrics.keys():
                    metrics_epoch_history.setdefault(k, 0.0)
                    metrics_epoch_history[k] += metrics[k]

            if (step % val_interval == 0) and (self.rank == 0):
                log.info(
                    f"GPU : {self.device} | Steps : {step + 1}/{len(loader)} "
                    f"| Iteration : {self.iteration + step}"
                )
                log.info("Batch metrics " + ", ".join(f"{k}: {v:.5g}" for k, v in metrics.items()))

        self.iteration += step + 1

        if self.rank == 0:
            for k in metrics_epoch_history.keys():
                metrics_epoch_history[k] /= (step + 1)
            self._append_epoch_history("train", metrics_epoch_history)

        return metrics_epoch_history

    def sub_validate(self, loader, forward_type="val", **kwargs):
        metrics_epoch_history = {"loss": 0.0}

        self.model.eval()
        with torch.no_grad():
            for step, val_batch in enumerate(loader):
                batch = self._move_to_device(val_batch)
                self.data = batch
                self._extract_target_if_any_(batch)

                outputs, metrics = self.forward(forward_type=forward_type)

                if self.is_distributed:
                    metrics = self.get_reduced(metrics)
                    outputs = self.get_reduced(outputs)

                if self.rank == 0:
                    metrics = {k: v.item() for k, v in metrics.items()}
                    outputs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in outputs.items()}

                    for k in metrics.keys():
                        metrics_epoch_history.setdefault(k, 0.0)
                        metrics_epoch_history[k] += metrics[k]

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {"val_batch_" + k: v for k, v in outputs.items()}
                            | {"val_batch_" + k: v for k, v in metrics.items()}
                        )

        if self.rank == 0:
            for k in metrics_epoch_history.keys():
                metrics_epoch_history[k] /= (step + 1)

            self._append_epoch_history("val", metrics_epoch_history)
            self._plot_curves()

        enabled = list((self._diagnostic_cfg or {}).get("enabled", []) or [])
        if enabled and (not self.is_distributed or self.rank == 0):
            run_diagnostics = _load_run_diagnostics()
            eval_model = self.module if self.is_distributed else self.model
            diag_cfg = dict(self._diagnostic_cfg)
            diag_cfg["enabled"] = enabled
            run_diagnostics(
                model=eval_model,
                criterion=self.criterion,
                val_subset=self.val_subset,
                device=self.device,
                run_dir=Path(os.getcwd()),
                diag_cfg=diag_cfg,
            )

        if self.is_distributed:
            torch.distributed.barrier()

        return metrics_epoch_history

    def train(self, epochs=0, val_interval=20, checkpointing=False, save_interval=None, diagnostic=None, **kwargs):

        start_run_time = datetime.now()
        log.info(f"Engine : {self.rank} | Dataloaders : {self.data_loaders}")
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
            show_training_banner(
                engine="multiring/segmentation",
                device=self.device,
                params=n_params,
                epochs=epochs,
            )

        self.iteration = 1

        self.best_training_loss = np.inf
        self.best_validation_loss = np.inf

        train_loader = self.data_loaders["train"]
        val_loader = self.data_loaders["validation"]
        self._diagnostic_cfg = to_container(diagnostic)

        cache_refresh_epochs = self.cache_refresh_epochs if self.cache_refresh_epochs else 15

        if self.wandb_run is not None and self.rank == 0:
            self.wandb_run.watch(self.module, log="all", log_freq=val_interval * 2, log_graph=False)

        for epoch in range(epochs):
            self.epoch = epoch
            if cache_refresh_epochs and (epoch % cache_refresh_epochs == 0):
                ds = train_loader.dataset
                root_ds = getattr(ds, "dataset", ds)
                if hasattr(root_ds, "clear_cache"):
                    if self.rank == 0:
                        log.info(f"[Epoch {epoch + 1}] Clearing train dataset cache")
                    root_ds.clear_cache()

            epoch_start_time = datetime.now()
            if self.rank == 0:
                log.info(f"\n\nTraining epoch {self.epoch + 1}/{epochs} starting at {epoch_start_time}")
                if self.optimizer is not None:
                    current_lr0 = self.optimizer.param_groups[0]["lr"]
                    log.info(f"[Epoch {self.epoch + 1}] Current LR: {current_lr0:.6g}")
                    if self.wandb_run is not None:
                        self.wandb_run.log({"learning_rate": current_lr0, "epoch": self.epoch})

            if self.is_distributed:
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)
            else:
                bsampler = getattr(train_loader, "batch_sampler", None)
                if bsampler is not None and hasattr(bsampler, "set_epoch"):
                    bsampler.set_epoch(self.epoch)

            metrics_epoch_history = self.sub_train(train_loader, val_interval)

            if (self.scheduler is not None) and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                if self.scheduler.get_last_lr() != current_lr:
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")

            epoch_end_time = datetime.now()

            if self.rank == 0:
                log.info(f"(Train) Epoch : {epoch + 1} completed in {(epoch_end_time - epoch_start_time)} | Iteration : {self.iteration} ")
                log.info(f"Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info("Metrics over the (train) epoch " + ", ".join(f"{k}: {v:.5g}" for k, v in metrics_epoch_history.items()))

                if self.wandb_run is not None:
                    self.wandb_run.log({"epoch": epoch})
                    self.wandb_run.log({"train_epoch_" + k: v for k, v in metrics_epoch_history.items()})

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.wandb_run.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})
                        else:
                            self.wandb_run.log({"learning_rate": self.scheduler.get_last_lr()[0]})

                    if metrics_epoch_history["loss"] < self.best_training_loss:
                        self.best_training_loss = metrics_epoch_history["loss"]
                        self.wandb_run.log({"best_train_epoch_loss": self.best_training_loss})

            if self.rank == 0:
                log.info("")
                log.info(f" -- Validation epoch starting at {datetime.now()}")

            metrics_epoch_history = self.sub_validate(val_loader, forward_type="val")

            if self.early_stopping is not None:
                stop_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
                if self.rank == 0:
                    self.early_stopping(metrics_epoch_history["loss"])
                    stop_flag.fill_(int(self.early_stopping.should_stop))
                if self.is_distributed:
                    torch.distributed.broadcast(stop_flag, src=0)
            else:
                stop_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)

            if (self.scheduler is not None) and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(metrics_epoch_history["loss"])
                if self.optimizer.param_groups[0]["lr"] != current_lr:
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")

            if self.rank == 0:
                log.info(f" -- Validation epoch completed | Iteration : {self.iteration}")
                log.info(" -- Metrics over the (val) epoch " + ", ".join(f"{k}: {v:.5g}" for k, v in metrics_epoch_history.items()))

                if self.wandb_run is not None:
                    self.wandb_run.log({"val_epoch_" + k: v for k, v in metrics_epoch_history.items()})

                if checkpointing:
                    self.save_state()

                if metrics_epoch_history["loss"] < self.best_validation_loss:
                    log.info(" ... Best validation loss so far!")
                    self.best_validation_loss = metrics_epoch_history["loss"]
                    self.save_state(suffix="_BEST")

            if self.early_stopping is not None:
                if stop_flag.item():
                    if self.rank == 0:
                        log.info("Early stopping triggered.")
                    if self.is_distributed:
                        torch.distributed.barrier()
                    break

    # ------------------------------------------------------------------ #
    # Checkpointing: save_state / restore_state / restore_best_state are
    # inherited from BaseEngine (which persists epoch + scheduler and logs the
    # wandb artifact). Only checkpoint-path resolution for test() is local.
    # ------------------------------------------------------------------ #
    def _resolve_checkpoint_path(self) -> Path:
        if self.train_output_dir_p is not None:
            ckpt = self.train_output_dir_p / self.CKPT_NAME
        else:
            ckpt = Path(self.dump_path) / self.CKPT_NAME
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    @torch.no_grad()
    def test(
        self,
        diagnostic: Optional[Dict[str, Any]] = None,
        stats_max_events: Optional[int] = None,
        start_at: int = 0,
        n_events_html: int = 30,
        max_voxels: int = 8000,
        poisson_fluctuate_pred: bool = True,
        poisson_seed: int = 0,
    ) -> Dict[str, float]:
        if self.rank != 0:
            return {}

        ckpt_path = self._resolve_checkpoint_path()
        self.restore_state(str(ckpt_path))
        self.model.eval()

        diag = to_container(diagnostic)
        diag.setdefault("out_dir", "img")
        diag.setdefault("stats", {})
        diag["stats"] = dict(diag["stats"] or {})
        diag["stats"].setdefault("max_events", stats_max_events)
        diag["stats"].setdefault("start_at", int(start_at))

        if "enabled" in diag and isinstance(diag["enabled"], (list, tuple)):
            diag["enabled"] = list(diag["enabled"])
        else:
            diag["enabled"] = []

        run_diagnostics = _load_run_diagnostics()
        res = run_diagnostics(
            model=self.model,
            criterion=None,
            val_subset=self.test_subset,
            device=self.device,
            run_dir=Path(os.getcwd()),
            diag_cfg=diag,
        )

        log.info(f"[TEST] Diagnostics wrote to: {res.get('out_dir')}")
        log.info(f"[TEST] Diagnostics ran: {res.get('ran')}")

        summary: Dict[str, float] = {}
        stats = res.get("stats", None)
        if isinstance(stats, dict):
            dm = stats.get("dice_mean", None)
            if isinstance(dm, np.ndarray) and dm.size > 0:
                summary["dice_mean/mean"] = float(np.mean(dm))
                summary["dice_mean/median"] = float(np.median(dm))

        if self.wandb_run is not None and summary:
            self.wandb_run.log({f"test/{k}": v for k, v in summary.items()})

        return summary
