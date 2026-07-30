"""
Base engine and build_loader for reconstruction training.
Supports both PyG DataLoaders and regular torch DataLoaders based on config kind.
"""

# generic imports
import numpy as np
from abc import ABC, abstractmethod

# hydra imports
from hydra.utils import instantiate

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

# wandb is an OPTIONAL dependency (see requirements-tracking.txt). It is imported lazily
# where it is actually used (save_state), so a run with no wandb_run - i.e. CSV-only
# tracking - never needs the package installed.
# torch_geometric is imported lazily in build_loader, see there.

# watchmal imports
from watchmal.dataset.samplers.samplers import DistributedSamplerWrapper
from watchmal.utils.logging_utils import setup_logging
from watchmal.utils.tracking import RunTracker

log = setup_logging(__name__)


def build_loader(
        dataset,
        split_path,
        split_key,
        sampler_config,
        is_pyg,
        seed,
        is_distributed=False,
        batch_size=2,
        num_workers=0,
        sampler_drop_last=True,
        **kwargs
):
    """
    Build a DataLoader (torch or PyG) for a pre-built dataset. sampler_config is required.
    If the dataset has a callable _dataset_collate() method, its return value is
    used as collate_fn unless collate_fn is passed in kwargs (caller override).

    `seed` is the run-wide seed, i.e. the mandatory top-level `seed` key. It drives
    both the sampler's generator and DistributedSamplerWrapper.
    """
    assert sampler_config is not None, "sampler_config is required"

    # A dedicated CPU generator, seeded from the run-wide seed. This is what keeps the
    # ranks in agreement: DistributedSamplerWrapper re-runs this sampler on every
    # set_epoch, so drawing from the global RNG instead would make the ordering depend
    # on how much randomness the rest of the epoch happened to consume - which is not
    # guaranteed to match across ranks. Keeping it on CPU also avoids the
    # generator/device mismatch that torch.randperm raises for a CUDA device.
    generator = torch.Generator()
    generator.manual_seed(seed)

    split_indices = np.load(split_path, allow_pickle=True)[split_key]
    sampler = instantiate(sampler_config, indices=split_indices, generator=generator)

    # Ensure we have at least 1 step
    if split_indices.shape[0] < batch_size:
        batch_size = split_indices.shape[0]

    # Wrap the sampler in case of distributed training
    if ( is_distributed ) and ( split_key not in ['test_idxs']) :
        ngpus = torch.distributed.get_world_size()
        batch_size = max(batch_size // ngpus, 1) # If using mp, ensure that the batch size is at least 1 per GPU.
        # drop_last=True for training (equal step counts across ranks); False for
        # validation so no val event is silently excluded from the metric.
        sampler = DistributedSamplerWrapper(sampler=sampler, seed=seed, drop_last=sampler_drop_last)

    # Use dataset-provided collate if present (e.g. _dataset_collate()), unless caller passed collate_fn in kwargs
    collate_fn = dataset._dataset_collate() if hasattr(dataset, "_dataset_collate") and callable(getattr(dataset, "_dataset_collate")) else None

    # Handle persistent_workers: must be False if num_workers == 0, otherwise can be True
    # Pop from kwargs to avoid override, then set correctly based on num_workers
    persistent_workers = kwargs.pop("persistent_workers", None)
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)
    elif num_workers == 0 and persistent_workers:
        # Can't have persistent_workers=True when num_workers=0
        persistent_workers = False

    if is_pyg:
        # Imported here rather than at module scope: this is the only torch_geometric
        # dependency the engine has, and the graph datasets and models reach PyG through
        # their own hydra _target_. Keeping it local means a run that never touches a
        # graph dataset - multi-ring, image - can work without torch_geometric installed.
        from torch_geometric.loader import DataLoader as PyGDataLoader

        return PyGDataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    else:
        return DataLoader(
            dataset,
            sampler=sampler,
            persistent_workers=persistent_workers,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )


class BaseEngine(ABC):
    """
    Base engine for reconstruction training. Holds common state, configuration,
    distributed helpers, and checkpointing. Subclasses (graph-like, image-like)
    implement their own forward/batch logic and train/validate/evaluate loops.
    Supports both PyG DataLoaders and regular torch DataLoaders based on config kind.
    """

    def __init__(
        self,
        target_key,
        model,
        rank,
        device,
        dump_path,
        wandb_run=None,
        dataset=None,
        logging_csv=True,
    ):
        """
        Parameters
        ==========
        target_key : str
            Name of the key for the target values in the dictionary returned by the dataloader.
        model : nn.Module
            Full network used for training or evaluation.
        rank : int
            Rank of this process (in multiprocessing mode).
        device : torch.device or int
            Device to run on.
        dump_path : str
            Path to store outputs (should end with '/').
        wandb_run : optional
            If set, subclasses may use it for logging (e.g. save_state can log artifacts).
        dataset : optional
            Pre-built dataset instance (for graph-style pipeline). If None, set later via set_dataset or via config-based configure_data_loaders.
        logging_csv : bool
            Whether to write the analysis-compatible CSV logs (default True). CSV stays an
            option; wandb is enabled independently by whether `wandb_run` is set.
        """
        self.dump_path = dump_path
        self.wandb_run = wandb_run
        self.rank = rank
        # Unified tracker: analysis-compatible CSV (default on) + optional wandb. Every
        # engine logs through this, so all model families share the same tracking options.
        self.tracker = RunTracker(
            dump_path=dump_path, rank=rank, wandb_run=wandb_run, csv_enabled=logging_csv
        )
        self.device = torch.device(device)
        self.model = model
        self.target_key = target_key

        self.epoch = 0
        self.iteration = 0
        self.best_validation_loss = np.inf
        self.best_training_loss = np.inf

        self.dataset = dataset
        self.is_pyg = None
        self.split_path = None
        self.target_names = []

        # Run-wide seed. main.py requires a top-level `seed` and calls
        # torch.manual_seed() with it before building the engine, so this reads back
        # that exact value on every rank. It is the single source of randomness for
        # the loaders and for any dataset that needs to split deterministically.
        self.seed = torch.initial_seed()

        self.data_loaders = {}

        if isinstance(self.model, DistributedDataParallel):
            self.is_distributed = True
            self.module = self.model.module
            self.n_gpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.module = self.model
            self.n_gpus = 1

        self.data = None
        self.target = None
        self.loss = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # Automatic Mixed Precision (used by the CNN reconstruction loop via
        # configure_amp(); the graph / multi-ring loops simply never enable it).
        self.use_amp = False
        self.scaler = None

        # Optional early stopping (graph / multi-ring loops use it; the CNN loop ignores
        # it). Held on the base so the single worker can configure any engine uniformly.
        self.early_stopping = None

    def configure_early_stopping(self, early_stopping_config):
        """Instantiate an early-stopping helper from a hydra config."""
        self.early_stopping = instantiate(early_stopping_config)

    def configure_amp(self, amp_enabled: bool = False):
        """Configure automatic mixed precision (AMP). No-op unless on CUDA.

        The GradScaler import is deliberately inside the enabled branch: the worker
        calls this method on every run of every family, and `torch.amp.GradScaler` only
        exists in recent torch releases, so importing it eagerly would make an
        AMP-disabled run fail on older cluster containers for no reason. The fallback
        covers those containers when AMP *is* requested.
        """
        self.use_amp = bool(amp_enabled) and (self.device.type == "cuda")
        if self.use_amp:
            try:
                from torch.amp import GradScaler

                self.scaler = GradScaler("cuda")
            except ImportError:  # older torch: no torch.amp.GradScaler
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()
        if self.rank == 0:
            log.info(f"AMP enabled: {self.use_amp}")

    def setup_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Uniform data-loader entry point called by the worker for every engine family.

        Default (graph / multi-ring) path: build the dataset then the loaders in two
        steps. The CNN engine overrides this method with its own single-step
        get_data_loader path. Keeping one entry-point signature is what lets the single
        worker configure any engine without knowing its family; the two dataset pipelines
        stay separate underneath.
        """
        self.configure_dataset(data_config)
        self.configure_data_loaders(loaders_config)

    def configure_loss(self, loss_config):
        """Instantiate loss from a hydra config."""
        self.criterion = instantiate(loss_config)

    def configure_optimizers(self, optimizer_config):
        """Instantiate optimizer from a hydra config."""
        self.optimizer = instantiate(optimizer_config, params=self.module.parameters())
        total_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        opt_params = sum(p.numel() for g in self.optimizer.param_groups for p in g['params'])
        log.info(f"Total trainable parameters: {total_params}")
        log.info(f"Parameters passed to optimizer: {opt_params}")

    def configure_scheduler(self, scheduler_config):
        """Instantiate scheduler from a hydra config."""
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

    def set_dataset(self, dataset, dataset_config):
        """Set the dataset and split path (e.g. after building dataset outside the engine)."""
        if self.dataset is not None:
            raise ValueError(f"Dataset is already set in the engine (rank {self.rank}).")
        self.dataset = dataset
        self.split_path = dataset_config.split_path
        self.is_pyg = ("pyg" in dataset_config.kind)
        self.target_names = list(dataset_config.target_names)
        log.info(f"dataset: {dataset[0]}")

    def configure_dataset(self, data_config):
        """
        Configure the dataset from data_config. Graph / multi-ring engines override this
        to set self.dataset / self.split_path / self.is_pyg / self.target_names. The CNN
        engine builds its loaders directly in setup_data_loaders and never calls this, so
        the default just signals misuse rather than being abstract (which would force a
        stub on the CNN engine).

        Parameters
        ----------
        data_config : DictConfig or dict
            Configuration containing dataset parameters, split_path, etc.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement configure_dataset(); its data "
            f"loaders are built in setup_data_loaders()."
        )

    def configure_data_loaders(self, loaders_config):
        """
        Set up data loaders from loaders_config using the configured dataset and split_path.
        Uses build_loader (torch or PyG DataLoader based on config kind).
        Each loader_config must have "split_key" and "sampler_config".
        
        configure_dataset() must be called before this method.
        
        Parameters
        ----------
        loaders_config : dict
            Dictionary of loader configs (e.g., {'train': {...}, 'validation': {...}}).
        """
        # Ensure dataset has been configured
        assert self.dataset is not None, "Dataset must be configured. Call configure_dataset() first."
        assert self.split_path is not None and self.split_path != "", "Split path must be set. Call configure_dataset() first."
        assert self.is_pyg is not None, "is_pyg must be set. Call configure_dataset() first."
        
        for name, loader_config in loaders_config.items():
            log.info(f"Building data loader {name}..")
            if "seed" in loader_config:
                log.warning(
                    f"Data loader '{name}' still sets a per-loader `seed`; it is ignored. "
                    f"Randomness now comes from the top-level `seed` ({self.seed}). "
                    f"Remove the key from the config."
                )
            self.data_loaders[name] = build_loader(
                dataset=self.dataset,
                split_path=self.split_path,
                is_pyg=self.is_pyg,
                split_key=loader_config["split_key"],
                sampler_config=loader_config["sampler_config"],
                seed=self.seed,
                is_distributed=self.is_distributed,
                batch_size=loader_config.get("batch_size", 2),
                num_workers=loader_config.get("num_workers", 0),
                # Only the training split drops its tail (equal step counts across ranks);
                # validation pads instead so every val event is included in the metric.
                sampler_drop_last=(name == "train"),
                **{k: v for k, v in loader_config.items() if k not in ("split_key", "sampler_config", "seed", "batch_size", "num_workers", "is_graph")},
            )
            log.info(f"Data loader {name} built.")

    def get_reduced(self, outputs, op=torch.distributed.ReduceOp.SUM):
        """
        Reduce tensors from all processes to rank 0 (sum then divide by n_gpus on rank 0).
        Only rank 0's returned dict is populated; other ranks get an empty dict.
        Returns dict of tensors (no .item()); caller may convert to scalars if needed.
        """
        new_outputs = {}
        for name, tensor in outputs.items():
            torch.distributed.reduce(tensor, 0, op=op)
            if self.rank == 0:
                new_outputs[name] = tensor / self.n_gpus
        return new_outputs

    def get_gathered(self, tensor):
        """
        Gather tensor from all processes to rank 0 and concatenate along dim=0.
        Rank 0 returns the concatenated tensor; other ranks return their local tensor.
        """
        if getattr(self, "is_distributed", False) and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)] if rank == 0 else None
            torch.distributed.gather(tensor, gather_list=gather_list, dst=0)
            if rank == 0:
                return torch.cat(gather_list, dim=0)
            return tensor
        return tensor

    # ------------------------------------------------------------------ #
    # CNN-family DDP adapters (same collectives as get_reduced/get_gathered,
    # but with the original CNN-core return types: python floats / numpy). They
    # live on the shared base so every engine family reduces/gathers through one
    # definition site. get_reduced/get_gathered keep tensors (graph/MR loops call
    # them only when distributed); get_synchronized_* handle the non-distributed
    # case internally and are what the CNN reconstruction loop uses.
    # ------------------------------------------------------------------ #
    def get_synchronized_outputs(self, output_dict):
        """
        Gather per-process output tensors to rank 0 (concatenated) and return numpy arrays.
        Non-distributed: just detach/cpu/numpy each tensor.
        """
        global_output_dict = {}
        for name, tensor in output_dict.items():
            if self.is_distributed:
                if self.rank == 0:
                    tensor_list = [torch.zeros_like(tensor, device=self.device) for _ in range(self.n_gpus)]
                    torch.distributed.gather(tensor, tensor_list)
                    global_output_dict[name] = torch.cat(tensor_list).detach().cpu().numpy()
                else:
                    torch.distributed.gather(tensor, dst=0)
            else:
                global_output_dict[name] = tensor.detach().cpu().numpy()
        return global_output_dict

    def get_synchronized_metrics(self, metric_dict):
        """
        Reduce (sum then divide by n_gpus) per-process metric tensors to rank 0 and return
        python floats. Non-distributed: just .item() each tensor.
        """
        global_metric_dict = {}
        for name, tensor in zip(metric_dict.keys(), metric_dict.values()):
            if self.is_distributed:
                torch.distributed.reduce(tensor, 0)
                if self.rank == 0:
                    global_metric_dict[name] = tensor.item() / self.n_gpus
            else:
                global_metric_dict[name] = tensor.item()
        return global_metric_dict

    def backward(self):
        """Backward pass using the loss computed for the current mini-batch."""
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def save_state(self, suffix="", name=None):
        """
        Save model weights and training state to a file, and (if a wandb run is
        active on rank 0) log the checkpoint as a wandb artifact.

        suffix: e.g. "_BEST" for best validation state.
        name: filename base; default is {EngineClass}_{ModelClass}.
        """
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        filename = f"{self.dump_path}{name}{suffix}.pth"
        torch.save(
            {
                "global_step": self.iteration,
                "epoch": self.epoch,
                # The un-offset run seed, so a checkpoint carries the randomness that
                # produced it and a later run can be checked against it.
                "seed": self.seed,
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else {},
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else {},
                "state_dict": self.module.state_dict(),
            },
            filename,
        )
        log.info(f"Saved state as: {filename}")

        if self.wandb_run is not None and self.rank == 0 and suffix != "_LASTGOOD":
            import wandb  # optional dependency; only needed when a wandb run is active
            artifact = wandb.Artifact(name=f"model-and-opti-checkpoints-{self.wandb_run.id}", type="model-and-opti")
            artifact.add_file(filename)
            artifact.metadata["checkpoints_dir"] = filename
            aliases = ["ite_" + str(self.iteration)]
            if suffix:
                aliases.append(suffix)
            if suffix == "_BEST":
                artifact.description = f"Validation loss : {self.best_validation_loss:.4g}"
                self.wandb_run.log({"best_val_epoch_loss": self.best_validation_loss})
            self.wandb_run.log_artifact(artifact, aliases=aliases)
            log.info("Save state on wandb")

        return filename

    def restore_state(self, weight_file):
        """Restore model and (when present) optimizer/scheduler/epoch from a file."""
        with open(weight_file, "rb") as f:
            log.info(f"Restoring state from {weight_file}")
            if self.is_distributed:
                torch.distributed.barrier()
            checkpoint = torch.load(f, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            self.module.load_state_dict(state_dict)
            if isinstance(checkpoint, dict):
                if self.optimizer is not None and checkpoint.get("optimizer"):
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if self.scheduler is not None and checkpoint.get("scheduler"):
                    try:
                        self.scheduler.load_state_dict(checkpoint["scheduler"])
                    except Exception as e:
                        log.warning(f"Could not restore scheduler state: {e}")
                self.iteration = checkpoint.get("global_step", self.iteration)
                self.epoch = checkpoint.get("epoch", self.epoch)
                # Report, never override: the config stays the single source of truth for
                # the seed, so a mismatch is surfaced rather than silently corrected.
                ckpt_seed = checkpoint.get("seed")
                if ckpt_seed is None:
                    log.info("Checkpoint predates seed storage; its seed is unknown.")
                elif ckpt_seed != self.seed:
                    log.warning(
                        f"Checkpoint was produced with seed {ckpt_seed}, this run uses "
                        f"{self.seed}. Set `seed: {ckpt_seed}` in the config to reproduce "
                        f"the original stream."
                    )

    def restore_best_state(self, name=None, complete_path=False):
        """
        Restore model from the best checkpoint.
        complete_path=True treats `name` as a full path; otherwise the path is
        built as {dump_path}{name}_BEST.pth.
        """
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        full_path = name if complete_path else f"{self.dump_path}{name}_BEST.pth"
        self.restore_state(full_path)
