
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

from ultralytics.nn.modules.block import Bottleneck, PSABlock

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (torch.amp.GradScaler): Gradient scaler for AMP.
        data (dict): Dataset dictionary containing paths and metadata.
        ema (ModelEMA): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (callable): Learning rate scheduling function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (torch.Tensor): Current loss value.
        tloss (torch.Tensor): Running mean of loss items.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the val set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the BaseTrainer class.

        Args:
            cfg (str | dict | SimpleNamespace, optional): Path to a configuration file or configuration object.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            # Save run args, serializing augmentations as reprs for resume compatibility
            args_dict = vars(self.args).copy()
            if args_dict.get("augmentations") is not None:
                # Serialize Albumentations transforms as their repr strings for checkpoint compatibility
                args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "args.yaml", args_dict)  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Callbacks - initialize early so on_pretrain_routine_start can capture original args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Execute the training process, using DDP subprocess for multi-GPU or direct training for single-GPU."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _build_train_pipeline(self):
        """Build dataloaders, optimizer, and scheduler for current batch size."""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # ============================= Chuẩn bị Sparsity Training ==========================
        # Giữ giá trị sr đã được gán từ model.py, fallback 0.0 nếu chưa set
        if not hasattr(self, 'sr') or self.sr is None:
            self.sr = 0.0
        self.ignore_bn_list = []

        if self.sr > 0:
            LOGGER.info(f"Sparsity Training (sr={self.sr}) is ENABLED. Disabling AMP...")
            self.args.amp = False  # Bắt buộc tắt AMP khi ép trọng số L1

            for k, m in unwrap_model(self.model).named_modules():
                # 1. Xử lý Bottleneck
                if isinstance(m, Bottleneck):
                    if m.add:
                        self.ignore_bn_list.append(k + '.cv2.bn')
                        if len(k.split('.')) >= 2 and k.split('.')[-2] == 'm':
                            parent_name = k.rsplit(".", 2)[0]
                            self.ignore_bn_list.append(parent_name + ".cv1.bn")

                # 2. Xử lý PSABlock
                elif isinstance(m, PSABlock):
                    for sub_k, sub_m in m.named_modules():
                        if isinstance(sub_m, nn.BatchNorm2d):
                            self.ignore_bn_list.append(f"{k}.{sub_k}")
                    parts = k.split('.')
                    if len(parts) >= 2:
                        layer_idx = parts[1]
                        self.ignore_bn_list.append(f"model.{layer_idx}.cv1.bn")

            self.ignore_bn_list = list(set(self.ignore_bn_list))
            LOGGER.info(f"[Sparsity] Locked {len(self.ignore_bn_list)} BN layers.")
        # ============================= Chuẩn bị Sparsity Training ==========================

        # ============================= DMS (Differentiable Model Scaling) ==========================
        self.dms_enabled = getattr(self, 'dms', False)
        self.a_params = {}
        self.dms_hooks = []
        self.dms_importance = getattr(self, 'dms_importance', 'gamma')
        self.dms_divisor = getattr(self, 'dms_divisor', 8)
        self.taylor_buffers = {}

        if self.dms_enabled:
            from dms.dms_utils import (
                build_ignore_bn_list, profile_per_layer_flops,
                build_conv_bn_mapping, make_soft_mask_hook,
            )

            LOGGER.info(
                f"DMS Search ENABLED: target={self.dms_target}, "
                f"lambda={self.dms_lambda}, freeze={self.dms_freeze}, "
                f"importance={self.dms_importance}"
            )

            # Giữ AMP cho DMS (paper gốc dùng AMP)
            # Scaler chỉ dùng cho main optimizer, DMS optimizer xử lý riêng

            # Build ignore list nếu chưa có (sparsity đã build nếu sr > 0)
            if not self.ignore_bn_list:
                self.ignore_bn_list = build_ignore_bn_list(unwrap_model(self.model))
                LOGGER.info(f"[DMS] Built ignore_bn_list: {len(self.ignore_bn_list)} BN layers locked.")

            # Collect prunable BNs
            prunable_bns = {}
            for name, m in unwrap_model(self.model).named_modules():
                if isinstance(m, nn.BatchNorm2d) and name not in self.ignore_bn_list:
                    prunable_bns[name] = m

            # Create learnable a params (init = 0 → mask ≈ 1, resource loss drives a toward target)
            for name, m in prunable_bns.items():
                a = nn.Parameter(torch.tensor(0.0, device=self.device))
                self.a_params[name] = a
                # Init taylor buffer (zeros → fallback to gamma until populated)
                if self.dms_importance == 'taylor':
                    self.taylor_buffers[name] = torch.zeros(m.num_features, device=self.device)

            # Profile per-layer FLOPs
            imgsz = self.args.imgsz
            self.conv_flops, self.total_flops = profile_per_layer_flops(
                unwrap_model(self.model), imgsz=imgsz, device=self.device
            )
            self.conv_bn_map, self.bn_channels = build_conv_bn_mapping(
                unwrap_model(self.model), self.ignore_bn_list
            )

            # Build BN→Conv mapping for L1 norm importance
            self.bn_to_conv = {}
            if self.dms_importance == 'l1':
                modules_dict = dict(unwrap_model(self.model).named_modules())
                for bn_name in prunable_bns:
                    conv_name = bn_name[:-2] + 'conv'  # model.X.cv1.bn → model.X.cv1.conv
                    if conv_name in modules_dict and isinstance(modules_dict[conv_name], nn.Conv2d):
                        self.bn_to_conv[bn_name] = modules_dict[conv_name]
                    else:
                        LOGGER.warning(f"[DMS] Conv not found for {bn_name}, fallback to gamma")
                LOGGER.info(f"[DMS] L1 norm importance: mapped {len(self.bn_to_conv)}/{len(prunable_bns)} BN→Conv pairs")

            # Register forward hooks on prunable BNs
            for name, m in prunable_bns.items():
                hook = m.register_forward_hook(make_soft_mask_hook(
                    name, self.a_params,
                    importance=self.dms_importance,
                    taylor_buffers=self.taylor_buffers if self.dms_importance == 'taylor' else None,
                    conv_module=self.bn_to_conv.get(name) if self.dms_importance == 'l1' else None,
                ))
                self.dms_hooks.append(hook)

            # NOTE: add_param_group deferred to after _build_train_pipeline() creates optimizer

            # Freeze model weights if mode freeze
            if self.dms_freeze:
                LOGGER.info("[DMS] Freeze mode: model weights frozen, only training a params.")
                for p in unwrap_model(self.model).parameters():
                    p.requires_grad = False

            LOGGER.info(
                f"[DMS] {len(self.a_params)} learnable a params, "
                f"{self.total_flops / 1e9:.2f} GFLOPs original"
            )
        # ============================= DMS (Differentiable Model Scaling) ==========================

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                if getattr(self, 'dms_freeze', False):
                    pass  # DMS freeze mode: keep model weights frozen
                else:
                    LOGGER.warning(
                        f"setting 'requires_grad=True' for frozen layer '{k}'. "
                        "See ultralytics.engine.trainer for customization of frozen layers."
                    )
                    v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        self._build_train_pipeline()

        # ============================= DMS: separate optimizer for a_params ==========================
        # Use separate optimizer to avoid scheduler mismatch (scheduler tracks model optimizer only)
        if self.dms_enabled and self.a_params:
            dms_lr = getattr(self, 'dms_lr', 5e-3)
            self.dms_optimizer = torch.optim.Adam(
                list(self.a_params.values()), lr=dms_lr
            )
            LOGGER.info(f"[DMS] Created separate Adam optimizer for {len(self.a_params)} a_params (lr={dms_lr}).")
        # ============================= DMS: separate optimizer for a_params ==========================

        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

        # ============================= CWD (Channel-Wise Distillation) ==========================
        self.kd_enabled = getattr(self, 'kd', False)
        if self.kd_enabled:
            import math as _math
            from distillation.cwd_loss import CWDLoss, setup_hooks, build_kd_channel_masks
            from ultralytics.nn.autobackend import AutoBackend

            teacher_path = getattr(self, 'kd_teacher', None)
            assert teacher_path, "kd=True requires kd_teacher='path/to/teacher.pt'"

            # Load + freeze teacher
            self.teacher_model = AutoBackend(teacher_path, fuse=False)
            self.teacher_model.eval().to(self.device)
            for p in self.teacher_model.parameters():
                p.requires_grad = False

            # Distill layer indices
            layers_cfg = getattr(self, 'kd_layers', 'neck')
            if layers_cfg == "neck":
                layer_indices = [13, 16, 19, 22]
            elif layers_cfg == "backbone":
                layer_indices = [2, 4, 6, 8]
            else:  # "all"
                layer_indices = [2, 4, 6, 8, 13, 16, 19, 22]

            self.kd_layer_names = [f"model.{i}" for i in layer_indices]

            # Setup passive hooks on student and teacher
            self.student_hooks = setup_hooks(unwrap_model(self.model), self.kd_layer_names)
            self.teacher_hooks = setup_hooks(self.teacher_model.model, self.kd_layer_names)

            # Build channel masks from maskbndict (if student is pruned)
            maskbndict = getattr(self, 'kd_maskbndict', None)
            if maskbndict is not None:
                self.kd_channel_masks = build_kd_channel_masks(maskbndict, layer_indices)
                LOGGER.info(f"[KD] Channel masks: {len(self.kd_channel_masks)} layers have mismatch")
            else:
                self.kd_channel_masks = {}

            # Temperature: float=fixed, "learnable"=auto
            temp_cfg = getattr(self, 'cwd_temperature', 9.0)
            if isinstance(temp_cfg, str) and temp_cfg == "learnable":
                import math as _math
                self.cwd_temp_mode = "learnable"
                tau_init = getattr(self, 'cwd_learnable_tau_init', 9.0)
                self.cwd_log_tau = nn.Parameter(
                    torch.tensor(_math.log(tau_init), device=self.device)
                )
                tau_lr = getattr(self, 'cwd_learnable_tau_lr', 1e-3)
                self.cwd_tau_optimizer = torch.optim.Adam([self.cwd_log_tau], lr=tau_lr)
                LOGGER.info(f"[CWD] Learnable tau: init={tau_init:.1f}, lr={tau_lr}")
            else:
                self.cwd_temp_mode = "fixed"
                self.cwd_temp_value = float(temp_cfg)

            self._kd_lambda = getattr(self, 'kd_lambda', 0.5)

            # KD method selection: cwd (default), response, fitnets, mgd
            self._kd_method = getattr(self, 'kd_method', 'cwd')

            if self._kd_method == "cwd":
                self.kd_criterion = CWDLoss()
            elif self._kd_method == "response":
                from distillation.kd_losses import ResponseKDLoss
                self.kd_criterion = ResponseKDLoss()
            elif self._kd_method == "fitnets":
                from distillation.kd_losses import FitNetsLoss
                normalize = getattr(self, 'fitnets_normalize', True)
                self.kd_criterion = FitNetsLoss(normalize=normalize)
            elif self._kd_method == "mgd":
                from distillation.kd_losses import MGDLoss
                mgd_mask_ratio = getattr(self, 'mgd_mask_ratio', 0.5)
                self.kd_criterion = MGDLoss(mask_ratio=mgd_mask_ratio)

                # MGD cần biết channel sizes → chạy dummy forward để lấy
                LOGGER.info("[MGD] Detecting channel sizes for generators...")
                dummy = torch.zeros(1, 3, self.args.imgsz, self.args.imgsz, device=self.device)
                with torch.no_grad():
                    unwrap_model(self.model)(dummy)
                    self.teacher_model(dummy)

                for name in self.kd_layer_names:
                    s_ch = self.student_hooks[name].features.shape[1]
                    t_ch = self.teacher_hooks[name].features.shape[1]
                    self.kd_criterion.add_generator(name, s_ch, t_ch)
                    LOGGER.info(f"[MGD] {name}: student={s_ch}ch → teacher={t_ch}ch")

                # Move generators lên device, thêm vào optimizer
                self.kd_criterion.to(self.device)
                self._mgd_optimizer = torch.optim.Adam(
                    self.kd_criterion.parameters(), lr=1e-3
                )
            else:
                raise ValueError(f"Unknown kd_method: {self._kd_method}. "
                                 f"Choose from: cwd, response, fitnets, mgd")

            LOGGER.info(f"[KD] ENABLED: method={self._kd_method}, teacher={teacher_path}, lambda={self._kd_lambda}")
            temp_info = f"learnable (init={getattr(self, 'cwd_learnable_tau_init', 9.0)})" if self.cwd_temp_mode == "learnable" else self.cwd_temp_mode
            LOGGER.info(f"[KD] layers={self.kd_layer_names}, temp={temp_info}")
            if getattr(self.args, 'resume', False):
                LOGGER.info("[KD] Resuming KD training...")
        # ============================= CWD (Channel-Wise Distillation) ==========================

    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        elif self.dms_enabled:
                            # ====== DMS NaN debug: test forward KHÔNG có DMS hooks ======
                            if i == 0 and not getattr(self, '_dms_debug_done', False):
                                self._dms_debug_done = True
                                # Test 1: forward KHÔNG hooks
                                for h in self.dms_hooks:
                                    h.remove()
                                with torch.no_grad():
                                    _loss_no_hook, _ = self.model(batch)
                                _vals = [l.item() for l in _loss_no_hook]
                                LOGGER.info(f"[DMS-DEBUG] Forward WITHOUT hooks: loss={_vals}")
                                # Đăng ký lại hooks
                                self.dms_hooks.clear()
                                from dms.dms_utils import make_soft_mask_hook
                                for name, m in unwrap_model(self.model).named_modules():
                                    if isinstance(m, nn.BatchNorm2d) and name in self.a_params:
                                        hook = m.register_forward_hook(make_soft_mask_hook(
                                            name, self.a_params,
                                            importance=self.dms_importance,
                                            taylor_buffers=self.taylor_buffers if self.dms_importance == 'taylor' else None,
                                            conv_module=self.bn_to_conv.get(name) if self.dms_importance == 'l1' else None,
                                        ))
                                        self.dms_hooks.append(hook)
                                # Test 2: forward CÓ hooks
                                with torch.no_grad():
                                    _loss_hook, _ = self.model(batch)
                                _vals2 = [l.item() for l in _loss_hook]
                                LOGGER.info(f"[DMS-DEBUG] Forward WITH hooks:    loss={_vals2}")
                                # Test 3: check BN weights cho NaN
                                _nan_bns = []
                                for name, m in unwrap_model(self.model).named_modules():
                                    if isinstance(m, nn.BatchNorm2d):
                                        if m.weight.isnan().any():
                                            _nan_bns.append(f"{name}.weight")
                                        if m.bias.isnan().any():
                                            _nan_bns.append(f"{name}.bias")
                                        if m.running_mean is not None and m.running_mean.isnan().any():
                                            _nan_bns.append(f"{name}.running_mean")
                                        if m.running_var is not None and m.running_var.isnan().any():
                                            _nan_bns.append(f"{name}.running_var")
                                if _nan_bns:
                                    LOGGER.error(f"[DMS-DEBUG] NaN in BN params: {_nan_bns}")
                                else:
                                    LOGGER.info(f"[DMS-DEBUG] All BN params are finite")
                                # Test 4: check tất cả model weights
                                _nan_params = []
                                for name, p in unwrap_model(self.model).named_parameters():
                                    if p.isnan().any() or p.isinf().any():
                                        _nan_params.append(name)
                                if _nan_params:
                                    LOGGER.error(f"[DMS-DEBUG] NaN/Inf weights: {_nan_params}")
                                else:
                                    LOGGER.info(f"[DMS-DEBUG] All model weights are finite")
                            # ====== end debug ======

                            loss, self.loss_items = self.model(batch)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size

                        # ============================= DMS loss ==========================
                        if self.dms_enabled:
                            dms_warmup = getattr(self, 'dms_warmup', 0)
                            if epoch >= dms_warmup:
                                from dms.dms_utils import compute_resource_loss
                                loss_resource = compute_resource_loss(
                                    self.a_params, self.conv_flops, self.conv_bn_map,
                                    self.bn_channels, self.total_flops, self.dms_target,
                                )
                                # Ramp up resource loss over 5 epochs after warmup
                                dms_ramp = min((epoch - dms_warmup) / 5.0, 1.0)
                                self.loss = self.loss + dms_ramp * self.dms_lambda * loss_resource
                        # ============================= DMS loss ==========================

                        self.tloss = (
                            self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                        )

                    # ============================= KD loss (CWD/Response/FitNets/MGD) ==========================
                    if getattr(self, 'kd_enabled', False):
                        import math as _math

                        kd_warmup = getattr(self, 'kd_warmup', 5)
                        if epoch >= kd_warmup:
                            with torch.no_grad():
                                self.teacher_model(batch["img"])

                            # Temperature (dùng cho CWD và Response KD)
                            if self.cwd_temp_mode == "learnable":
                                tau = self.cwd_log_tau.exp().clamp(0.5, 20.0)
                            else:
                                tau = self.cwd_temp_value

                            # Compute KD loss under autocast to match AMP dtypes
                            with autocast(self.amp):
                                # Dispatch theo kd_method
                                kd_method = getattr(self, '_kd_method', 'cwd')
                                if kd_method == "cwd":
                                    from distillation.cwd_loss import compute_cwd_loss
                                    kd_loss_val = compute_cwd_loss(
                                        self.student_hooks, self.teacher_hooks,
                                        self.kd_criterion, self.kd_channel_masks,
                                        temperature=tau,
                                    )
                                elif kd_method == "response":
                                    from distillation.kd_losses import compute_response_kd_loss
                                    kd_loss_val = compute_response_kd_loss(
                                        self.student_hooks, self.teacher_hooks,
                                        self.kd_criterion, self.kd_channel_masks,
                                        temperature=tau,
                                    )
                                elif kd_method == "fitnets":
                                    from distillation.kd_losses import compute_fitnets_loss
                                    kd_loss_val = compute_fitnets_loss(
                                        self.student_hooks, self.teacher_hooks,
                                        self.kd_criterion, self.kd_channel_masks,
                                    )
                                elif kd_method == "mgd":
                                    from distillation.kd_losses import compute_mgd_loss
                                    kd_loss_val = compute_mgd_loss(
                                        self.student_hooks, self.teacher_hooks,
                                        self.kd_criterion, self.kd_channel_masks,
                                    )

                            # Ramp up lambda linearly over first 5 epochs after warmup
                            kd_ramp = min((epoch - kd_warmup) / 5.0, 1.0)
                            self.loss = self.loss + kd_ramp * self._kd_lambda * kd_loss_val
                        elif epoch == kd_warmup - 1 and ni == 0:
                            LOGGER.info(f"[KD] Warmup: {getattr(self, '_kd_method', 'cwd')} will start at epoch {kd_warmup}")
                    # ============================= KD loss ==========================

                    # Backward
                    # ============================= disable scaler ==========================
                    if getattr(self, 'sr', 0.0) > 0:
                        self.loss.backward()
                    elif self.dms_enabled:
                        if getattr(self, 'dms_freeze', False):
                            # Freeze: model không có grad, không cần scaler
                            if self.loss.isfinite():
                                self.loss.backward()
                        else:
                            # Non-freeze: dùng scaler cho model grads
                            self.scaler.scale(self.loss).backward()
                    else:
                        self.scaler.scale(self.loss).backward()

                    # ============================= sparsity training ==========================
                    if getattr(self, 'sr', 0.0) > 0:
                        srtmp = self.sr * (1 - 0.9 * self.epoch / self.epochs)
                        for k, m in unwrap_model(self.model).named_modules():
                            if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):
                                if m.weight.grad is None:
                                    if i == 0:  # only log once per epoch
                                        LOGGER.warning(
                                            f"[SR] BN '{k}' has weight.grad=None "
                                            f"(requires_grad={m.weight.requires_grad}), skipping"
                                        )
                                    continue
                                m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))
                    # ============================= sparsity training ==========================
                except torch.cuda.OutOfMemoryError:
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"CUDA out of memory with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    self._clear_memory()
                    self._build_train_pipeline()  # rebuild dataloaders, optimizer, scheduler
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # restart epoch loop with reduced batch size
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch

            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            # ============================= SR epoch logging ==========================
            if getattr(self, 'sr', 0.0) > 0 and RANK in {-1, 0}:
                srtmp = self.sr * (1 - 0.9 * epoch / self.epochs)
                gammas = []
                for k, m in unwrap_model(self.model).named_modules():
                    if isinstance(m, nn.BatchNorm2d) and k not in self.ignore_bn_list:
                        gammas.append(m.weight.data.abs().cpu())
                if gammas:
                    all_gamma = torch.cat(gammas)
                    sparsity = (all_gamma < 0.01).float().mean().item() * 100
                    gamma_mean = all_gamma.mean().item()
                    gamma_std = all_gamma.std().item()
                    self.sr_metrics = {"sr/sparsity": sparsity, "sr/gamma_mean": gamma_mean, "sr/gamma_std": gamma_std}
                    LOGGER.info(f"[SR] Epoch {epoch}: sr_tmp={srtmp:.6f}, sparsity={sparsity:.1f}%, gamma_mean={gamma_mean:.4f}, gamma_std={gamma_std:.4f}")
            # ============================= SR epoch logging ==========================

            # ============================= DMS epoch logging ==========================
            if getattr(self, 'dms_enabled', False) and self.a_params and RANK in {-1, 0}:
                dms_warmup = getattr(self, 'dms_warmup', 0)
                avg_a = sum(a.item() for a in self.a_params.values()) / len(self.a_params)
                min_a = min(a.item() for a in self.a_params.values())
                max_a = max(a.item() for a in self.a_params.values())
                if epoch < dms_warmup:
                    LOGGER.info(
                        f"[DMS] Epoch {epoch}: WARMUP ({epoch+1}/{dms_warmup}), "
                        f"a params frozen"
                    )
                else:
                    dms_ramp = min((epoch - dms_warmup) / 5.0, 1.0)
                    LOGGER.info(
                        f"[DMS] Epoch {epoch}: avg_a={avg_a:.4f}, "
                        f"min={min_a:.4f}, max={max_a:.4f}"
                        f"{f', ramp={dms_ramp:.2f}' if dms_ramp < 1.0 else ''}"
                    )
            # ============================= DMS epoch logging ==========================

            # ============================= CWD epoch logging ==========================
            if getattr(self, 'kd_enabled', False) and RANK in {-1, 0}:
                if self.cwd_temp_mode == "learnable":
                    tau_val = self.cwd_log_tau.exp().clamp(0.5, 20.0).item()
                    LOGGER.info(f"[CWD] Epoch {epoch}: tau={tau_val:.4f} (learnable, log_tau={self.cwd_log_tau.item():.4f})")
                else:
                    LOGGER.info(f"[CWD] Epoch {epoch}: tau={self.cwd_temp_value:.2f}")
            # ============================= CWD epoch logging ==========================

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                sr_m = getattr(self, 'sr_metrics', {})
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr, **sr_m})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        ckpt_dict = {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
        }

        # ============================= Custom args: save for resume ==========================
        ckpt_dict["custom_training_args"] = {
            "sr": getattr(self, 'sr', None),
            "dms": getattr(self, 'dms_enabled', False),
            "dms_target": getattr(self, 'dms_target', 0.3),
            "dms_lambda": getattr(self, 'dms_lambda', 1.0),
            "dms_lr": getattr(self, 'dms_lr', 5e-3),
            "dms_freeze": getattr(self, 'dms_freeze', False),
            "dms_importance": getattr(self, 'dms_importance', 'gamma'),
            "dms_warmup": getattr(self, 'dms_warmup', 0),
            "finetune": getattr(self, 'finetune', False),
            "kd": getattr(self, 'kd_enabled', False),
            "kd_teacher": getattr(self, 'kd_teacher', None),
            "kd_lambda": getattr(self, '_kd_lambda', 0.5),
            "cwd_temperature": getattr(self, 'cwd_temperature', 9.0),
            "kd_layers": getattr(self, 'kd_layers', 'neck'),
            "kd_warmup": getattr(self, 'kd_warmup', 5),
            "cwd_learnable_tau_lr": getattr(self, 'cwd_learnable_tau_lr', 1e-3),
            "cwd_learnable_tau_init": getattr(self, 'cwd_learnable_tau_init', 9.0),
            "kd_method": getattr(self, '_kd_method', 'cwd'),
            "mgd_mask_ratio": getattr(self, 'mgd_mask_ratio', 0.5),
            "fitnets_normalize": getattr(self, 'fitnets_normalize', True),
        }
        # ============================= Custom args: save for resume ==========================

        # ============================= Finetune: save maskbndict for resume ==========================
        if getattr(self, 'finetune', False):
            maskbndict = getattr(self, 'maskbndict', None) or getattr(self, 'kd_maskbndict', None)
            if maskbndict is not None:
                ckpt_dict["maskbndict"] = maskbndict
        # ============================= Finetune: save maskbndict for resume ==========================

        # ============================= DMS: save a params + optimizer ==========================
        if getattr(self, 'dms_enabled', False) and self.a_params:
            ckpt_dict["dms_a_params"] = {
                name: a.detach().cpu() for name, a in self.a_params.items()
            }
            if hasattr(self, 'dms_optimizer'):
                ckpt_dict["dms_optimizer"] = self.dms_optimizer.state_dict()
        # ============================= DMS: save a params + optimizer ==========================

        # ============================= DMS: clean hooks from EMA copy (closures can't be pickled) ==
        if getattr(self, 'dms_enabled', False) and self.dms_hooks:
            ema_model = ckpt_dict.get("ema")
            if ema_model is not None:
                for m in ema_model.modules():
                    m._forward_hooks.clear()
        # =======================================================================================

        # ============================= CWD: save state for resume ==========================
        if getattr(self, 'kd_enabled', False):
            ckpt_dict["kd_state"] = {
                "teacher": getattr(self, 'kd_teacher', None),
                "lambda": self._kd_lambda,
                "temperature": getattr(self, 'cwd_temperature', 6.0),
                "layers": getattr(self, 'kd_layers', 'neck'),
            }
            # Save learnable tau state
            if self.cwd_temp_mode == "learnable" and hasattr(self, 'cwd_log_tau'):
                ckpt_dict["cwd_learnable_tau_state"] = {
                    "log_tau": self.cwd_log_tau.detach().cpu(),
                    "optimizer": self.cwd_tau_optimizer.state_dict(),
                }
            maskbndict = getattr(self, 'kd_maskbndict', None)
            if maskbndict is not None:
                ckpt_dict["maskbndict"] = maskbndict
        # ============================= CWD: save state for resume ==========================

        # ============================= CWD: clean hooks from EMA copy ==========================
        if getattr(self, 'kd_enabled', False):
            ema_model = ckpt_dict.get("ema")
            if ema_model is not None:
                for m in ema_model.modules():
                    m._forward_hooks.clear()
        # ============================= CWD: clean hooks from EMA copy ==========================

        torch.save(ckpt_dict, buffer)
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            # Convert ul:// platform URIs and NDJSON files to local dataset format first
            data_str = str(self.args.data)
            if data_str.endswith(".ndjson") or (data_str.startswith("ul://") and "/datasets/" in data_str):
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo
                from ultralytics.utils.checks import check_file

                self.args.data = str(asyncio.run(convert_ndjson_to_yolo(check_file(self.args.data))))

            # Task-specific dataset checking
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict | None): Checkpoint to resume training from, or None if no checkpoint is loaded.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        # Truyen maskbndict khi resume finetune de build DetectionModelPruned
        maskbndict = None
        if getattr(self, 'finetune', False) and ckpt is not None:
            maskbndict = ckpt.get('maskbndict', None) or getattr(self, 'maskbndict', None)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1, maskbndict=maskbndict)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # ============================= disable scaler/grad clip =============================
        if getattr(self, 'sr', 0.0) > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        # ============================= disable scaler/grad clip =============================
        elif getattr(self, 'dms_enabled', False):
            dms_in_warmup = self.epoch < getattr(self, 'dms_warmup', 0)
            if getattr(self, 'dms_freeze', False):
                # ===== FREEZE MODE: không cần scaler (model đóng băng) =====
                # Chỉ step DMS optimizer, skip nếu NaN hoặc warmup
                if not dms_in_warmup:
                    dms_skip = False
                    for a in self.a_params.values():
                        if a.grad is not None and not torch.isfinite(a.grad).all():
                            dms_skip = True
                            break
                    if not dms_skip:
                        self.dms_optimizer.step()
                self.dms_optimizer.zero_grad()
                self.optimizer.zero_grad()
            else:
                # ===== NON-FREEZE MODE: scaler cho main, thủ công cho DMS =====
                dms_scale = self.scaler.get_scale()
                # Main optimizer qua scaler chuẩn
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # DMS optimizer: unscale thủ công, skip khi warmup
                if not dms_in_warmup:
                    dms_skip = False
                    for a in self.a_params.values():
                        if a.grad is not None:
                            a.grad.div_(dms_scale)
                            if not torch.isfinite(a.grad).all():
                                dms_skip = True
                    if not dms_skip:
                        self.dms_optimizer.step()
                self.dms_optimizer.zero_grad()

            # Clamp a to valid range [0, 1 - divisor/N] per layer
            with torch.no_grad():
                divisor = getattr(self, 'dms_divisor', 8)
                for name, a in self.a_params.items():
                    n_channels = self.bn_channels.get(name, 256)
                    a_max = 1.0 - divisor / n_channels
                    a.clamp_(0.0, a_max)
        else:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # MGD generator optimizer step (riêng biệt, giống DMS)
        if getattr(self, '_kd_method', None) == 'mgd' and hasattr(self, '_mgd_optimizer'):
            self._mgd_optimizer.step()
            self._mgd_optimizer.zero_grad()

        # CWD learnable tau optimizer step
        if self.cwd_temp_mode == "learnable" and hasattr(self, 'cwd_tau_optimizer'):
            # Unscale gradient manually (scaler chỉ biết main optimizer)
            scale = self.scaler.get_scale()
            if self.cwd_log_tau.grad is not None:
                self.cwd_log_tau.grad.div_(scale)
                if torch.isfinite(self.cwd_log_tau.grad).all():
                    self.cwd_tau_optimizer.step()
            self.cwd_tau_optimizer.zero_grad()
            # Clamp log_tau để tau ∈ [0.5, 20]
            with torch.no_grad():
                import math as _math
                self.cwd_log_tau.clamp_(_math.log(0.5), _math.log(20.0))

        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing of model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """Run validation on val set using self.validator.

        Returns:
            (tuple): A tuple containing:
                - metrics (dict | None): Dictionary of validation metrics, or None if validation was skipped.
                - fitness (float | None): Fitness score for the validation, or None if validation was skipped.
        """
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)

        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items, or a list of loss names if loss_items is None.

        Notes:
            This is not needed for classification but necessary for segmentation & detection.
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        unwrap_model(self.model).load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from a given checkpoint."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)

        # ============================= DMS: restore a_params + optimizer ==========================
        if getattr(self, 'dms_enabled', False) and self.a_params:
            saved_a = ckpt.get('dms_a_params', {})
            if saved_a:
                restored = 0
                for name, a_param in self.a_params.items():
                    if name in saved_a:
                        with torch.no_grad():
                            a_param.copy_(saved_a[name].to(a_param.device))
                        restored += 1
                LOGGER.info(f"[DMS] Restored {restored}/{len(self.a_params)} a_params from checkpoint.")
            else:
                LOGGER.warning("[DMS] No dms_a_params in checkpoint, using default init.")
            # Restore dms_optimizer state (Adam momentum etc.)
            saved_dms_opt = ckpt.get('dms_optimizer')
            if saved_dms_opt and hasattr(self, 'dms_optimizer'):
                self.dms_optimizer.load_state_dict(saved_dms_opt)
                LOGGER.info("[DMS] Restored dms_optimizer state from checkpoint.")
        # ============================= DMS: restore a_params + optimizer ==========================

        # ============================= CWD: restore learnable tau ==========================
        if self.cwd_temp_mode == "learnable" and hasattr(self, 'cwd_log_tau'):
            saved_tau_state = ckpt.get('cwd_learnable_tau_state', {})
            if saved_tau_state:
                saved_log_tau = saved_tau_state.get('log_tau')
                if saved_log_tau is not None:
                    with torch.no_grad():
                        self.cwd_log_tau.copy_(saved_log_tau.to(self.cwd_log_tau.device))
                    LOGGER.info(f"[CWD] Restored learnable tau={self.cwd_log_tau.exp().item():.4f}")
                saved_tau_opt = saved_tau_state.get('optimizer')
                if saved_tau_opt and hasattr(self, 'cwd_tau_optimizer'):
                    self.cwd_tau_optimizer.load_state_dict(saved_tau_opt)
                    LOGGER.info("[CWD] Restored tau optimizer state.")
        # ============================= CWD: restore learnable tau ==========================

        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # convert to list of params

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            import re

            # higher lr for certain parameters in MuSGD when funetuning
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
            g_ = []  # new param groups
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer
