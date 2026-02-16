# ----------------------------------------------------------------------------------
# train.py (full) — Pangu S2S trainer with DDP, AMP, compile(), SDPA fast paths,
#                   throttled metrics/logging, and correctness fixes.
# ----------------------------------------------------------------------------------

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta, datetime
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from itertools import product
import time
from multiprocessing import Process
import psutil
import shutil
import uuid
import os
import numpy as np
import argparse
import xarray as xr
import logging
import torch
import torchvision
from torchvision.utils import save_image
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib


def _str_to_bool(val):
    """Robust bool parser that accepts common truthy/falsey strings."""
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    if val in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if val in {"0", "false", "no", "n", "f", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")

try:
    # Deprecated in newer PyTorch; unused, safe fallback
    from torch.backends.cuda import sdp_kernel as _sdp_kernel
except Exception:
    _sdp_kernel = None

from utils import logging_utils
from utils.power_spectrum import *
from utils.losses import (
    Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,
    Latitude_weighted_CRPSLoss, Kl_divergence_gaussians
)
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
from utils.integrate import Integrator, forward_euler
from networks.pangu import PanguModel_Plasim
from utils.utils import log_memory_usage, log_gpu_memory

# ------------------------------------
# Feature detection
# ------------------------------------
def _is_torch_compile_available() -> bool:
    """Return True if torch.compile can be used in this environment."""
    has_compile = hasattr(torch, "compile")
    dynamo_ok = getattr(torch._dynamo, "is_dynamo_supported", lambda: False)()
    return bool(has_compile and dynamo_ok)

# ------------------------------------
# Global torch defaults & distributed
# ------------------------------------
logging_utils.config_logger()
torch._dynamo.config.optimize_ddp = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # may be overridden by CLI later
torch.cuda.empty_cache()

# H100-specific optimizations
if torch.cuda.is_available():
    # Enable TF32 for matmul and convolutions for H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prefer reduced-precision reductions to keep tensor cores busy
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    # Disable cuDNN benchmarking during training (already set later, but ensure it's consistent)
    torch.backends.cudnn.benchmark = True
    # Enable CUDA graphs if supported (H100 friendly)
    torch.backends.cudnn.allow_cudnn_rnn_fallback = False

logging.info("Torch version: {}".format(torch.__version__))

# Initialize DDP (torchrun sets env://)
dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=30))
world_rank = dist.get_rank()
print(f"World rank: {world_rank}")

# ------------------------------------
# Metrics helpers
# ------------------------------------
def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted/torch.sum(lat_weights_unweighted)

def weighted_rmse_torch_channels(pred, target, latitudes=None, weight=None):
    # [n, c, h, w] -> per-channel lat-weighted RMSE
    if weight is None:
        weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))
    return result

def weighted_rmse_torch_3D(pred, target, latitudes=None, weight=None):
    # [n, c, z, h, w] -> per-channel per-level lat-weighted RMSE
    if weight is None:
        weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))
    return result

def grad_norm(model):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def grad_max(model):
    max_grad = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_max = torch.max(torch.abs(p.grad.detach().data))
        if max_grad < param_max.item():
            max_grad = param_max.item()
    return param_max

def evaluate_iterative_forecast(da_fc, da_true, func, clim, mean_dims=['lat', 'lon', 'time'], weighted=True):
    scores = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        true = da_true.sel(lead_time=f)
        score = func(fc, true, clim, mean_dims=mean_dims, weighted=weighted)
        scores.append(score)
    return xr.concat(scores, dim='lead_time')

def compute_weighted_acc(da_fc, da_true, clim=None, weighted=True, mean_dims=xr.ALL_DIMS, **kwargs):
    da_fc = da_fc.assign_coords(dayofyear=da_fc['time'].dt.dayofyear)
    da_true = da_true.assign_coords(dayofyear=da_true['time'].dt.dayofyear)
    if clim is not None:
        if True:
            if 'zsfc' in clim:
                clim = clim.drop_vars('zsfc')
            if 'pr_12h' in da_fc:
                clim['pr_12h'] = clim['tas'].copy(); clim['pr_12h'][:] = 0.
            if 'pr_6h' in da_fc:
                clim['pr_6h'] = clim['tas'].copy(); clim['pr_6h'][:] = 0.
            if 'mrso' in da_fc:
                clim['mrso'] = clim['tas'].copy(); clim['mrso'][:] = 0.
            clim = clim[list(da_fc.data_vars)]
            clim = clim.transpose('dayofyear', 'plev', 'lat', 'lon')
            climatology_aligned = clim.sel(dayofyear=da_fc['dayofyear'])
            climatology_aligned = climatology_aligned.transpose(*da_fc.dims)
            climatology_aligned = climatology_aligned.assign_coords(lat=da_fc.lat)
            fa = da_fc - climatology_aligned
            a = da_true - climatology_aligned
        else:
            print(f"Error during climatology alignment")
            return xr.DataArray(np.nan, dims=['time'])
    else:
        fa = da_fc
        a = da_true

    fa = fa.drop_vars('dayofyear', errors='ignore')
    a = a.drop_vars('dayofyear', errors='ignore')

    if weighted:
        weights_lat = np.cos(np.deg2rad(a.lat)); weights_lat /= weights_lat.mean()
    else:
        weights_lat = 1.
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    numerator = (w * fa_prime * a_prime).sum(mean_dims)
    denominator = np.sqrt((w * fa_prime ** 2).sum(mean_dims) * (w * a_prime ** 2).sum(mean_dims))
    acc = numerator / denominator
    return acc

def to_ensemble_batch(data, ens_members):
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...)."""
    return (data.unsqueeze(1) * torch.ones(1, ens_members, *data.shape[1:]).to(data.device)).flatten(0, 1)

# ------------------------------------
# Trainer
# ------------------------------------
class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.compile_available = _is_torch_compile_available()
        
        # Warn about incompatible SDPA settings
        if getattr(params, 'enable_sdp_flash', False):
            logging.warning("⚠️  --enable-sdp-flash is NOT compatible with Pangu model architecture!")
            logging.warning("⚠️  The model's 3D windowed attention uses 5D tensors which require the math kernel")
            logging.warning("⚠️  Disabling this flag to prevent 'No available kernel' errors")
            params.enable_sdp_flash = False
        
        self.iters = 0
        self.startEpoch = 0
        self.epoch = self.startEpoch
        self.early_stop_epoch = params['early_stop_epoch'] - 1 if 'early_stop_epoch' in params else None
        self.run_uuid = str(uuid.uuid4())
        self.check_land_ocean_variables()
        self.get_dataset()
        self.spectra_dir, self.diagnostics_dir, self.output_dir = self.create_dirs(self.run_uuid)

        # AMP dtype selection (bf16 preferred for H100)
        amp_choice = getattr(self.params, "amp_dtype", "auto")
        if amp_choice == "bf16" and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
        elif amp_choice == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
        elif amp_choice == "fp32":
            self.amp_dtype = torch.float32
            logging.info("Using FP32 (no mixed precision)")
        else:
            # H100 strongly prefers BF16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                logging.info("Using BF16 for H100 optimization")
            elif torch.cuda.is_available():
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float32

        # Init wandb
        if params.log_to_wandb:
            resume = "allow" if params.resuming else "never"
            wandb.init(
                config=params, name=f'{params.name}-{params.run_iter}',
                entity=params.entity, group=params.group,
                project=params.project, resume=resume
            )
            logging.info("WandB initialized with config: %s", params)
        self.init_wandb(self.params)

        logging.info('Params %s', params)

    def setup_model(self):
        # Model + integrator
        self.mask_bool, self.land_mask = self.get_land_mask_bool()
        self.model = self.get_model()
        self.scaler = GradScaler(enabled=(torch.cuda.is_available() and self.amp_dtype == torch.float16))

        # >>> create optimizer BEFORE scheduler (and before resume)
        self.optimizer = self.get_optimizer()

        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path)
            logging.info("Resuming from checkpoint: %s", params.checkpoint_path)
        else:
            logging.info("Starting fresh training run")

        self.setup_scheduler()
        self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic = self.setup_loss_fun()

    def check_land_ocean_variables(self) -> None:
        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(self.params, 'land_variables'):
            if len(self.params.land_variables) > 0: self.has_land = True
        else:
            self.params['land_variables'] = []
        if hasattr(self.params, 'ocean_variables'):
            if len(self.params.ocean_variables) > 0: self.has_ocean = True
        else:
            self.params['ocean_variables'] = []
        if hasattr(self.params, 'mask_output'):
            self.mask_output = self.params.mask_output

    def create_dirs(self, run_uuid:int)->tuple[str, str, str]:
        main_dirs = ["spectra_out", "gif_out", "acc_plots"]
        for dir_name in main_dirs:
            os.makedirs(os.path.join(os.getcwd(), dir_name), exist_ok=True)
        spectra_dir = os.path.join(os.getcwd(), "spectra_out", self.run_uuid)
        diagnostics_dir = os.path.join(os.getcwd(), "gif_out", self.run_uuid)
        output_dir = os.path.join(os.getcwd(), "acc_plots", self.run_uuid)
        logging.info('The output directories %s ; %s; %s', spectra_dir, diagnostics_dir, output_dir)
        if world_rank == 0:
            os.makedirs(spectra_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created directory: {spectra_dir}")
            logging.info(f"Created directory: {diagnostics_dir}")
            logging.info(f"Created directory: {output_dir}")
        return spectra_dir, diagnostics_dir, output_dir

    def get_dataset(self):
        logging.info('rank %d, begin data loader init', self.world_rank)
        if self.params.train_year_to_year:
            self.train_data_loaders, self.train_datasets, self.train_samplers = [], [], []
            for year_start in range(params.train_year_start, params.train_year_end):
                year_end = year_start + 1
                train_data_loader, train_dataset, train_sampler = get_data_loader(
                    params, params.data_dir, dist.is_initialized(),
                    year_start=year_start, year_end=year_end, train=True
                )
                self.train_data_loaders.append(train_data_loader)
                self.train_datasets.append(train_dataset)
                self.train_samplers.append(train_sampler)
        else:
            train_data_loader, train_dataset, train_sampler = get_data_loader(
                params, params.data_dir, dist.is_initialized(),
                year_start=params.train_year_start,
                year_end=params.train_year_end,
                train=True
            )
            self.train_data_loaders = [train_data_loader]
            self.train_datasets = [train_dataset]
            self.train_samplers = [train_sampler]

        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params, params.data_dir, dist.is_initialized(),
            year_start=params.val_year_start, year_end=params.val_year_end, train=False,
            num_inferences=params.num_inferences, validate=True
        )

        self.constant_boundary_data = self.train_datasets[0].constant_binary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1) \
            if hasattr(self.train_datasets[0], 'constant_binary_data') else \
            self.train_datasets[0].constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        
        # Optimize memory format for potential speed gains
        if torch.cuda.is_available() and getattr(params, 'use_channels_last', False):
            try:
                self.constant_boundary_data = self.constant_boundary_data.to(memory_format=torch.channels_last)
            except Exception:
                pass
        
        if params.num_ensemble_members > 1:
            self.constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, params.num_ensemble_members)
            logging.info('Ensemble Mode. Ensemble size = {params.num_ensemble_members}\n')

        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path).rename({'time': 'dayofyear'})
        self.lat_t = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)
        # Precompute latitude weights to avoid per-step cos/sum work
        with torch.inference_mode():
            _lat_w = latitude_weighting_factor_torch(self.lat_t)   # [H]
            self.lat_weight_2d = _lat_w.view(1, 1, -1, 1)          # [1,1,H,1]
            self.lat_weight_3d = _lat_w.view(1, 1, 1, -1, 1)       # [1,1,1,H,1]

        if world_rank == 0:
            logging.info('rank %d, data loader initialized', self.world_rank)

    def init_wandb(self, params:dict):
        if params.log_to_wandb:
            wandb.define_metric("epoch")
            wandb.define_metric("ACC_plot", step_metric="epoch")
            wandb.define_metric("power_spectrum_plot", step_metric="epoch")
            epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air', 'valid_mean_norm_lwrmse']
            for l, steps in enumerate(params.forecast_lead_times):
                epoch_metrics.append(f"valid_lwrmse_sfc_{steps}step")
                epoch_metrics.append(f"valid_lwrmse_pl_{steps}step")
                epoch_metrics.append(f"valid_loss_{steps}step")
                for j, var in enumerate(self.valid_dataset.surface_variables):
                    epoch_metrics.append(f'valid_{var}_{steps}step_lwrmse')
                for j, var in enumerate(self.valid_dataset.upper_air_variables):
                    for k, level in enumerate(self.valid_dataset.levels):
                        epoch_metrics.append(f'valid_{var}_level{level:.3f}_{steps}step_lwrmse')
            for metric in epoch_metrics:
                wandb.define_metric(metric, step_metric="epoch")

    def get_land_mask_bool(self) -> torch.Tensor:
        mask_bool = []
        land_mask = []
        if self.params.nettype == 'pangu_plasim':
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = torch.clone(self.train_datasets[0].land_mask.detach()).to(self.device)
                print(f'Land Mask shape: {land_mask.shape}')
                mask_bool = []
                for var in self.params.surface_variables:
                    if var in self.params.land_variables:
                        mask_bool.append(torch.clone(land_mask).to(torch.bool))
                    elif var in self.params.ocean_variables:
                        mask_bool.append(torch.logical_not(torch.clone(land_mask).to(torch.bool)))
                    else:
                        mask_bool.append(torch.ones(land_mask.shape, device=self.device, dtype=torch.bool))
                mask_bool = torch.stack(mask_bool)
            else:
                land_mask = None
        else:
            raise Exception("not implemented")
        return mask_bool, land_mask

    def get_model(self):
        """Build model and wrap with DDP (plus optional compile)."""
        if self.params.nettype == 'pangu_plasim':
            if self.params.predict_delta:
                self.model = PanguModel_Plasim(params, land_mask=self.land_mask).to(self.device)
                self.integrator = Integrator(
                    params,
                    surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                    surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                    upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                    upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)
                ).to(self.device)
            else:
                if hasattr(params, 'mask_fill'):
                    self.model = PanguModel_Plasim(params, land_mask=self.land_mask, mask_fill=params.mask_fill).to(self.device)
                else:
                    self.model = PanguModel_Plasim(params, land_mask=self.land_mask, mask_fill=self.train_datasets[0].mask_fill).to(self.device)
        else:
            raise Exception("not implemented")

        # Only compile if NOT resuming from a non-compiled checkpoint
        should_compile = bool(getattr(params, "torch_compile", False))
        if should_compile and not self.compile_available:
            logging.warning("torch.compile requested but not available; skipping compilation.")
            should_compile = False
        if should_compile and params.resuming:
            # Check if checkpoint was compiled
            try:
                ckpt = torch.load(params.checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_has_orig_mod = any('_orig_mod' in k for k in ckpt['model_state'].keys())
                if not checkpoint_has_orig_mod:
                    logging.warning("Checkpoint was not compiled. Disabling torch.compile for compatibility.")
                    should_compile = False
            except Exception as e:
                logging.warning(f"Could not check checkpoint compilation status: {e}")
        
        # Optional torch.compile before wrapping with DDP
        if should_compile:
            try:
                _compile_mode = getattr(params, "compile_mode", "reduce-overhead")
                # Use max-autotune for H100
                if getattr(params, "compile_max_autotune", False):
                    _compile_mode = "max-autotune"
                self.model = torch.compile(self.model, mode=_compile_mode, fullgraph=False)
                logging.info(f"torch.compile enabled (mode={_compile_mode})")
            except Exception as _e:
                logging.warning(f"torch.compile failed: {_e}; continuing without compile.")

        # Wrap with DDP
        if dist.is_initialized():
            ddp_static = bool(getattr(params, "ddp_static_graph", False))
            bucket_cap = int(getattr(params, "ddp_bucket_cap_mb", 25))
            
            # For H100, increase bucket size for better bandwidth utilization
            if bucket_cap == 25:  # default
                bucket_cap = 200  # Increase to 200MB for H100
            
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=[params.local_rank],
                find_unused_parameters=not ddp_static,
                static_graph=ddp_static,
                bucket_cap_mb=bucket_cap,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,  # Skip buffer broadcast if not needed
            )

            # Optional: PowerSGD gradient compression
            if getattr(params, 'ddp_powersgd', False):
                try:
                    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
                    state = powerSGD.PowerSGDState(
                        process_group=dist.group.WORLD,
                        warm_start=True,
                        use_error_feedback=True,
                        start_powerSGD_iter=10,
                        matrix_approximation_rank=int(getattr(params, 'powersgd_rank', 1)),
                    )
                    self.model.register_comm_hook(state, powerSGD.powerSGD_hook)
                    logging.info("DDP PowerSGD hook enabled (rank=%s)", getattr(params, 'powersgd_rank', 1))
                except Exception as _e:
                    logging.warning(f"PowerSGD enable failed/skipped: {_e}")
            
            # FP16 compression for gradients (H100 has high bandwidth, but still helps)
            elif getattr(params, 'ddp_fp16_compress', False):
                try:
                    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
                    self.model.register_comm_hook(state=None, hook=default.fp16_compress_hook)
                    logging.info("DDP FP16 compression hook enabled")
                except Exception as _e:
                    logging.warning(f"FP16 compression enable failed/skipped: {_e}")

        # W&B watch (optional)
        if self.params.log_to_wandb and getattr(self.params, 'watch_model', False):
            wandb.watch(self.model, log="all", log_freq=getattr(self.params, 'watch_log_freq', 200))
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
        return self.model

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_optimizer(self):
        # Prefer fused Adam on CUDA; fallback to foreach; else default
        if getattr(params, 'optimizer_type', 'Adam') == 'FusedAdam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay, fused=True
            )
        else:
            if torch.cuda.is_available():
                try:
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay, fused=True
                    )
                except (TypeError, RuntimeError):
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay, foreach=True
                    )
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        return self.optimizer

    def setup_scheduler(self):
        if self.params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode='min'
            )
        elif self.params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.params.max_epochs, last_epoch=self.startEpoch-1
            )
        elif self.params.scheduler == 'OneCycleLR':
            steps_per_epoch = sum(len(loader) for loader in self.train_data_loaders)
            pct_start        = getattr(self.params, 'oc_pct_start', 0.3)
            div_factor       = getattr(self.params, 'oc_div_factor', 25)
            final_div_factor = getattr(self.params, 'oc_final_div_factor', 1e4)
            if self.startEpoch < 1:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=self.params.lr,
                    epochs=self.params.max_epochs, steps_per_epoch=steps_per_epoch,
                    pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=self.params.lr,
                    epochs=self.params.max_epochs, steps_per_epoch=steps_per_epoch,
                    last_epoch=(self.startEpoch-1)*steps_per_epoch,
                    pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
                )
            logging.info("Scheduler is setup")
        else:
            self.scheduler = None

    def setup_loss_fun(self):
        self.loss_obj_pl = 0
        self.loss_obj_sfc = 0
        self.loss_obj_diagnostic = 0
        self.loss_vae = 0
        if self.params.vae_loss:
            self.loss_vae = Kl_divergence_gaussians()
            logging.info("VAE loss is setup")
        if self.params.loss == 'l1':
            self.loss_obj_pl = torch.nn.L1Loss()
            self.loss_obj_sfc = Masked_L1Loss(self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else torch.nn.L1Loss()
            if self.params.has_diagnostic: self.loss_obj_diagnostic = torch.nn.L1Loss()
        elif self.params.loss == 'l2':
            self.loss_obj_pl = torch.nn.MSELoss()
            self.loss_obj_sfc = Masked_MSELoss(self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else torch.nn.MSELoss()
            if self.params.has_diagnostic: self.loss_obj_diagnostic = torch.nn.MSELoss()
        elif self.params.loss == 'weightedl1':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_L1Loss(self.lat)
            self.loss_obj_sfc = Latitude_weighted_masked_L1Loss(self.lat, self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else Latitude_weighted_L1Loss(self.lat)
            if self.params.has_diagnostic: self.loss_obj_diagnostic = Latitude_weighted_L1Loss(self.lat)
        elif self.params.loss == 'weightedl2':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_MSELoss(self.lat)
            self.loss_obj_sfc = Latitude_weighted_masked_MSELoss(self.lat, self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else Latitude_weighted_MSELoss(self.lat)
            if self.params.has_diagnostic: self.loss_obj_diagnostic = Latitude_weighted_MSELoss(self.lat)
        elif self.params.loss == 'weightedCRPS':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
            self.loss_obj_sfc = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members, self.mask_bool) if self.has_land or self.has_ocean else Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
            if self.params.has_diagnostic: self.loss_obj_diagnostic = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
        else:
            raise NotImplementedError
        
        # Optional: compile loss functions for speed
        if getattr(self.params, 'compile_loss', False) and torch.cuda.is_available() and self.compile_available:
            try:
                if self.loss_obj_pl != 0:
                    self.loss_obj_pl = torch.compile(self.loss_obj_pl, mode='reduce-overhead')
                if self.loss_obj_sfc != 0:
                    self.loss_obj_sfc = torch.compile(self.loss_obj_sfc, mode='reduce-overhead')
                if self.loss_obj_diagnostic != 0:
                    self.loss_obj_diagnostic = torch.compile(self.loss_obj_diagnostic, mode='reduce-overhead')
                logging.info("Loss functions compiled")
            except Exception as _e:
                logging.warning(f"Loss compilation skipped: {_e}")
        elif getattr(self.params, 'compile_loss', False) and not self.compile_available:
            logging.warning("Loss compilation requested but torch.compile is unavailable on this setup.")
        
        logging.info("Losses is setup")
        return self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")
        best_valid_loss = 1.e6
        early_stopping_counter = 0
        early_stop_epoch_triggered = False

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if world_rank == 0:
                logging.info(f'Starting epoch {epoch + 1}/{self.params.max_epochs}')

            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                if self.params.log_to_screen:
                    logging.info(f'Completed early stop epoch {self.early_stop_epoch}. Terminating training.')
                early_stop_epoch_triggered = True
                break

            if dist.is_initialized():
                for sampler in self.train_samplers:
                    sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            logging.info(f"Epoch {epoch + 1} training time: {tr_time:.2f} seconds, data loading time: {data_time:.2f} seconds")

            # Print benchmark summary if max_steps was used
            _max_steps = int(getattr(self.params, 'max_steps', 0))
            if _max_steps > 0 and self.iters >= _max_steps and world_rank == 0:
                _actual_steps = self.iters
                _sec_per_step = tr_time / max(1, _actual_steps)
                _samples_per_sec = int(getattr(self.params, 'global_batch_size', self.params.batch_size)) / _sec_per_step
                print(f"BENCHMARK_RESULT: steps={_actual_steps} train_time={tr_time:.2f}s sec_per_step={_sec_per_step:.4f} samples_per_sec={_samples_per_sec:.2f}", flush=True)

            valid_time, valid_logs = self.validate_one_epoch()
            logging.info(f"Epoch {epoch + 1} validation time: {valid_time:.2f} seconds")
            torch.cuda.empty_cache()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    break

            if valid_logs['valid_loss'] <= best_valid_loss:
                best_valid_loss = valid_logs['valid_loss']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        self.save_checkpoint(self.params.best_checkpoint_path)
            if world_rank == 0:
                self.log_wandb_epoch(epoch)
                self.log_screen_epoch(epoch, start, train_logs, valid_logs, early_stopping_counter)

            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                if self.params.log_to_screen and world_rank == 0:
                    logging.info('Early stopping triggered. Terminating training.')
                break

        if self.params.log_to_screen and world_rank == 0:
            if early_stop_epoch_triggered:
                logging.info(f'Training finished early at epoch {self.early_stop_epoch} due to early_stop_epoch setting.')
            else:
                logging.info('Completed all epochs. Training finished normally.')

    def log_wandb_epoch(self, epoch:int)->None:
        if self.params.log_to_wandb:
            for pg in self.optimizer.param_groups:
                lr = pg['lr']
            wandb.log({'lr': lr, 'epoch': self.epoch})

    def log_screen_epoch(self, epoch:int, start, train_logs, valid_logs, early_stopping_counter, **kwargs) ->None:
        if self.params.log_to_screen:
            logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            logging.info('Train loss: {}. Validation loss: {}. Surface Val loss: {}. Upper Air Val loss: {}'.format(
                train_logs['train_loss'], valid_logs['valid_loss'], valid_logs['valid_loss_sfc'], valid_logs['valid_loss_upper_air']))
            lead_times_steps = self.params.forecast_lead_times
            multi_step_loss_str = '. '.join([f"{step}-step Val loss: {valid_logs.get(f'valid_loss_{step}step', 'N/A')}" for step in lead_times_steps])
            logging.info(f'Multi-step validation losses: {multi_step_loss_str}')
            if self.params.early_stopping:
                logging.info(f'EarlyStopping counter: {early_stopping_counter} out of {self.params.early_stopping_patience}')

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0.0
        data_time = 0.0
        total_iterations = sum(len(loader) for loader in self.train_data_loaders)
        diagnostic_logs = {}
        loss = 0.0

        logging.info(f"Expected total batches: {total_iterations}")
        if not self.train_data_loaders:
            logging.warning("No training data loaders available.")
            return 0, 0, {"train_loss": 0.0}

        self.model.train()
        
        # Disable progress bar on non-zero ranks to reduce overhead
        if self.world_rank == 0:
            pbar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        else:
            pbar = None
        
        running_results = {"batch_sizes": 0, "loss": 0.0}
        log_every = max(1, int(getattr(self.params, "log_every_n_steps", 20)))
        metrics_every = int(getattr(self.params, "metrics_every", 0)) or log_every

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            logging.debug(f"Processing year idx {year_idx}")
            current_dataset = self.train_datasets[year_idx]
            if self.params.train_year_to_year:
                logging.debug(f"Processing year {self.params.train_year_start + year_idx}")
            else:
                logging.debug(f"Processing years {self.params.train_year_start} to {self.params.train_year_end}")

            for i, data in enumerate(train_data_loader):
                # Benchmark early exit: stop after max_steps iterations
                _max_steps = int(getattr(self.params, 'max_steps', 0))
                if _max_steps > 0 and self.iters >= _max_steps:
                    if self.world_rank == 0:
                        logging.info(f"Benchmark: reached max_steps={_max_steps}, stopping.")
                    break

                if (self.world_rank == 0) and (i % (10 * log_every) == 0):
                    logging.info("training on batch %d of year %d", i, self.params.train_year_start + year_idx)

                if self.params.mode == "test" and i >= self.params.test_iterations:
                    if pbar is not None:
                        pbar.update(total_iterations - self.iters)
                else:
                    self.iters += 1
                    data_start = time.time()
                    input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = self._prepare_inputs_batch(data)
                    data_time += time.time() - data_start

                    tr_start = time.time()
                    _accum_steps = max(1, int(getattr(self.params, "accum_steps", 1)))
                    if (i % _accum_steps) == 0:
                        self.optimizer.zero_grad(set_to_none=True)

                    # forward + loss
                    output_surface, output_upper_air, output_diagnostic, loss_sfc, loss_pl, loss_diagnostic, loss_vae, loss = self.cal_loss(
                        input_surface, self.constant_boundary_data, varying_boundary_data, input_upper_air,
                        target_diagnostic, target_surface, target_upper_air
                    )

                    # accumulation with no_sync + optional grad clipping
                    _is_last_micro = ((i + 1) % _accum_steps == 0) or (i == len(train_data_loader) - 1)
                    if dist.is_initialized() and hasattr(self.model, "no_sync") and not _is_last_micro:
                        _sync_cm = self.model.no_sync()
                    else:
                        _sync_cm = contextlib.nullcontext()
                    with _sync_cm:
                        self.scaler.scale(loss / _accum_steps).backward()
                    if _is_last_micro:
                        if float(getattr(self.params, 'max_grad_norm', 0.0)) > 0.0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.params.max_grad_norm))
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.params.scheduler == 'OneCycleLR':
                            self.scheduler.step()
                    tr_end_time = time.time()
                    
                    # Only log timing on rank 0 and less frequently
                    if (self.world_rank == 0) and (i % (5 * log_every) == 0):
                        logging.info(f"Backprop + step took {tr_end_time - tr_start:.4f}s")

                    # Compute metrics only every N steps to reduce overhead
                    if (i % metrics_every == 0):
                        with torch.inference_mode():
                            if self.params.predict_delta:
                                output_surface, output_upper_air = self.integrator(input_surface, input_upper_air, output_surface, output_upper_air)
                                target_surface_int, target_upper_air_int = self.integrator(input_surface, input_upper_air, target_surface, target_upper_air)
                            else:
                                target_surface_int = target_surface
                                target_upper_air_int = target_upper_air

                            # Use pre-computed weights
                            surface_lwrmse = weighted_rmse_torch_channels(output_surface, target_surface_int, weight=self.lat_weight_2d)
                            upper_air_lwrmse = weighted_rmse_torch_3D(output_upper_air, target_upper_air_int, weight=self.lat_weight_3d)

                            if self.params.has_diagnostic:
                                diagnostic_lwrmse = weighted_rmse_torch_channels(output_diagnostic, target_diagnostic, weight=self.lat_weight_2d)
                                mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, diagnostic_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim=-1))
                            else:
                                diagnostic_lwrmse = None
                                mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim=-1))

                            diagnostic_logs = self.diagnostic_log_per_iter(
                                diagnostic_logs, diagnostic_lwrmse, surface_lwrmse, upper_air_lwrmse, current_dataset,
                                step=i,
                                train_batch_loss=loss.detach(), train_batch_loss_sfc=loss_sfc.detach(),
                                train_batch_loss_upper_air=loss_pl.detach(), train_batch_loss_diagnostic=loss_diagnostic.detach() if isinstance(loss_diagnostic, torch.Tensor) else loss_diagnostic,
                                train_batch_loss_vae=loss_vae.detach() if isinstance(loss_vae, torch.Tensor) else loss_vae, 
                                train_mean_norm_lwrmse=mean_norm_lwrmse
                            )

                            # Throttle W&B logging for speed
                            if self.world_rank == 0 and self.params.log_to_wandb and (self.iters % log_every == 0):
                                wandb.log(diagnostic_logs, step=self.iters)

                    tr_time += time.time() - tr_start
                    # pbar line expects float
                    _disp_loss = diagnostic_logs.get('train_batch_loss', 0.0)
                    if isinstance(_disp_loss, torch.Tensor):
                        _disp_loss = float(_disp_loss.item())
                    if pbar is not None and (i % log_every == 0):
                        pbar.set_description(f"Year {self.params.train_year_start + year_idx}, Loss: {_disp_loss:.4f}")
                        pbar.update(log_every)
                    elif pbar is not None:
                        pbar.update(1)

            # Also break the outer year loop if max_steps reached
            _max_steps = int(getattr(self.params, 'max_steps', 0))
            if _max_steps > 0 and self.iters >= _max_steps:
                break
        
        if pbar is not None:
            pbar.close()

        logs = self.diagnostic_log_per_epoch(diagnostic_logs, train_loss=loss, epoch=self.epoch)
        return tr_time, data_time, logs

    def _prepare_inputs_batch(self, data:torch.Tensor):
        input_surface = input_upper_air = target_surface = target_upper_air = target_diagnostic = varying_boundary_data = 0
        if self.params.has_diagnostic:
            input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
        else:
            input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)

        if self.params.num_ensemble_members > 1:
            if self.params.has_diagnostic:
                ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in
                                    [input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data]]
                input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = ensemble_batches
            else:
                ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in
                                    [input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data]]
                input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data = ensemble_batches

        return input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data

    def cal_loss(self, input_surface, constant_boundary_data, varying_boundary_data, input_upper_air,
                 target_diagnostic, target_surface, target_upper_air, **kwargs):
        output_surface = output_upper_air = output_diagnostic = 0
        loss = loss_diagnostic = loss_pl = loss_sfc = loss_vae = 0
        with autocast(device_type="cuda",
                      dtype=self.amp_dtype if torch.cuda.is_available() else torch.float32,
                      enabled=torch.cuda.is_available() and self.amp_dtype != torch.float32):
            if self.params.has_diagnostic:
                output_surface, output_upper_air, output_diagnostic, mu, sigma, mu2, sigma2 = self.model(
                    input_surface, self.constant_boundary_data, varying_boundary_data, input_upper_air,
                    target_surface, target_upper_air, train=True)
                loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
            else:
                output_surface, output_upper_air, mu, sigma, mu2, sigma2 = self.model(
                    input_surface, self.constant_boundary_data, varying_boundary_data, input_upper_air,
                    target_surface, target_upper_air, train=True)

            loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
            loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

            if self.params.has_diagnostic:
                loss = (loss_sfc + loss_diagnostic) * 0.25 + loss_pl
            else:
                loss = (loss_sfc * 0.25) + loss_pl

            if self.params.vae_loss:
                loss_vae = self.loss_vae(mu, sigma, mu2, sigma2)
                loss += self.params.vae_loss_weight * loss_vae

        return output_surface, output_upper_air, output_diagnostic, loss_sfc, loss_pl, loss_diagnostic, loss_vae, loss

    def diagnostic_log_per_iter(self, diagnostic_logs, diagnostic_lwrmse, surface_lwrmse, upper_air_lwrmse, current_dataset, step=None, **kwargs)->dict:
        """Logging per iteration (optionally throttled)."""
        # Optional grad stats (expensive)
        g_every = int(getattr(self.params, 'grad_stats_every', 0) or 0)
        if g_every > 0 and step is not None and (step % g_every == 0):
            diagnostic_logs['batch_grad_norm'] = torch.tensor([grad_norm(self.model)], device=self.device, dtype=torch.float32)
            diagnostic_logs['batch_grad_max']  = torch.tensor([grad_max(self.model)], device=self.device, dtype=torch.float32)

        # Always carry forward the tensors we need to all_reduce
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                diagnostic_logs[key] = value
            else:
                diagnostic_logs[key] = torch.tensor(value, device=self.device, dtype=torch.float32)

        # Compute/add RMSEs only when we asked for them
        if self.params.has_diagnostic and diagnostic_lwrmse is not None:
            for j, var in enumerate(current_dataset.diagnostic_variables):
                diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(diagnostic_lwrmse[:, j]) * current_dataset.diagnostic_std[j]
        if surface_lwrmse is not None and upper_air_lwrmse is not None:
            for j, var in enumerate(current_dataset.surface_variables):
                diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(surface_lwrmse[:, j]) * current_dataset.surface_std[j]
            for j, var in enumerate(current_dataset.upper_air_variables):
                for k, level in enumerate(current_dataset.levels):
                    diagnostic_logs[f'train_{var}_level{level:.4f}_lwrmse'] = torch.mean(upper_air_lwrmse[:, j, k]) * current_dataset.upper_air_std[j, k]

        if dist.is_initialized():
            for key in sorted(list(diagnostic_logs.keys())):
                if key == 'batch_grad_max':
                    grad_max_tensor = torch.zeros(dist.get_world_size(), dtype=torch.float32, device=self.device)
                    dist.all_gather_into_tensor(grad_max_tensor, diagnostic_logs[key])
                    diagnostic_logs[key] = torch.max(grad_max_tensor)
                else:
                    val = diagnostic_logs[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(val, device=self.device, dtype=torch.float32)
                    dist.all_reduce(val)
                    diagnostic_logs[key] = float(val / dist.get_world_size())
        else:
            # Convert to float for single-GPU case
            for key in list(diagnostic_logs.keys()):
                val = diagnostic_logs[key]
                if isinstance(val, torch.Tensor):
                    diagnostic_logs[key] = float(val.item())
        return diagnostic_logs

    def diagnostic_log_per_epoch(self, diagnostic_logs, train_loss, epoch, **kwargs)->dict:
        logs = {}
        if self.params.diagnostic_logs:
            with torch.inference_mode():
                diagnostic_logs['train_loss'] = train_loss if isinstance(train_loss, float) else float(train_loss)
                if dist.is_initialized():
                    val = torch.tensor(diagnostic_logs['train_loss'], device=self.device, dtype=torch.float32)
                    dist.all_reduce(val)
                    diagnostic_logs['train_loss'] = float(val / dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                if self.params.log_to_wandb:
                    wandb.log(logs)
                return diagnostic_logs
        else:
            with torch.inference_mode():
                logs = {'train_loss': float(train_loss) if not isinstance(train_loss, float) else train_loss, 'epoch': self.epoch}
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    if isinstance(logs[key], (int, float)):
                        t = torch.tensor(logs[key], device=self.device, dtype=torch.float32)
                    else:
                        t = logs[key]
                    dist.all_reduce(t)
                    logs[key] = float(t / dist.get_world_size())
            if self.params.log_to_wandb:
                wandb.log(logs)
            return logs

    def inti_valid_loss(self, lead_times_steps) -> tuple:
        if self.params.has_diagnostic:
            valid_buff = torch.zeros((5), dtype=torch.float32, device=self.device)
            valid_loss_diag = valid_buff[3].view(-1)
        else:
            valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
            valid_loss_diag = None
        valid_loss = valid_buff[0].view(-1)
        valid_loss_sfc = valid_buff[1].view(-1)
        valid_loss_pl = valid_buff[2].view(-1)
        valid_steps = valid_buff[-1].view(-1)

        valid_surface_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.surface_variables)), dtype=torch.float32, device=self.device)
        valid_upper_air_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.upper_air_variables), len(self.valid_dataset.levels)), dtype=torch.float32, device=self.device)
        if self.params.has_diagnostic:
            valid_diagnostic_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.diagnostic_variables)), dtype=torch.float32, device=self.device)
        else:
            valid_diagnostic_lwrmse = None

        multi_step_losses = {f"valid_loss_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        if self.params.has_diagnostic:
            multi_step_rmse = {f"valid_lwrmse_sfc_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps} |\
                {f"valid_lwrmse_pl_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}|\
                {f"valid_lwrmse_diag_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        else:
            multi_step_rmse = {f"valid_lwrmse_sfc_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps} |\
                {f"valid_lwrmse_pl_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        return valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, multi_step_losses, multi_step_rmse

    def validate_one_epoch(self):
        if world_rank == 0:
            print("Validating...")
        self.model.eval()

        lead_times_steps = self.params.forecast_lead_times
        with torch.inference_mode():
            latitudes = self.lat_t

        valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, \
        valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, \
        multi_step_losses, multi_step_rmse = self.inti_valid_loss(lead_times_steps)

        valid_start = time.time()
        nb = len(self.valid_data_loader)
        diagnostic_logs = {}
        sample_idx = np.random.randint(len(self.valid_data_loader))

        all_predictions = []
        all_ground_truths = []
        acc_predictions = []
        acc_ground_truths = []

        with torch.inference_mode():
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
                if world_rank == 0 and (i % 10 == 0):
                    print(f"Validating batch {i+1}/{nb}")
                if self.params.predict_delta:
                    if self.params.has_diagnostic:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_diagnostic, val_target_surface_delta, val_target_upper_air_delta,\
                            val_varying_boundary_data, times = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    else:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_surface_delta, val_target_upper_air_delta,\
                            val_varying_boundary_data, times = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                else:
                    if self.params.has_diagnostic:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_diagnostic, val_varying_boundary_data, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    else:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)

                if self.params.num_ensemble_members > 1:
                    if self.params.has_diagnostic:
                        ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in
                                        [val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
                                         val_target_diagnostic, val_varying_boundary_data]]
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, \
                        val_target_diagnostic, val_varying_boundary_data = ensemble_batches
                    else:
                        ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in
                                        [val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
                                         val_varying_boundary_data]]
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, \
                        val_varying_boundary_data = ensemble_batches

                # start times per sample
                start_times = []
                for i2 in range(times.shape[0]):
                    start_time = self.valid_dataset.datetime_class(times[i2,0].item(), times[i2,1].item(), times[i2,2].item(), hour=times[i2,3].item())
                    start_times.append(start_time)

                max_lead_time = max(lead_times_steps)

                # ACC tensors (all time steps)
                val_output_surface_acc = np.zeros((val_input_surface.shape[0], max_lead_time,
                                                   val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]), dtype=np.float32)
                val_output_upper_air_acc = np.zeros((val_input_upper_air.shape[0], max_lead_time,
                                                     val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                     val_input_upper_air.shape[3], val_input_upper_air.shape[4]), dtype=np.float32)
                if self.params.has_diagnostic:
                    val_output_diagnostic_acc = np.zeros((val_target_diagnostic.shape[0], max_lead_time,
                                                          val_target_diagnostic.shape[2], val_target_diagnostic.shape[3],
                                                          val_target_diagnostic.shape[4]), dtype=np.float32)
                    val_output_diagnostic_t = np.zeros((val_target_diagnostic.shape[0], len(lead_times_steps),
                                                        val_target_diagnostic.shape[2], val_target_diagnostic.shape[3],
                                                        val_target_diagnostic.shape[4]), dtype=np.float32)

                # tensors for selected lead times
                val_output_surface_t = np.zeros((val_input_surface.shape[0], len(lead_times_steps),
                                                 val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]), dtype=np.float32)
                val_output_upper_air_t = np.zeros((val_input_upper_air.shape[0], len(lead_times_steps),
                                                   val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                   val_input_upper_air.shape[3], val_input_upper_air.shape[4]), dtype=np.float32)

                step_idx = 0
                for step in range(max_lead_time):
                    if self.params.has_diagnostic:
                        val_output_surface, val_output_upper_air, val_output_diagnostic, _, _  = self.model(
                            val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                    else:
                        val_output_surface, val_output_upper_air,  _, _ = self.model(
                            val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)

                    if (step + 1) in lead_times_steps:
                        target_index = step
                        if self.params.predict_delta:
                            loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface_delta[:, target_index])
                            loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air_delta[:, target_index])
                        else:
                            loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface[:, target_index])
                            loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air[:, target_index])
                        if self.params.has_diagnostic:
                            loss_diag = self.loss_obj_diagnostic(val_output_diagnostic, val_target_diagnostic[:, target_index])
                            loss = (loss_sfc + loss_diag) * 0.25 + loss_pl
                        else:
                            loss = (loss_sfc * 0.25 + loss_pl)
                        multi_step_losses[f"valid_loss_{step+1}step"] += loss

                        if step == 0:
                            valid_loss += loss
                            valid_loss_sfc += loss_sfc
                            valid_loss_pl += loss_pl
                            if self.params.has_diagnostic:
                                valid_loss_diag += loss_diag

                    if self.params.predict_delta:
                        val_output_surface, val_output_upper_air = self.integrator(val_input_surface, val_input_upper_air, val_output_surface, val_output_upper_air)

                    val_output_surface_acc[:, step] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                    val_output_upper_air_acc[:, step] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()
                    if self.params.has_diagnostic:
                        val_output_diagnostic_acc[:, step] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()

                    if (step + 1) in lead_times_steps:
                        # Use pre-computed weights
                        rmse_sfc = weighted_rmse_torch_channels(val_output_surface, val_target_surface[:, target_index], weight=self.lat_weight_2d)
                        rmse_pl = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air[:, target_index], weight=self.lat_weight_3d)
                        if self.params.has_diagnostic:
                            rmse_diag = weighted_rmse_torch_channels(val_output_diagnostic, val_target_diagnostic[:, target_index], weight=self.lat_weight_2d)
                            multi_step_rmse[f"valid_lwrmse_diag_{step+1}step"] += torch.mean(rmse_diag)
                            valid_diagnostic_lwrmse[step_idx] += torch.mean(rmse_diag, dim=0)
                            val_output_diagnostic_t[:, step_idx] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()

                        multi_step_rmse[f"valid_lwrmse_sfc_{step+1}step"] += torch.mean(rmse_sfc)
                        multi_step_rmse[f"valid_lwrmse_pl_{step+1}step"] += torch.mean(rmse_pl)

                        valid_surface_lwrmse[step_idx] += torch.mean(rmse_sfc, dim=0)
                        valid_upper_air_lwrmse[step_idx] += torch.mean(rmse_pl, dim=0)

                        val_output_surface_t[:, step_idx] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                        val_output_upper_air_t[:, step_idx] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()

                        if step + 1 == max_lead_time:
                            # ACC prep
                            if self.params.diagnostic_acc or self.params.diagnostic_gif:
                                if self.params.has_diagnostic:
                                    acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc=True, diagnostic_prediction=val_output_diagnostic_acc)
                                else:
                                    acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc=True)
                                acc_prepared = [self.prepare_preds(ds, acc=True) for ds in acc_datasets]
                                acc_predictions.append(self.combine_datasets(acc_prepared))

                                acc_gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface.cpu()).numpy()
                                acc_gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air.cpu()).numpy()
                                if self.params.has_diagnostic:
                                    acc_gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic.cpu()).numpy()
                                    acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc=True, diagnostic_prediction=acc_gt_diagnostic)
                                else:
                                    acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc=True)
                                acc_gt_prepared = [self.prepare_preds(ds, acc=True) for ds in acc_gt_datasets]
                                acc_ground_truths.append(self.combine_datasets(acc_gt_prepared))

                            # Spectra prep
                            lead_time_indices = [lt - 1 for lt in self.params.forecast_lead_times]
                            if self.params.diagnostic_spectra:
                                if self.params.has_diagnostic:
                                    datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc=False, diagnostic_prediction=val_output_diagnostic_t)
                                else:
                                    datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc=False)
                                prepared_datasets = [self.prepare_preds(ds, acc=False) for ds in datasets]
                                all_predictions.append(self.combine_datasets(prepared_datasets))

                                gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface[:, lead_time_indices].cpu()).numpy()
                                gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air[:, lead_time_indices].cpu()).numpy()
                                if self.params.has_diagnostic:
                                    gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic[:, lead_time_indices].cpu()).numpy()
                                    gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc=False, diagnostic_prediction=gt_diagnostic)
                                else:
                                    gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc=False)
                                gt_prepared = [self.prepare_preds(ds, acc=False) for ds in gt_datasets]
                                all_ground_truths.append(self.combine_datasets(gt_prepared))
                        step_idx += 1

                    val_input_surface, val_input_upper_air = val_output_surface, val_output_upper_air

                del val_output_surface, val_output_upper_air, val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air
                valid_steps += 1.
            print("Finished batch validation.")

        if self.params.diagnostic_spectra and len(all_predictions) > 0:
            combined_predictions = xr.concat(all_predictions, dim='time')
            combined_ground_truths = xr.concat(all_ground_truths, dim='time')
        else:
            combined_predictions = combined_ground_truths = None
        print("\nFinished combining predictions and ground truths.")

        if self.params.diagnostic_acc and len(acc_predictions) > 0:
            acc_combined_predictions = xr.concat(acc_predictions, dim='time')
            acc_combined_ground_truths = xr.concat(acc_ground_truths, dim='time')
        else:
            acc_combined_predictions = acc_combined_ground_truths = None
        print("\nFinished combining ACC predictions and ground truths.")

        max_lead_time = max(self.params.forecast_lead_times)
        acc_times_hours = [(lt + 1) * self.params.timedelta_hours for lt in range(max_lead_time)]

        if self.params.diagnostic_acc and acc_combined_predictions is not None:
            print("\nComputing ACC...")
            acc = OrderedDict({
                'Pangu': evaluate_iterative_forecast(
                    acc_combined_predictions, acc_combined_ground_truths,
                    compute_weighted_acc, mean_dims=['lat', 'lon', 'time'], clim=self.climatology
                )
            })
            fig, axs = plot_acc_over_lead_time(acc, acc_times_hours)

        if self.params.diagnostic_spectra and combined_predictions is not None:
            print("\nCalculating power spectrum...")
            k_x_pred, power_spectrum_avg_pred = zonal_averaged_power_spectrum(combined_predictions, time_avg=True)
            k_x_gt, power_spectrum_avg_gt = zonal_averaged_power_spectrum(combined_ground_truths, time_avg=True)
            preds_times = combined_predictions.time.values
            preds_times = preds_times if not isinstance(preds_times, torch.Tensor) else preds_times.cpu().numpy()
            print("\nFinished calculating power spectrum.")

        if self.world_rank == 0:
            if self.params.diagnostic_acc and self.world_rank == 0 and acc_combined_predictions is not None:
                plot_filename = os.path.join(self.output_dir, f"acc_plot_epoch_{self.epoch}.png")
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight'); plt.close(fig)
                print("\nFinished ACC..")
            if self.params.diagnostic_gif and self.world_rank == 0 and acc_combined_predictions is not None:
                print("\nMaking GIF...")
                gif_filename = os.path.join(self.diagnostics_dir, f"geopotential_height_animation_epoch_{self.epoch}.gif")
                make_gif(acc_combined_predictions, acc_combined_ground_truths, self.climatology, "Model Forecast", "geopotential", gif_filename, plev=50000)
                print("\nFinished creating GIF animation.")
            if self.params.diagnostic_spectra and self.world_rank == 0 and combined_predictions is not None:
                print("\nMaking Power Spectrum...")
                path_filename = os.path.join(self.spectra_dir, f"power_spectrum_epoch_{self.epoch}.png")
                preds_times = preds_times if not isinstance(preds_times, torch.Tensor) else preds_times.cpu().numpy()
                self.plot_in_separate_process(power_spectrum_avg_pred, power_spectrum_avg_gt, preds_times, path_filename)
                print("\nFinished Power Spectrum...")

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_surface_lwrmse)
            dist.all_reduce(valid_upper_air_lwrmse)
            if self.params.has_diagnostic:
                dist.all_reduce(valid_diagnostic_lwrmse)
            for loss_tensor in multi_step_losses.values():
                dist.all_reduce(loss_tensor)

        valid_buff[0:-1] = valid_buff[0:-1] / valid_buff[-1]
        valid_surface_lwrmse = valid_surface_lwrmse / valid_buff[-1]
        valid_upper_air_lwrmse = valid_upper_air_lwrmse / valid_buff[-1]
        if self.params.has_diagnostic:
            valid_diagnostic_lwrmse = valid_diagnostic_lwrmse / valid_buff[-1]
        for key in multi_step_losses:
            multi_step_losses[key] /= valid_buff[-1]

        valid_buff_cpu = valid_buff
        diagnostic_logs['epoch'] = self.epoch
        diagnostic_logs['valid_loss'] = float(valid_buff_cpu[0])
        diagnostic_logs['valid_loss_sfc'] = float(valid_buff_cpu[1])
        diagnostic_logs['valid_loss_upper_air'] = float(valid_buff_cpu[2])
        if self.params.has_diagnostic:
            diagnostic_logs['valid_loss_diag'] = float(valid_buff_cpu[3])

        for l, steps in enumerate(lead_times_steps):
            for j, var in enumerate(self.valid_dataset.surface_variables):
                diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = float(valid_surface_lwrmse[l, j] * self.valid_dataset.surface_std[j])
            for j, var in enumerate(self.valid_dataset.upper_air_variables):
                for k, level in enumerate(self.valid_dataset.levels):
                    diagnostic_logs[f'valid_{var}_level{level:.3f}_{steps}step_lwrmse'] = float(valid_upper_air_lwrmse[l, j, k] * self.valid_dataset.upper_air_std[j, k])
            if self.params.has_diagnostic:
                for j, var in enumerate(self.valid_dataset.diagnostic_variables):
                    diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = float(valid_diagnostic_lwrmse[l, j] * self.valid_dataset.diagnostic_std[j])

        for key, value in multi_step_losses.items():
            diagnostic_logs[key] = float(value.item()) if isinstance(value, torch.Tensor) else float(value)

        if self.params.log_to_wandb:
            wandb.log(diagnostic_logs)
            if self.params.diagnostic_acc and self.world_rank == 0 and acc_combined_predictions is not None:
                wandb.log({"ACC_plot": wandb.Image(plot_filename), "epoch": self.epoch})
            if self.params.diagnostic_gif and self.world_rank == 0 and acc_combined_predictions is not None:
                if gif_filename:
                    wandb.log({"Evolution_GIF": wandb.Video(gif_filename), "epoch": self.epoch})
            if self.params.diagnostic_spectra and self.world_rank == 0 and combined_predictions is not None:
                wandb.log({"power_spectrum_plot": wandb.Image(path_filename), "epoch": self.epoch})

        valid_time = time.time() - valid_start
        return valid_time, diagnostic_logs

    def prepare_preds(self, preds, acc = False):
        preds = preds.rename({'time': 'lead_time'})
        preds['time'] = preds.lead_time.values[0:1]
        preds = preds.set_coords('time')
        if acc:
            lead_times = range(1, len(preds.lead_time) + 1)
        else:
            lead_times = self.params['forecast_lead_times']
        preds['lead_time'] = [lt * self.params['timedelta_hours'] for lt in lead_times]
        return preds

    def convert_to_xarray(self, surface_prediction, upper_air_prediction, start_times, params, valid_dataset, acc = True, diagnostic_prediction = None):
        batch_size, time_steps, num_surface_vars, lat, lon = surface_prediction.shape
        datasets = []
        for sample in range(batch_size):
            if acc:
                time_range = [start_times[sample] + timedelta(hours=step * params['timedelta_hours']) for step in range(1, time_steps + 1)]
            else:
                time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            level_coord_name = 'lev' if params.lev == 'lev' else 'plev'
            coordinates = {'time': time_range, level_coord_name: valid_dataset.levels, 'lat': self.params.lat, 'lon': self.params.lon}
            dataset = xr.Dataset(coords=coordinates, attrs=dict(description=f"Prediction from {params.nettype} model run, sample {sample}"))
            for idx, var in enumerate(valid_dataset.surface_variables):
                da = xr.DataArray(data=surface_prediction[sample, :, idx], dims=["time", "lat", "lon"],
                                  coords={'time': time_range, 'lat': dataset.lat.values, 'lon': dataset.lon.values})
                dataset[var] = da
            if type(diagnostic_prediction) is not type(None):
                for idx, var in enumerate(valid_dataset.diagnostic_variables):
                    da = xr.DataArray(data=diagnostic_prediction[sample, :, idx], dims=["time", "lat", "lon"],
                                      coords={'time': time_range, 'lat': dataset.lat.values, 'lon': dataset.lon.values})
                    dataset[var] = da
            for idx, var in enumerate(valid_dataset.upper_air_variables):
                da = xr.DataArray(data=upper_air_prediction[sample, :, idx], dims=["time", level_coord_name, "lat", "lon"], coords=coordinates)
                dataset[var] = da
            datasets.append(dataset)
        return datasets

    def combine_datasets(self, datasets):
        return xr.concat(datasets, dim='time')

    def plot_in_separate_process(self, power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename):
        lead_times_hours = [step * self.params.timedelta_hours for step in self.params.forecast_lead_times]
        p = Process(target=plot_power_spectrum_test, args=(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times_hours))
        p.start(); p.join()

    def log_all_plots_to_wandb(self):
        if self.params.log_to_wandb:
            output_dir = self.spectra_dir
            plot_files = sorted([f for f in os.listdir(output_dir) if f.startswith("power_spectrum_epoch_")])
            for plot_file in plot_files:
                try:
                    epoch = int(plot_file.split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                try:
                    wandb.log({"power_spectrum_plot": wandb.Image(os.path.join(output_dir, plot_file)), "custom_step": epoch})
                except Exception:
                    pass
            shutil.rmtree(output_dir)

    def print_acc(self, acc):
        print("\nACC Results:")
        variables = ["2m_temperature", "temperature", "geopotential", "u_component_of_wind"]
        pressure_levels = [None, 850, 500, 250]
        pressure_levels_pa = [p * 100 if p is not None else None for p in pressure_levels]
        for var, plev, plev_pa in zip(variables, pressure_levels, pressure_levels_pa):
            print(f"\nVariable: {var}" + (f" at {plev} hPa" if plev else " (Surface)"))
            if isinstance(acc['Pangu'], xr.DataArray):
                data = acc['Pangu'][var]
                if plev_pa and 'plev' in data.dims:
                    data = data.sel(plev=plev_pa, method='nearest')
                for lt in self.params.forecast_lead_times:
                    hours = lt * self.params.timedelta_hours
                    acc_value = data.sel(lead_time=lt).values
                    print(f"  Lead time {hours:2d}h (Step {lt:2d}): {acc_value:.4f}")
            else:
                print("  Unexpected data type for ACC score.")

    def cleanup_acc_plots(self):
        output_dir = os.path.join(os.getcwd(), "acc_plots", self.run_uuid)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def cleanup_power_spectrum_plots(self):
        output_dir = os.path.join(os.getcwd(), "spectra_out", self.run_uuid)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def cleanup_gifs(self):
        if os.path.exists(self.diagnostics_dir):
            shutil.rmtree(self.diagnostics_dir)

    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """Load model and training state from existing checkpoint."""
        # PyTorch 2.6+ requires weights_only=False for non-tensor objects in checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except RuntimeError as e:
            if "_orig_mod" in str(e):
                logging.warning("Checkpoint keys mismatch due to torch.compile. Attempting to fix...")
                # Remove 'module.' prefix and add/remove '_orig_mod.' as needed
                new_state_dict = {}
                model_state = checkpoint['model_state']
                
                checkpoint_has_module = any(k.startswith('module.') for k in model_state.keys())
                model_has_orig_mod = any('_orig_mod' in k for k in self.model.state_dict().keys())
                
                for key, value in model_state.items():
                    new_key = key
                    
                    if checkpoint_has_module and model_has_orig_mod:
                        if key.startswith('module.') and not '_orig_mod' in key:
                            new_key = key.replace('module.', 'module._orig_mod.', 1)
                    elif not checkpoint_has_module and model_has_orig_mod:
                        if not key.startswith('module.'):
                            new_key = f'module._orig_mod.{key}'
                    
                    new_state_dict[new_key] = value
                
                self.model.load_state_dict(new_state_dict)
                logging.info("Successfully loaded checkpoint with key remapping")
            else:
                raise e
        
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logging.info(f"Resuming from checkpoint at iters {self.iters}, epoch {self.startEpoch}")

# ------------------------------------
# __main__
# ------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0100', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str)
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    parser.add_argument("--fresh_start", default=False, action="store_true", help="Start training from scratch, ignoring existing checkpoints")
    parser.add_argument("--local-rank", type=int)

    # Perf/engine toggles
    parser.add_argument("--accum-steps", default=1, type=int, help="Gradient accumulation steps per optimizer step")
    parser.add_argument("--max-grad-norm", default=0.0, type=float, help="Clip grad norm if > 0")
    parser.add_argument(
        "--torch-compile",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Enable torch.compile on the model before DDP; pass as --torch-compile or --torch-compile True/False",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead", type=str, help="torch.compile mode")
    parser.add_argument("--compile-max-autotune", action="store_true", help="Use max-autotune mode for torch.compile (slower compile, faster runtime)")
    parser.add_argument("--compile-loss", action="store_true", help="Compile loss functions")
    parser.add_argument("--enable-sdp-flash", action="store_true", help="Enable Flash/mem-efficient/cuDNN SDPA kernels (NOT compatible with Pangu model - causes 'No available kernel' error)")
    parser.add_argument("--ddp-static-graph", action="store_true", help="Enable DDP static graph optimization")
    parser.add_argument("--ddp-bucket-cap-mb", default=200, type=int, help="DDP gradient bucket size in MB (default 200 for H100)")
    parser.add_argument("--ddp-powersgd", action="store_true", help="Enable PowerSGD gradient compression (may affect accuracy)")
    parser.add_argument("--ddp-fp16-compress", action="store_true", help="Enable FP16 gradient compression")
    parser.add_argument("--powersgd-rank", default=1, type=int, help="PowerSGD low-rank size")
    parser.add_argument("--amp-dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"], help="AMP compute dtype (bf16 recommended for H100)")
    parser.add_argument("--log-every-n-steps", default=20, type=int, help="Throttle per-step logging/W&B to every N steps (1 = log every step)")
    parser.add_argument("--metrics-every", default=100, type=int, help="Compute per-step metrics (RMSE etc.) every N steps; 0 = disable")
    parser.add_argument("--grad-stats-every", default=0, type=int, help="Compute grad norm / max every N steps; 0 = disable")
    parser.add_argument("--watch-model", action="store_true", help="Enable wandb.watch(model) (can slow training)")
    parser.add_argument("--watch-log-freq", default=200, type=int, help="W&B watch log frequency in steps")
    parser.add_argument("--fp32-matmul-precision", default="high", choices=["highest", "high", "medium"], help="torch.set_float32_matmul_precision")
    parser.add_argument("--use-channels-last", action="store_true", help="Use channels-last memory format for tensors")

    # Benchmark-specific overrides (do not change default training behavior)
    parser.add_argument("--batch-size-override", default=None, type=int,
                        help="Override batch_size from YAML config (for batch size sweep benchmarks)")
    parser.add_argument("--data-dir-override", default=None, type=str,
                        help="Override data_dir from YAML config (for local vs GPFS benchmarks)")
    parser.add_argument("--no-tf32", action="store_true", default=False,
                        help="Disable TF32 for pure FP32 precision baseline")
    parser.add_argument("--max-steps", default=0, type=int,
                        help="Stop training after this many steps (0 = run full epoch). Used for benchmarking.")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Wire CLI flags to params
    try:
        params['accum_steps'] = max(1, int(getattr(args, 'accum_steps', 1)))
        params['max_grad_norm'] = float(getattr(args, 'max_grad_norm', 0.0))
        params['torch_compile'] = bool(getattr(args, 'torch_compile', False) or getattr(params, 'torch_compile', False))
        params['compile_mode'] = getattr(args, 'compile_mode', 'reduce-overhead')
        params['compile_max_autotune'] = bool(getattr(args, 'compile_max_autotune', False))
        params['compile_loss'] = bool(getattr(args, 'compile_loss', False))
        params['enable_sdp_flash'] = bool(getattr(args, 'enable_sdp_flash', False))
        params['ddp_static_graph'] = bool(getattr(args, 'ddp_static_graph', False))
        params['ddp_bucket_cap_mb'] = int(getattr(args, 'ddp_bucket_cap_mb', 200))
        params['ddp_powersgd'] = bool(getattr(args, 'ddp_powersgd', False))
        params['ddp_fp16_compress'] = bool(getattr(args, 'ddp_fp16_compress', False))
        params['powersgd_rank'] = int(getattr(args, 'powersgd_rank', 1))
        params['amp_dtype'] = getattr(args, 'amp_dtype', 'bf16')
        params['log_every_n_steps'] = int(getattr(args, 'log_every_n_steps', 20))
        params['metrics_every'] = int(getattr(args, 'metrics_every', 100))
        params['grad_stats_every'] = int(getattr(args, 'grad_stats_every', 0))
        params['watch_model'] = bool(getattr(args, 'watch_model', False))
        params['watch_log_freq'] = int(getattr(args, 'watch_log_freq', 200))
        params['fp32_matmul_precision'] = getattr(args, 'fp32_matmul_precision', 'high')
        params['use_channels_last'] = bool(getattr(args, 'use_channels_last', False))
    except Exception as _e:
        logging.warning(f"Failed to wire CLI flags to params: {_e}")

    # Benchmark overrides
    if args.batch_size_override is not None:
        logging.info(f"Overriding batch_size: {params.batch_size} -> {args.batch_size_override}")
        params['batch_size'] = args.batch_size_override
    if args.data_dir_override is not None:
        logging.info(f"Overriding data_dir: {params.data_dir} -> {args.data_dir_override}")
        params['data_dir'] = args.data_dir_override
    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        logging.info("TF32 DISABLED — running pure FP32")

    # Max steps for benchmarking (0 = disabled, run full epoch)
    if args.max_steps > 0:
        params['max_steps'] = args.max_steps
        logging.info(f"Benchmark mode: stopping after {args.max_steps} steps")

    # DataLoader knobs - CRITICAL FOR PERFORMANCE
    if not hasattr(params, 'pin_memory'):
        params['pin_memory'] = True
    if not hasattr(params, 'persistent_workers'):
        params['persistent_workers'] = True
    if not hasattr(params, 'prefetch_factor'):
        params['prefetch_factor'] = 4  # Increase prefetch for faster data loading
    try:
        if not hasattr(params, 'num_workers'):
            # Use SLURM_CPUS_PER_TASK if available, otherwise default to 8
            params['num_workers'] = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    except Exception:
        if not hasattr(params, 'num_workers'):
            params['num_workers'] = 8

    # Enable SDPA Flash/mem-efficient/cuDNN kernels
    if getattr(params, 'enable_sdp_flash', False) and torch.cuda.is_available():
        try:
            # WARNING: This may not work with all model architectures (e.g., Pangu's 5D attention)
            # The model's attention uses 5D tensors and custom masks which aren't compatible
            # with fused SDPA kernels. Comment out or set to False if you get kernel errors.
            # torch.backends.cuda.enable_flash_sdp(True)
            # torch.backends.cuda.enable_mem_efficient_sdp(True)
            # torch.backends.cuda.enable_math_sdp(True)
            # torch.backends.cuda.enable_cudnn_sdp(True)
            # logging.info("Enabled Flash, mem-efficient and cuDNN SDPA kernels")
            logging.warning("SDPA kernel optimization disabled - incompatible with Pangu's 5D attention mechanism")
            logging.warning("The model uses 5D tensors in attention which require the default math kernel")
        except Exception as e:
            logging.warning(f"Could not enable SDPA kernels: {e}")

    # Matmul precision (override earlier default)
    try:
        torch.set_float32_matmul_precision(params.fp32_matmul_precision)
    except Exception as _e:
        logging.info(f"matmul precision set skipped: {_e}")

    print("This is the starting point f")
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.run_iter
    params['run_iter'] = args.run_iter
    if hasattr(params, 'diagnostic_variables'):
        params['has_diagnostic'] = True if len(params.diagnostic_variables) > 0 else False
    else:
        params['has_diagnostic'] = False

    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1

    if hasattr(params, "wandb_offline"):
        if params.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

    print('World size from OS: %d' % int(os.environ.get('WORLD_SIZE', '1')))
    print('World size from Cuda: %d' % torch.cuda.device_count())

    # Check GPU memory banner (rank-local)
    try:
        print(torch.cuda.get_device_name(0))
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    except Exception:
        pass

    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE']); print(params['world_size'])
    else:
        params['world_size'] = torch.cuda.device_count(); print(params['world_size'])

    if params['world_size'] > 1:
        if 'derecho' in str(Path(__file__)):
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Experiment dir
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir); os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path = 'training_checkpoints/ckpt.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path'] = os.path.join(expDir, ckpt_path)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

    checkpoint_exists = os.path.isfile(params.checkpoint_path)

    if getattr(params, 'fresh_start', False) or args.fresh_start:
        params['resuming'] = False
        if checkpoint_exists and world_rank == 0:
            logging.info("Fresh start requested. Ignoring existing checkpoint.")
    elif checkpoint_exists:
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        params['resuming'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")

    params['local_rank'] = local_rank

    # Precision/engine banner
    if getattr(params, 'use_transformer_engine', False):
        print("Using Transformer Engine")
    else:
        print("Using PyTorch native")

    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, log_file))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.setup_model()
    trainer.train()
    logging.info('DONE ---- rank %d', world_rank)
