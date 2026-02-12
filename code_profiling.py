# ----------------------------------------------------------------------------------
# train_optimized.py — Pangu S2S trainer highly optimized for 4x H100 GPUs
# With integrated PyTorch Profiler support for performance analysis
# ----------------------------------------------------------------------------------

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
import time
from multiprocessing import Process
import shutil
import uuid
import os
import numpy as np
import argparse
import xarray as xr
import logging
import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import contextlib
from typing import Iterator, Tuple, Optional
import functools

# Profiler imports
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# ------------------------------------
# H100-specific Environment Tuning (MUST be set before any CUDA calls)
# ------------------------------------
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "0")
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")
os.environ.setdefault("NCCL_P2P_LEVEL", "5")
os.environ.setdefault("NCCL_SOCKET_NTHREADS", "8")
os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
os.environ.setdefault("NCCL_BUFFSIZE", "16777216")
os.environ.setdefault("NCCL_NTHREADS", "512")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", 
    "expandable_segments:True,garbage_collection_threshold:0.8")


def _str_to_bool(val):
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    if val in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if val in {"0", "false", "no", "n", "f", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")

from utils import logging_utils
from utils.power_spectrum import (
    plot_acc_over_lead_time, zonal_averaged_power_spectrum,
    make_gif, plot_power_spectrum_test,
)
from utils.losses import (
    Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,
    Latitude_weighted_CRPSLoss, Kl_divergence_gaussians
)

# ------------------------------------
# Constants - Tuned for H100
# ------------------------------------
DEFAULT_LOSS_WEIGHTS = {'surface': 0.25, 'diagnostic': 0.25, 'upper_air': 1.0}

H100_DEFAULTS = {
    'ddp_bucket_cap_mb': 256,
    'prefetch_factor': 4,
    'num_workers': 8,
    'log_every_n_steps': 50,
    'metrics_every': 200,
    'grad_stats_every': 0,
}

# Profiling defaults
PROFILING_DEFAULTS = {
    'wait_steps': 5,       # Steps to skip before profiling (warmup)
    'warmup_steps': 3,     # Steps for profiler warmup
    'active_steps': 10,    # Steps to actively profile
    'repeat': 1,           # Number of profiling cycles
    'profile_memory': True,
    'with_stack': True,
    'with_flops': True,
    'with_modules': True,
}

from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
from utils.integrate import Integrator
from networks.pangu import PanguModel_Plasim

# ------------------------------------
# Feature detection
# ------------------------------------
def _is_torch_compile_available() -> bool:
    has_compile = hasattr(torch, "compile")
    dynamo_ok = getattr(torch._dynamo, "is_dynamo_supported", lambda: False)()
    return bool(has_compile and dynamo_ok)

# ------------------------------------
# Global torch defaults
# ------------------------------------
logging_utils.config_logger()
torch._dynamo.config.optimize_ddp = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cudnn, 'benchmark_limit'):
        torch.backends.cudnn.benchmark_limit = 0

logging.info("Torch version: {}".format(torch.__version__))

dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=30))
world_rank = dist.get_rank()
print(f"World rank: {world_rank}")

# ------------------------------------
# Metrics helpers
# ------------------------------------
def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted/torch.sum(lat_weights_unweighted)

def weighted_rmse_torch_channels(pred, target, weight):
    return torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))

def weighted_rmse_torch_3D(pred, target, weight):
    return torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1, -2)))

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5

def to_ensemble_batch(data, ens_members):
    return (data.unsqueeze(1) * torch.ones(1, ens_members, *data.shape[1:], device=data.device)).flatten(0, 1)


# ------------------------------------
# Double-Buffered CUDA Prefetcher
# ------------------------------------
class DoubleBufferPrefetcher:
    def __init__(self, loader: Iterator, device: torch.device, has_diagnostic: bool = False):
        self.loader = iter(loader)
        self.device = device
        self.has_diagnostic = has_diagnostic
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_data = None
        self.buffered_data = None
        self._preload()
        self._preload()
    
    def _preload(self):
        self.buffered_data = self.next_data
        try:
            data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = tuple(
                    t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in data
                )
        else:
            self.next_data = data
    
    def __next__(self) -> Tuple:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        data = self.buffered_data
        if data is None:
            raise StopIteration
        
        if self.stream is not None:
            for t in data:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return data
    
    def __iter__(self):
        return self


# ------------------------------------
# Profiler Manager
# ------------------------------------
class ProfilerManager:
    """
    Manages PyTorch Profiler for performance analysis.
    Outputs traces compatible with TensorBoard and Chrome Tracing.
    """
    def __init__(self, params, output_dir: str, rank: int = 0):
        self.params = params
        self.rank = rank
        self.enabled = getattr(params, 'profiling', False)
        self.output_dir = os.path.join(output_dir, 'profiler_traces')
        
        if self.enabled and rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Profiler enabled. Traces will be saved to: {self.output_dir}")
        
        # Profiling configuration
        self.wait_steps = getattr(params, 'profile_wait_steps', PROFILING_DEFAULTS['wait_steps'])
        self.warmup_steps = getattr(params, 'profile_warmup_steps', PROFILING_DEFAULTS['warmup_steps'])
        self.active_steps = getattr(params, 'profile_active_steps', PROFILING_DEFAULTS['active_steps'])
        self.repeat = getattr(params, 'profile_repeat', PROFILING_DEFAULTS['repeat'])
        self.profile_memory = getattr(params, 'profile_memory', PROFILING_DEFAULTS['profile_memory'])
        self.with_stack = getattr(params, 'profile_with_stack', PROFILING_DEFAULTS['with_stack'])
        self.with_flops = getattr(params, 'profile_with_flops', PROFILING_DEFAULTS['with_flops'])
        self.with_modules = getattr(params, 'profile_with_modules', PROFILING_DEFAULTS['with_modules'])
        
        self.profiler = None
        self.step_count = 0
    
    def get_schedule(self):
        """Create profiler schedule."""
        return schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat
        )
    
    def get_activities(self):
        """Get list of activities to profile."""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities
    
    def trace_handler(self, prof):
        """Handle trace export after each profiling cycle."""
        if self.rank != 0:
            return
        
        # Export Chrome trace
        trace_file = os.path.join(
            self.output_dir, 
            f"trace_rank{self.rank}_step{self.step_count}.json"
        )
        prof.export_chrome_trace(trace_file)
        logging.info(f"Exported Chrome trace to: {trace_file}")
        
        # Export stacks for flame graph (if enabled)
        if self.with_stack:
            stacks_file = os.path.join(
                self.output_dir,
                f"stacks_rank{self.rank}_step{self.step_count}.txt"
            )
            prof.export_stacks(stacks_file, "self_cuda_time_total")
            logging.info(f"Exported stacks to: {stacks_file}")
        
        # Print summary tables
        if self.rank == 0:
            print("\n" + "="*80)
            print("PROFILER SUMMARY - CUDA Time")
            print("="*80)
            print(prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=20
            ))
            
            print("\n" + "="*80)
            print("PROFILER SUMMARY - CPU Time")
            print("="*80)
            print(prof.key_averages().table(
                sort_by="cpu_time_total", 
                row_limit=20
            ))
            
            if self.profile_memory:
                print("\n" + "="*80)
                print("PROFILER SUMMARY - Memory")
                print("="*80)
                print(prof.key_averages().table(
                    sort_by="self_cuda_memory_usage", 
                    row_limit=20
                ))
            
            # Group by input shape (useful for finding shape-related issues)
            print("\n" + "="*80)
            print("PROFILER SUMMARY - Grouped by Input Shape")
            print("="*80)
            print(prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total",
                row_limit=15
            ))
    
    def create_profiler(self):
        """Create and return the profiler context manager."""
        if not self.enabled:
            return contextlib.nullcontext()
        
        return profile(
            activities=self.get_activities(),
            schedule=self.get_schedule(),
            on_trace_ready=self.trace_handler,
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )
    
    def step(self):
        """Call after each training step when profiling is active."""
        self.step_count += 1
        if self.profiler is not None:
            self.profiler.step()
    
    def should_stop_training(self) -> bool:
        """Check if we should stop training after profiling is complete."""
        if not self.enabled:
            return False
        
        total_profile_steps = (self.wait_steps + self.warmup_steps + self.active_steps) * self.repeat
        return self.step_count >= total_profile_steps
    
    def get_total_steps(self) -> int:
        """Get total number of steps needed for profiling."""
        return (self.wait_steps + self.warmup_steps + self.active_steps) * self.repeat


# ------------------------------------
# Memory Profiler Utilities
# ------------------------------------
class MemoryProfiler:
    """Utility class for tracking GPU memory usage."""
    
    def __init__(self, enabled: bool = False, rank: int = 0):
        self.enabled = enabled and torch.cuda.is_available()
        self.rank = rank
        self.snapshots = []
    
    def snapshot(self, label: str):
        """Take a memory snapshot with a label."""
        if not self.enabled or self.rank != 0:
            return
        
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        self.snapshots.append({
            'label': label,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'timestamp': time.time()
        })
        
        logging.info(f"[Memory] {label}: Allocated={allocated:.2f}GB, "
                    f"Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
    
    def report(self):
        """Print memory usage report."""
        if not self.enabled or self.rank != 0 or not self.snapshots:
            return
        
        print("\n" + "="*80)
        print("MEMORY USAGE REPORT")
        print("="*80)
        print(f"{'Label':<40} {'Allocated (GB)':<15} {'Reserved (GB)':<15} {'Max (GB)':<15}")
        print("-"*80)
        for snap in self.snapshots:
            print(f"{snap['label']:<40} {snap['allocated_gb']:<15.2f} "
                  f"{snap['reserved_gb']:<15.2f} {snap['max_allocated_gb']:<15.2f}")
        print("="*80 + "\n")


# ------------------------------------
# Trainer
# ------------------------------------
class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.compile_available = _is_torch_compile_available()
        
        if getattr(params, 'enable_sdp_flash', False):
            logging.warning("⚠️ --enable-sdp-flash disabled - incompatible with Pangu 3D attention")
            params.enable_sdp_flash = False
        
        self.iters = 0
        self.startEpoch = 0
        self.epoch = self.startEpoch
        self.early_stop_epoch = params.get('early_stop_epoch', None)
        if self.early_stop_epoch:
            self.early_stop_epoch -= 1
        self.run_uuid = str(uuid.uuid4())
        
        # Throttling settings
        self.log_every = max(1, int(getattr(params, "log_every_n_steps", H100_DEFAULTS['log_every_n_steps'])))
        self.metrics_every = int(getattr(params, "metrics_every", H100_DEFAULTS['metrics_every']))
        self.grad_stats_every = int(getattr(params, "grad_stats_every", H100_DEFAULTS['grad_stats_every']))
        
        self.check_land_ocean_variables()
        self.get_dataset()
        self.spectra_dir, self.diagnostics_dir, self.output_dir = self.create_dirs(self.run_uuid)

        # Initialize profilers
        self.profiler_manager = ProfilerManager(params, params.experiment_dir, world_rank)
        self.memory_profiler = MemoryProfiler(
            enabled=getattr(params, 'profile_memory', False),
            rank=world_rank
        )

        # AMP dtype
        amp_choice = getattr(self.params, "amp_dtype", "bf16")
        if amp_choice == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            logging.info("Using BF16 for H100 tensor cores")
        elif amp_choice == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Init wandb
        if params.log_to_wandb:
            wandb.init(
                config=params, name=f'{params.name}-{params.run_iter}',
                entity=params.entity, group=params.group,
                project=params.project, resume="allow" if params.resuming else "never"
            )
        self.init_wandb(self.params)
        logging.info('Params %s', params)

    def setup_model(self):
        self.memory_profiler.snapshot("Before model creation")
        
        self.mask_bool, self.land_mask = self.get_land_mask_bool()
        
        with record_function("model_build"):
            self.model = self._build_model()
        self.memory_profiler.snapshot("After model creation")
        
        with record_function("model_compile"):
            self.model = self._compile_model(self.model)
        self.memory_profiler.snapshot("After model compilation")
        
        with record_function("ddp_wrap"):
            self.model = self._wrap_ddp(self.model)
        self.memory_profiler.snapshot("After DDP wrapping")
        
        self.scaler = GradScaler(enabled=(torch.cuda.is_available() and self.amp_dtype == torch.float16))
        self.optimizer = self.get_optimizer()

        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path)
            logging.info("Resuming from checkpoint: %s", params.checkpoint_path)

        self.setup_scheduler()
        self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic = self.setup_loss_fun()
        
        self.memory_profiler.snapshot("After full setup")
        self.memory_profiler.report()

    def _build_model(self):
        if self.params.nettype == 'pangu_plasim':
            if self.params.predict_delta:
                model = PanguModel_Plasim(self.params, land_mask=self.land_mask).to(self.device)
                self.integrator = Integrator(
                    self.params,
                    surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                    surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                    upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                    upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)
                ).to(self.device)
            else:
                mask_fill = getattr(self.params, 'mask_fill', self.train_datasets[0].mask_fill)
                model = PanguModel_Plasim(self.params, land_mask=self.land_mask, mask_fill=mask_fill).to(self.device)
        else:
            raise Exception("not implemented")
        
        if getattr(self.params, 'gradient_checkpointing', False):
            if hasattr(model, 'set_gradient_checkpointing'):
                model.set_gradient_checkpointing(True)
                logging.info("Gradient checkpointing enabled")
        
        return model

    def _compile_model(self, model):
        should_compile = bool(getattr(self.params, "torch_compile", True))
        
        # Disable compilation during profiling for clearer traces
        if getattr(self.params, 'profiling', False) and getattr(self.params, 'profile_disable_compile', False):
            logging.info("torch.compile disabled for profiling (clearer traces)")
            return model
        
        if not should_compile:
            return model
        if not self.compile_available:
            logging.warning("torch.compile requested but not available")
            return model
        
        compile_mode = getattr(self.params, "compile_mode", "reduce-overhead")
        if getattr(self.params, "compile_max_autotune", False):
            compile_mode = "max-autotune"
        
        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            logging.info(f"torch.compile enabled (mode={compile_mode}) BEFORE DDP")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}")
        
        return model

    def _wrap_ddp(self, model):
        if not dist.is_initialized():
            return model
        
        ddp_static = bool(getattr(self.params, "ddp_static_graph", True))
        bucket_cap = int(getattr(self.params, "ddp_bucket_cap_mb", H100_DEFAULTS['ddp_bucket_cap_mb']))
        
        model = DistributedDataParallel(
            model,
            device_ids=[self.params.local_rank],
            output_device=[self.params.local_rank],
            find_unused_parameters=not ddp_static,
            static_graph=ddp_static,
            bucket_cap_mb=bucket_cap,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        
        if getattr(self.params, 'ddp_powersgd', False):
            self._setup_powersgd(model)
        elif getattr(self.params, 'ddp_fp16_compress', False):
            self._setup_fp16_compress(model)
        
        return model

    def _setup_powersgd(self, model):
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
            state = powerSGD.PowerSGDState(
                process_group=dist.group.WORLD,
                warm_start=True,
                use_error_feedback=True,
                start_powerSGD_iter=10,
                matrix_approximation_rank=int(getattr(self.params, 'powersgd_rank', 1)),
            )
            model.register_comm_hook(state, powerSGD.powerSGD_hook)
            logging.info("PowerSGD gradient compression enabled")
        except Exception as e:
            logging.warning(f"PowerSGD failed: {e}")

    def _setup_fp16_compress(self, model):
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
            logging.info("FP16 gradient compression enabled")
        except Exception as e:
            logging.warning(f"FP16 compression failed: {e}")

    def check_land_ocean_variables(self) -> None:
        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(self.params, 'land_variables') and len(self.params.land_variables) > 0:
            self.has_land = True
        else:
            self.params['land_variables'] = []
        if hasattr(self.params, 'ocean_variables') and len(self.params.ocean_variables) > 0:
            self.has_ocean = True
        else:
            self.params['ocean_variables'] = []
        if hasattr(self.params, 'mask_output'):
            self.mask_output = self.params.mask_output

    def create_dirs(self, run_uuid: int) -> tuple:
        for dir_name in ["spectra_out", "gif_out", "acc_plots"]:
            os.makedirs(os.path.join(os.getcwd(), dir_name), exist_ok=True)
        spectra_dir = os.path.join(os.getcwd(), "spectra_out", self.run_uuid)
        diagnostics_dir = os.path.join(os.getcwd(), "gif_out", self.run_uuid)
        output_dir = os.path.join(os.getcwd(), "acc_plots", self.run_uuid)
        if world_rank == 0:
            for d in [spectra_dir, diagnostics_dir, output_dir]:
                os.makedirs(d, exist_ok=True)
        return spectra_dir, diagnostics_dir, output_dir

    def get_dataset(self):
        logging.info('rank %d, begin data loader init', self.world_rank)
        
        dl_kwargs = {
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': H100_DEFAULTS['prefetch_factor'],
            'num_workers': int(os.environ.get("SLURM_CPUS_PER_TASK", H100_DEFAULTS['num_workers'])),
        }
        
        if self.params.train_year_to_year:
            self.train_data_loaders, self.train_datasets, self.train_samplers = [], [], []
            for year_start in range(params.train_year_start, params.train_year_end):
                loader, dataset, sampler = get_data_loader(
                    params, params.data_dir, dist.is_initialized(),
                    year_start=year_start, year_end=year_start + 1, train=True,
                    **dl_kwargs
                )
                self.train_data_loaders.append(loader)
                self.train_datasets.append(dataset)
                self.train_samplers.append(sampler)
        else:
            loader, dataset, sampler = get_data_loader(
                params, params.data_dir, dist.is_initialized(),
                year_start=params.train_year_start, year_end=params.train_year_end,
                train=True, **dl_kwargs
            )
            self.train_data_loaders = [loader]
            self.train_datasets = [dataset]
            self.train_samplers = [sampler]

        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params, params.data_dir, dist.is_initialized(),
            year_start=params.val_year_start, year_end=params.val_year_end,
            train=False, num_inferences=params.num_inferences, validate=True
        )

        self.constant_boundary_data = (
            self.train_datasets[0].constant_binary_data.unsqueeze(0)
            if hasattr(self.train_datasets[0], 'constant_binary_data')
            else self.train_datasets[0].constant_boundary_data.unsqueeze(0)
        ) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        
        if params.num_ensemble_members > 1:
            self.constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, params.num_ensemble_members)

        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path).rename({'time': 'dayofyear'})
        self.lat_t = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)
        
        with torch.inference_mode():
            _lat_w = latitude_weighting_factor_torch(self.lat_t)
            self.lat_weight_2d = _lat_w.view(1, 1, -1, 1)
            self.lat_weight_3d = _lat_w.view(1, 1, 1, -1, 1)

        if world_rank == 0:
            logging.info('Data loader initialized')

    def init_wandb(self, params: dict):
        if not params.log_to_wandb:
            return
        wandb.define_metric("epoch")
        epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air']
        for l, steps in enumerate(params.forecast_lead_times):
            epoch_metrics.extend([
                f"valid_lwrmse_sfc_{steps}step",
                f"valid_lwrmse_pl_{steps}step",
                f"valid_loss_{steps}step",
            ])
        for metric in epoch_metrics:
            wandb.define_metric(metric, step_metric="epoch")

    def get_land_mask_bool(self) -> tuple:
        mask_bool, land_mask = [], []
        if self.params.nettype == 'pangu_plasim':
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = self.train_datasets[0].land_mask.detach().clone().to(self.device)
                for var in self.params.surface_variables:
                    if var in self.params.land_variables:
                        mask_bool.append(land_mask.clone().to(torch.bool))
                    elif var in self.params.ocean_variables:
                        mask_bool.append(~land_mask.clone().to(torch.bool))
                    else:
                        mask_bool.append(torch.ones(land_mask.shape, device=self.device, dtype=torch.bool))
                mask_bool = torch.stack(mask_bool)
            else:
                land_mask = None
        return mask_bool, land_mask

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_optimizer(self):
        try:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
                fused=True
            )
            logging.info("Using fused Adam optimizer")
        except (TypeError, RuntimeError):
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
                foreach=True
            )
        return self.optimizer

    def _expand_for_ensemble(self, *tensors) -> tuple:
        if self.params.num_ensemble_members <= 1:
            return tensors
        return tuple(to_ensemble_batch(t, self.params.num_ensemble_members) for t in tensors)

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
            pct_start = getattr(self.params, 'oc_pct_start', 0.3)
            div_factor = getattr(self.params, 'oc_div_factor', 25)
            final_div_factor = getattr(self.params, 'oc_final_div_factor', 1e4)
            last_epoch = (self.startEpoch - 1) * steps_per_epoch if self.startEpoch >= 1 else -1
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.params.lr,
                epochs=self.params.max_epochs, steps_per_epoch=steps_per_epoch,
                last_epoch=last_epoch,
                pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
            )
        else:
            self.scheduler = None

    def setup_loss_fun(self):
        loss_pl = loss_sfc = loss_diag = 0
        self.loss_vae = 0
        
        if self.params.vae_loss:
            self.loss_vae = Kl_divergence_gaussians()
        
        lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
        
        if self.params.loss == 'l1':
            loss_pl = torch.nn.L1Loss()
            loss_sfc = Masked_L1Loss(self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else torch.nn.L1Loss()
            if self.params.has_diagnostic:
                loss_diag = torch.nn.L1Loss()
        elif self.params.loss == 'l2':
            loss_pl = torch.nn.MSELoss()
            loss_sfc = Masked_MSELoss(self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else torch.nn.MSELoss()
            if self.params.has_diagnostic:
                loss_diag = torch.nn.MSELoss()
        elif self.params.loss == 'weightedl1':
            loss_pl = Latitude_weighted_L1Loss(lat)
            loss_sfc = Latitude_weighted_masked_L1Loss(lat, self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else Latitude_weighted_L1Loss(lat)
            if self.params.has_diagnostic:
                loss_diag = Latitude_weighted_L1Loss(lat)
        elif self.params.loss == 'weightedl2':
            loss_pl = Latitude_weighted_MSELoss(lat)
            loss_sfc = Latitude_weighted_masked_MSELoss(lat, self.mask_bool) if (self.has_land or self.has_ocean) and self.mask_output else Latitude_weighted_MSELoss(lat)
            if self.params.has_diagnostic:
                loss_diag = Latitude_weighted_MSELoss(lat)
        elif self.params.loss == 'weightedCRPS':
            loss_pl = Latitude_weighted_CRPSLoss(lat, params.num_ensemble_members)
            loss_sfc = Latitude_weighted_CRPSLoss(lat, params.num_ensemble_members, self.mask_bool) if self.has_land or self.has_ocean else Latitude_weighted_CRPSLoss(lat, params.num_ensemble_members)
            if self.params.has_diagnostic:
                loss_diag = Latitude_weighted_CRPSLoss(lat, params.num_ensemble_members)
        else:
            raise NotImplementedError(f"Loss {self.params.loss} not implemented")
        
        return loss_pl, loss_sfc, loss_diag

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")
        
        # Check if we're in profiling mode
        if self.profiler_manager.enabled:
            logging.info(f"PROFILING MODE: Will run for {self.profiler_manager.get_total_steps()} steps then exit")
            self._train_with_profiling()
            return
        
        # Normal training loop
        best_valid_loss = 1.e6
        early_stopping_counter = 0

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if world_rank == 0:
                logging.info(f'Starting epoch {epoch + 1}/{self.params.max_epochs}')

            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                logging.info(f'Early stop epoch reached. Terminating.')
                break

            if dist.is_initialized():
                for sampler in self.train_samplers:
                    sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()

            if valid_logs['valid_loss'] <= best_valid_loss:
                best_valid_loss = valid_logs['valid_loss']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if self.world_rank == 0 and self.params.save_checkpoint:
                self.save_checkpoint(self.params.checkpoint_path)
                if valid_logs['valid_loss'] <= best_valid_loss:
                    self.save_checkpoint(self.params.best_checkpoint_path)

            if world_rank == 0:
                self.log_wandb_epoch(epoch)
                logging.info(f'Epoch {epoch+1}: train_loss={train_logs["train_loss"]:.4f}, '
                           f'valid_loss={valid_logs["valid_loss"]:.4f}, time={time.time()-start:.1f}s')

            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                logging.info('Early stopping triggered.')
                break

    def _train_with_profiling(self):
        """Training loop specifically for profiling."""
        self.epoch += 1
        self.model.train()
        
        if dist.is_initialized():
            for sampler in self.train_samplers:
                sampler.set_epoch(self.epoch)
        
        total_steps = self.profiler_manager.get_total_steps()
        logging.info(f"Starting profiling run for {total_steps} steps...")
        
        # Reset memory stats for clean profiling
        self.memory_profiler.reset_peak_stats()
        self.memory_profiler.snapshot("Before profiling loop")
        
        _accum_steps = max(1, int(getattr(self.params, "accum_steps", 1)))
        step = 0
        
        # Create profiler
        with self.profiler_manager.create_profiler() as prof:
            self.profiler_manager.profiler = prof
            
            for year_idx, train_data_loader in enumerate(self.train_data_loaders):
                current_dataset = self.train_datasets[year_idx]
                data_iter = DoubleBufferPrefetcher(train_data_loader, self.device, self.params.has_diagnostic)
                
                for i, data in enumerate(data_iter):
                    if step >= total_steps:
                        break
                    
                    with record_function("data_preparation"):
                        inp_sfc, inp_ua, tgt_sfc, tgt_ua, tgt_diag, vary_bd = self._prepare_inputs_prefetched(data)
                    
                    if (i % _accum_steps) == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    
                    with record_function("forward_backward"):
                        with record_function("forward"):
                            out_sfc, out_ua, out_diag, loss_sfc, loss_pl, loss_diag, loss_vae, loss = self.cal_loss(
                                inp_sfc, self.constant_boundary_data, vary_bd, inp_ua,
                                tgt_diag, tgt_sfc, tgt_ua
                            )
                        
                        _is_last = ((i + 1) % _accum_steps == 0) or (i == len(train_data_loader) - 1)
                        sync_ctx = (self.model.no_sync() if dist.is_initialized() and 
                                   hasattr(self.model, "no_sync") and not _is_last 
                                   else contextlib.nullcontext())
                        
                        with record_function("backward"):
                            with sync_ctx:
                                self.scaler.scale(loss / _accum_steps).backward()
                    
                    if _is_last:
                        with record_function("optimizer_step"):
                            max_grad = float(getattr(self.params, 'max_grad_norm', 0.0))
                            if max_grad > 0.0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            if self.params.scheduler == 'OneCycleLR':
                                self.scheduler.step()
                    
                    # Step the profiler
                    self.profiler_manager.step()
                    step += 1
                    
                    if world_rank == 0 and step % 5 == 0:
                        logging.info(f"Profiling step {step}/{total_steps}, loss={loss.item():.4f}")
                    
                    if step >= total_steps:
                        break
                
                if step >= total_steps:
                    break
        
        self.memory_profiler.snapshot("After profiling loop")
        self.memory_profiler.report()
        
        logging.info("Profiling complete. Check the profiler_traces directory for results.")
        logging.info(f"Trace files: {self.profiler_manager.output_dir}")
        
        # Print usage instructions
        if world_rank == 0:
            print("\n" + "="*80)
            print("PROFILING COMPLETE - HOW TO VIEW RESULTS")
            print("="*80)
            print(f"\n1. Chrome Trace Viewer:")
            print(f"   - Open chrome://tracing in Chrome browser")
            print(f"   - Load: {self.profiler_manager.output_dir}/trace_rank0_*.json")
            print(f"\n2. TensorBoard:")
            print(f"   tensorboard --logdir={self.profiler_manager.output_dir}")
            print(f"\n3. Flame Graph (if stacks exported):")
            print(f"   pip install flameprof")
            print(f"   flameprof {self.profiler_manager.output_dir}/stacks_rank0_*.txt > flamegraph.svg")
            print("="*80 + "\n")

    def log_wandb_epoch(self, epoch: int):
        if self.params.log_to_wandb:
            lr = self.optimizer.param_groups[0]['lr']
            wandb.log({'lr': lr, 'epoch': self.epoch})

    def train_one_epoch(self):
        self.epoch += 1
        tr_time, data_time = 0.0, 0.0
        total_iterations = sum(len(loader) for loader in self.train_data_loaders)
        diagnostic_logs = {}
        loss = 0.0

        self.model.train()
        pbar = tqdm(total=total_iterations, disable=(self.world_rank != 0),
                   bar_format='{l_bar}{bar:20}{r_bar}')
        
        running_loss = 0.0
        _accum_steps = max(1, int(getattr(self.params, "accum_steps", 1)))

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            current_dataset = self.train_datasets[year_idx]
            data_iter = DoubleBufferPrefetcher(train_data_loader, self.device, self.params.has_diagnostic)

            for i, data in enumerate(data_iter):
                if self.params.mode == "test" and i >= self.params.test_iterations:
                    break

                self.iters += 1
                data_start = time.time()
                inp_sfc, inp_ua, tgt_sfc, tgt_ua, tgt_diag, vary_bd = self._prepare_inputs_prefetched(data)
                data_time += time.time() - data_start

                tr_start = time.time()
                
                if (i % _accum_steps) == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                out_sfc, out_ua, out_diag, loss_sfc, loss_pl, loss_diag, loss_vae, loss = self.cal_loss(
                    inp_sfc, self.constant_boundary_data, vary_bd, inp_ua,
                    tgt_diag, tgt_sfc, tgt_ua
                )

                _is_last = ((i + 1) % _accum_steps == 0) or (i == len(train_data_loader) - 1)
                sync_ctx = (self.model.no_sync() if dist.is_initialized() and 
                           hasattr(self.model, "no_sync") and not _is_last 
                           else contextlib.nullcontext())
                
                with sync_ctx:
                    self.scaler.scale(loss / _accum_steps).backward()
                
                if _is_last:
                    max_grad = float(getattr(self.params, 'max_grad_norm', 0.0))
                    if max_grad > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.params.scheduler == 'OneCycleLR':
                        self.scheduler.step()

                tr_time += time.time() - tr_start
                running_loss += loss.item()

                if self.world_rank == 0 and (self.iters % self.log_every == 0):
                    avg_loss = running_loss / self.log_every
                    pbar.set_description(f"Loss: {avg_loss:.4f}")
                    pbar.update(self.log_every)
                    running_loss = 0.0
                    
                    if self.params.log_to_wandb:
                        wandb.log({'train_batch_loss': avg_loss}, step=self.iters)

                if self.metrics_every > 0 and (i % self.metrics_every == 0):
                    with torch.inference_mode():
                        diagnostic_logs = self._compute_train_metrics(
                            out_sfc, out_ua, tgt_sfc, tgt_ua, inp_sfc, inp_ua,
                            current_dataset, loss, loss_sfc, loss_pl
                        )

        pbar.close()
        
        train_loss = diagnostic_logs.get('train_batch_loss', loss.item())
        if dist.is_initialized():
            loss_tensor = torch.tensor(train_loss, device=self.device)
            dist.all_reduce(loss_tensor)
            train_loss = loss_tensor.item() / dist.get_world_size()
        
        return tr_time, data_time, {'train_loss': train_loss}

    def _prepare_inputs_prefetched(self, data: Tuple) -> Tuple:
        if self.params.has_diagnostic:
            inp_sfc, inp_ua, tgt_sfc, tgt_ua, tgt_diag, vary_bd = data[:6]
        else:
            inp_sfc, inp_ua, tgt_sfc, tgt_ua, vary_bd = data[:5]
            tgt_diag = 0
        
        inp_sfc = inp_sfc.float()
        inp_ua = inp_ua.float()
        tgt_sfc = tgt_sfc.float()
        tgt_ua = tgt_ua.float()
        vary_bd = vary_bd.float()
        if isinstance(tgt_diag, torch.Tensor):
            tgt_diag = tgt_diag.float()
        
        if self.params.num_ensemble_members > 1:
            inp_sfc, inp_ua, tgt_sfc, tgt_ua, vary_bd = self._expand_for_ensemble(
                inp_sfc, inp_ua, tgt_sfc, tgt_ua, vary_bd
            )
            if isinstance(tgt_diag, torch.Tensor):
                tgt_diag = to_ensemble_batch(tgt_diag, self.params.num_ensemble_members)
        
        return inp_sfc, inp_ua, tgt_sfc, tgt_ua, tgt_diag, vary_bd

    def cal_loss(self, inp_sfc, const_bd, vary_bd, inp_ua, tgt_diag, tgt_sfc, tgt_ua):
        out_sfc = out_ua = out_diag = 0
        loss = loss_diag = loss_pl = loss_sfc = loss_vae = 0
        
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=torch.cuda.is_available()):
            if self.params.has_diagnostic:
                out_sfc, out_ua, out_diag, mu, sigma, mu2, sigma2 = self.model(
                    inp_sfc, const_bd, vary_bd, inp_ua, tgt_sfc, tgt_ua, train=True
                )
                loss_diag = self.loss_obj_diagnostic(out_diag, tgt_diag)
            else:
                out_sfc, out_ua, mu, sigma, mu2, sigma2 = self.model(
                    inp_sfc, const_bd, vary_bd, inp_ua, tgt_sfc, tgt_ua, train=True
                )

            loss_sfc = self.loss_obj_sfc(out_sfc, tgt_sfc)
            loss_pl = self.loss_obj_pl(out_ua, tgt_ua)

            if self.params.has_diagnostic:
                loss = (loss_sfc * DEFAULT_LOSS_WEIGHTS['surface'] + 
                       loss_diag * DEFAULT_LOSS_WEIGHTS['diagnostic'] +
                       loss_pl * DEFAULT_LOSS_WEIGHTS['upper_air'])
            else:
                loss = (loss_sfc * DEFAULT_LOSS_WEIGHTS['surface'] + 
                       loss_pl * DEFAULT_LOSS_WEIGHTS['upper_air'])

            if self.params.vae_loss:
                loss_vae = self.loss_vae(mu, sigma, mu2, sigma2)
                loss += self.params.vae_loss_weight * loss_vae

        return out_sfc, out_ua, out_diag, loss_sfc, loss_pl, loss_diag, loss_vae, loss

    def _compute_train_metrics(self, out_sfc, out_ua, tgt_sfc, tgt_ua, inp_sfc, inp_ua, dataset, loss, loss_sfc, loss_pl):
        logs = {'train_batch_loss': loss.item(), 'train_batch_loss_sfc': loss_sfc.item(), 'train_batch_loss_upper_air': loss_pl.item()}
        
        if self.params.predict_delta:
            out_sfc, out_ua = self.integrator(inp_sfc, inp_ua, out_sfc, out_ua)
            tgt_sfc, tgt_ua = self.integrator(inp_sfc, inp_ua, tgt_sfc, tgt_ua)
        
        sfc_rmse = weighted_rmse_torch_channels(out_sfc, tgt_sfc, self.lat_weight_2d)
        ua_rmse = weighted_rmse_torch_3D(out_ua, tgt_ua, self.lat_weight_3d)
        
        logs['train_mean_norm_lwrmse'] = torch.mean(torch.cat([
            sfc_rmse, ua_rmse.reshape(out_ua.shape[0], -1)
        ], dim=-1)).item()
        
        return logs

    def validate_one_epoch(self):
        if world_rank == 0:
            print("Validating...")
        self.model.eval()
        
        lead_times = self.params.forecast_lead_times
        valid_loss = torch.zeros(1, device=self.device)
        valid_steps = torch.zeros(1, device=self.device)
        
        with torch.inference_mode():
            for i, data in enumerate(self.valid_data_loader):
                if self.params.has_diagnostic:
                    inp_sfc, inp_ua, tgt_sfc, tgt_ua, tgt_diag, vary_bd, times = [
                        x.to(self.device, dtype=torch.float32, non_blocking=True) if isinstance(x, torch.Tensor) else x
                        for x in data
                    ]
                else:
                    inp_sfc, inp_ua, tgt_sfc, tgt_ua, vary_bd, times = [
                        x.to(self.device, dtype=torch.float32, non_blocking=True) if isinstance(x, torch.Tensor) else x
                        for x in data
                    ]
                    tgt_diag = None
                
                if self.params.has_diagnostic:
                    out_sfc, out_ua, out_diag, _, _ = self.model(
                        inp_sfc, self.constant_boundary_data, vary_bd[:, 0] if vary_bd.dim() > 4 else vary_bd, inp_ua
                    )
                    loss_diag = self.loss_obj_diagnostic(out_diag, tgt_diag[:, 0] if tgt_diag.dim() > 4 else tgt_diag)
                else:
                    out_sfc, out_ua, _, _ = self.model(
                        inp_sfc, self.constant_boundary_data, vary_bd[:, 0] if vary_bd.dim() > 4 else vary_bd, inp_ua
                    )
                    loss_diag = 0
                
                tgt_sfc_step = tgt_sfc[:, 0] if tgt_sfc.dim() > 4 else tgt_sfc
                tgt_ua_step = tgt_ua[:, 0] if tgt_ua.dim() > 5 else tgt_ua
                
                loss_sfc = self.loss_obj_sfc(out_sfc, tgt_sfc_step)
                loss_pl = self.loss_obj_pl(out_ua, tgt_ua_step)
                
                if self.params.has_diagnostic:
                    loss = (loss_sfc * DEFAULT_LOSS_WEIGHTS['surface'] +
                           loss_diag * DEFAULT_LOSS_WEIGHTS['diagnostic'] +
                           loss_pl * DEFAULT_LOSS_WEIGHTS['upper_air'])
                else:
                    loss = (loss_sfc * DEFAULT_LOSS_WEIGHTS['surface'] +
                           loss_pl * DEFAULT_LOSS_WEIGHTS['upper_air'])
                
                valid_loss += loss
                valid_steps += 1
        
        if dist.is_initialized():
            dist.all_reduce(valid_loss)
            dist.all_reduce(valid_steps)
        
        avg_loss = (valid_loss / valid_steps).item()
        
        logs = {
            'valid_loss': avg_loss,
            'valid_loss_sfc': loss_sfc.item(),
            'valid_loss_upper_air': loss_pl.item(),
            'epoch': self.epoch,
        }
        
        if self.params.log_to_wandb:
            wandb.log(logs)
        
        return 0.0, logs

    def save_checkpoint(self, path, model=None):
        if model is None:
            model = self.model
        torch.save({
            'iters': self.iters,
            'epoch': self.epoch,
            'model_state': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }, path)

    def restore_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        try:
            self.model.load_state_dict(ckpt['model_state'])
        except RuntimeError:
            state = ckpt['model_state']
            new_state = {}
            for k, v in state.items():
                new_key = k
                if 'module.' in k and '_orig_mod' not in k:
                    new_key = k.replace('module.', 'module._orig_mod.', 1)
                new_state[new_key] = v
            self.model.load_state_dict(new_state)
        
        self.iters = ckpt['iters']
        self.startEpoch = ckpt['epoch']
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])


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
    parser.add_argument("--fresh_start", action="store_true")
    parser.add_argument("--local-rank", type=int)

    # H100 optimization flags
    parser.add_argument("--accum-steps", default=1, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--torch-compile", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--compile-mode", default="max-autotune", type=str)
    parser.add_argument("--compile-max-autotune", action="store_true", default=True)
    parser.add_argument("--ddp-static-graph", action="store_true", default=True)
    parser.add_argument("--ddp-bucket-cap-mb", default=256, type=int)
    parser.add_argument("--ddp-powersgd", action="store_true")
    parser.add_argument("--ddp-fp16-compress", action="store_true", default=True)
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--log-every-n-steps", default=50, type=int)
    parser.add_argument("--metrics-every", default=200, type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    
    # Profiling arguments
    parser.add_argument("--profiling", action="store_true",
                        help="Enable PyTorch Profiler for performance analysis")
    parser.add_argument("--profile-wait-steps", default=PROFILING_DEFAULTS['wait_steps'], type=int,
                        help="Number of steps to skip before profiling (warmup)")
    parser.add_argument("--profile-warmup-steps", default=PROFILING_DEFAULTS['warmup_steps'], type=int,
                        help="Number of profiler warmup steps")
    parser.add_argument("--profile-active-steps", default=PROFILING_DEFAULTS['active_steps'], type=int,
                        help="Number of steps to actively profile")
    parser.add_argument("--profile-repeat", default=PROFILING_DEFAULTS['repeat'], type=int,
                        help="Number of profiling cycles to repeat")
    parser.add_argument("--profile-memory", action="store_true", default=True,
                        help="Profile memory usage")
    parser.add_argument("--profile-with-stack", action="store_true", default=True,
                        help="Record stack traces for flame graph")
    parser.add_argument("--profile-with-flops", action="store_true", default=True,
                        help="Estimate FLOPS for operations")
    parser.add_argument("--profile-with-modules", action="store_true", default=True,
                        help="Record module hierarchy")
    parser.add_argument("--profile-disable-compile", action="store_true",
                        help="Disable torch.compile during profiling for clearer traces")
    
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Wire CLI flags
    for k, v in vars(args).items():
        if k not in ['yaml_config', 'config', 'local_rank']:
            params[k.replace('-', '_').replace('__', '_')] = v

    # Defaults
    params.setdefault('num_ensemble_members', 1)
    params.setdefault('has_diagnostic', bool(getattr(params, 'diagnostic_variables', [])))

    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    params['world_size'] = world_size

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = params.batch_size // world_size
    else:
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    params['local_rank'] = local_rank

    # Experiment dir
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = os.path.isfile(params.checkpoint_path) and not args.fresh_start

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params.get('log_to_wandb', False)
    params['log_to_screen'] = (world_rank == 0) and params.get('log_to_screen', True)

    # Print profiling info if enabled
    if args.profiling and world_rank == 0:
        print("\n" + "="*80)
        print("PROFILING MODE ENABLED")
        print("="*80)
        print(f"Wait steps:    {args.profile_wait_steps}")
        print(f"Warmup steps:  {args.profile_warmup_steps}")
        print(f"Active steps:  {args.profile_active_steps}")
        print(f"Repeat cycles: {args.profile_repeat}")
        total = (args.profile_wait_steps + args.profile_warmup_steps + args.profile_active_steps) * args.profile_repeat
        print(f"Total steps:   {total}")
        print(f"Memory prof:   {args.profile_memory}")
        print(f"Stack traces:  {args.profile_with_stack}")
        print(f"FLOPS:         {args.profile_with_flops}")
        print("="*80 + "\n")

    trainer = Trainer(params, world_rank)
    trainer.setup_model()
    trainer.train()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    logging.info('DONE ---- rank %d', world_rank)