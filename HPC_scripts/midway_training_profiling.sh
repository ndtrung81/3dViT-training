#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=20:10:00
#SBATCH -p pedramh-gpu 
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16  # Increased for better data loading parallelism
#SBATCH -o midway_ddp_%x_%j.out
#SBATCH -e midway_ddp_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youzhi@rcc.uchicago.edu

# =============================================================================
# CRITICAL: Environment setup BEFORE any CUDA/Python imports
# =============================================================================

# GPU support
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited

# =============================================================================
# NCCL optimizations for 4x H100 with NVLink (CRITICAL FOR PERFORMANCE)
# =============================================================================
# Use WARN in production, INFO only for debugging
export NCCL_DEBUG=WARN

# NVLink/P2P optimizations (CRITICAL for 4x H100 on single node)
export NCCL_P2P_LEVEL=5               # Full NVLink P2P communication
export NCCL_P2P_DISABLE=0             # Ensure P2P is enabled
export NCCL_SHM_DISABLE=0             # Enable shared memory
export NCCL_NET_GDR_LEVEL=5           # Max GPU Direct RDMA

# InfiniBand settings (if available)
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# Socket transport optimizations
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_NTHREADS=8         # More threads for H100
export NCCL_NSOCKS_PERTHREAD=4

# Buffer and thread optimizations for high-bandwidth H100
export NCCL_BUFFSIZE=16777216         # 16MB buffer (increased for H100)
export NCCL_NTHREADS=512              # More NCCL threads
export NCCL_MAX_NCHANNELS=32          # More channels for H100

# Async error handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF    # OFF in production

# =============================================================================
# CUDA optimizations for H100 Tensor Cores
# =============================================================================
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Memory management optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512

# Enable TF32 for matrix operations (automatic in code, but good to ensure)
export NVIDIA_TF32_OVERRIDE=1

# =============================================================================
# Python/PyTorch optimizations
# =============================================================================
# Disable tokenizers parallelism (conflicts with DataLoader workers)
export TOKENIZERS_PARALLELISM=false

# Optimize Python garbage collection
export PYTHONHASHSEED=0

# Use TCMalloc for better memory allocation (if available)
# export LD_PRELOAD=/path/to/libtcmalloc.so:$LD_PRELOAD

# =============================================================================
# Load environment
# =============================================================================
ml python
source activate /project/pedramh/bing/env
export WANDB_MODE=offline

# =============================================================================
# Display GPU info
# =============================================================================
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
nvidia-smi topo -m
echo ""

# Get GPU count
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${SLURM_NTASKS}"

# Configuration
config_file=../config/exp1.yaml

# =============================================================================
# Launch with MAXIMUM H100 optimizations
# =============================================================================
# Key changes from original:
#   --compile-mode max-autotune   : Slower compile, MUCH faster runtime
#   --ddp-static-graph            : Enable static graph (default True now)
#   --ddp-bucket-cap-mb 256       : Larger buckets for NVLink bandwidth
#   --ddp-fp16-compress           : Compress gradients (reduces comm overhead)
#   --max-grad-norm 1.0           : Gradient clipping for stability
#   --log-every-n-steps 100       : Reduce logging overhead
#   --metrics-every 500           : Compute expensive metrics less often
# =============================================================================

echo "=== Starting Training ==="
echo "Config: $config_file"
echo "Compile mode: max-autotune"
echo "AMP dtype: bf16"
echo ""

/project/pedramh/bing/env/bin/torchrun \
  --standalone \
  --nproc_per_node=$NUM_TASKS_PER_NODE \
  --master_port=29500 \
  ../train_optimized.py \
  --yaml_config=$config_file \
  --run_num=1 \
  --amp-dtype bf16 \
  --torch-compile True \
  --compile-mode max-autotune \
  --compile-max-autotune \
  --ddp-static-graph \
  --ddp-bucket-cap-mb 256 \
  --ddp-fp16-compress \
  --max-grad-norm 1.0 \
  --log-every-n-steps 100 \
  --metrics-every 500 \
  --accum-steps 1

# =============================================================================
# Alternative: Even more aggressive settings (may need tuning)
# =============================================================================
# If you want to try gradient accumulation for effectively larger batches:
#
# /project/pedramh/bing/env/bin/torchrun \
#   --standalone \
#   --nproc_per_node=$NUM_TASKS_PER_NODE \
#   ../train_optimized.py \
#   --yaml_config=$config_file \
#   --run_num=1 \
#   --amp-dtype bf16 \
#   --torch-compile True \
#   --compile-mode max-autotune \
#   --ddp-static-graph \
#   --ddp-bucket-cap-mb 256 \
#   --ddp-fp16-compress \
#   --accum-steps 2 \
#   --max-grad-norm 1.0 \
#   --log-every-n-steps 100 \
#   --metrics-every 500 \
#   --gradient-checkpointing  # Enable if OOM with larger batches

echo "=== Training Complete ==="

# =============================================================================
# PROFILING MODE - Uncomment to run performance analysis
# =============================================================================
# This will run for a limited number of steps and generate profiling traces.
# Results can be viewed in Chrome Tracing (chrome://tracing) or TensorBoard.
#
# /project/pedramh/bing/env/bin/torchrun \
#   --standalone \
#   --nproc_per_node=$NUM_TASKS_PER_NODE \
#   --master_port=29500 \
#   ../train_optimized.py \
#   --yaml_config=$config_file \
#   --run_num=1 \
#   --amp-dtype bf16 \
#   --torch-compile True \
#   --compile-mode reduce-overhead \
#   --ddp-static-graph \
#   --ddp-bucket-cap-mb 256 \
#   --profiling \
#   --profile-wait-steps 5 \
#   --profile-warmup-steps 3 \
#   --profile-active-steps 10 \
#   --profile-repeat 1 \
#   --profile-memory \
#   --profile-with-stack \
#   --profile-with-flops \
#   --profile-with-modules
#
# For cleaner traces without torch.compile overhead:
# Add: --profile-disable-compile
#
# After profiling, view results:
#   1. Chrome: Open chrome://tracing, load trace_rank0_*.json
#   2. TensorBoard: tensorboard --logdir=<exp_dir>/profiler_traces
#   3. Flame Graph: flameprof stacks_rank0_*.txt > flamegraph.svg