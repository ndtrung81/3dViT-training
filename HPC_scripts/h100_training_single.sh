#!/bin/bash -l
# ============================================================================
#
# H100 BENCHMARK SUITE  —  Baseline for comparison against B300 (Blackwell)
#
# Runs faster_train.py on NVIDIA H100 GPUs across four benchmark dimensions:
#
#   TASK 1 — Precision Sweep (4x H100, batch_size=8)
#            BF16, FP16, FP32 with TF32, FP32 pure (no TF32)
#            Measures: seconds per iteration for each precision mode.
#
#   TASK 2 — Batch Size Sweep (4x H100, BF16)
#            batch_size = 8, 16, 32
#            Measures: seconds per iteration to find memory saturation point.
#
#   TASK 3 — Data Location (4x H100, batch_size=8, BF16)
#            Local NVMe storage vs GPFS
#            Measures: seconds per iteration to isolate I/O bottleneck.
#
#   TASK 4 — GPU Scaling (H100, batch_size=16, BF16)
#            1, 2, 4 GPUs
#            Measures: seconds per iteration to assess multi-GPU scaling.
#            NOTE: The 1-GPU run skips DDP wrapping (no allreduce overhead).
#
# Usage:
#   sbatch h100_training.sh                              # submit to SLURM
#   bash h100_training.sh 2>&1 | tee h100_results.log   # interactive run
#
# Output:
#   - SLURM .out/.err files (or h100_results.log if run interactively)
#   - Per-run timing printed to stdout: "Duration: Xs (Y.Zm)"
#   - profiler_traces/ directory if PyTorch Profiler tasks are enabled
#
# ============================================================================
#SBATCH --account=pi-pedramh
#SBATCH --time=03:00:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o h100_bench_%x_%j.out
#SBATCH -e h100_bench_%x_%j.err
###SBATCH --mail-type=ALL
###SBATCH --mail-user=youzhi@rcc.uchicago.edu

# Exit on undefined variables; propagate pipeline failures
set -uo pipefail

# ============================================================================
# CONFIGURATION — Edit these paths for your environment
# ============================================================================

# Path to faster_train.py (relative to this script's location or absolute)
TRAIN_SCRIPT="${TRAIN_SCRIPT:-../faster_train.py}"

# YAML config for the model/data (must define batch_size, data_dir, etc.)
YAML_CONFIG="${YAML_CONFIG:-../config/exp1.yaml}"

# Starting run number — incremented after each training run so checkpoints don't collide
RUN_NUM="${RUN_NUM:-99}"

# Temp file for collecting results
RESULTS_FILE=$(mktemp /tmp/h100_bench_results.XXXXXX)
#trap "rm -f $RESULTS_FILE" EXIT

# ------------------------------------
# TASK 3 config: Data location comparison
# Set both to enable the Local-vs-GPFS benchmark.
# Leave empty ("") to skip TASK 3.
# ------------------------------------
DATA_DIR_GPFS="${DATA_DIR_GPFS:-}"        # e.g. /project/pedramh/h5data/h5data
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-}"       # e.g. /local/scratch/h5data

# ============================================================================
# CRITICAL: Environment setup BEFORE any CUDA/Python imports
# ============================================================================

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited

# ============================================================================
# NCCL optimizations for H100 (Hopper, NVLink 4, SM_90)
#
# These match the original midway_training.sh sbatch script.
# All settings are tuned for 4x H100 SXM on a single node with NVLink.
# ============================================================================
export NCCL_DEBUG=WARN                   # Use INFO for debugging, WARN for benchmarks
export NCCL_P2P_LEVEL=5                  # Full NVLink peer-to-peer
export NCCL_P2P_DISABLE=0               # Ensure P2P is enabled
export NCCL_SHM_DISABLE=0               # Shared memory enabled (intra-node)
export NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA level
export NCCL_IB_DISABLE=0                # Enable InfiniBand if available
export NCCL_IB_GID_INDEX=3              # InfiniBand GID index
export NCCL_IB_TIMEOUT=23               # IB timeout (high for stability)
export NCCL_IB_RETRY_CNT=7              # IB retries before failure
export NCCL_SOCKET_IFNAME=^lo,docker0   # Exclude loopback and docker interfaces
export NCCL_SOCKET_NTHREADS=8           # Socket threads for CPU-side NCCL ops
export NCCL_NSOCKS_PERTHREAD=4          # Sockets per NCCL thread
export NCCL_BUFFSIZE=16777216           # 16 MB NCCL buffer (default for H100)
export NCCL_NTHREADS=512                # NCCL GPU threads
export NCCL_MAX_NCHANNELS=32            # Max NCCL channels
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF       # OFF for benchmarks (DETAIL for debugging)

# ============================================================================
# CUDA optimizations for H100
# ============================================================================
export CUDA_LAUNCH_BLOCKING=0                       # Async kernel launches
export TORCH_CUDNN_V8_API_ENABLED=1                 # cuDNN v8 graph API
export CUDA_DEVICE_MAX_CONNECTIONS=1                 # Max concurrent CUDA streams
export NVIDIA_TF32_OVERRIDE=1                       # Enable TF32 globally (overridden per-run if needed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512

# ============================================================================
# Python/PyTorch optimizations
# ============================================================================
export TOKENIZERS_PARALLELISM=false     # Avoid deadlocks with HuggingFace tokenizers
export PYTHONHASHSEED=0                 # Reproducible hash ordering

# ============================================================================
# Load environment — EDIT FOR YOUR SYSTEM
# NOTE: set +u is needed because conda activation scripts reference unbound
#       variables (e.g. ADDR2LINE), which would fail under 'set -u'.
# ============================================================================
ml python
set +u
source activate /project/pedramh/bing/env
set -u
export WANDB_MODE=offline

# ============================================================================
# GPU detection and validation
# ============================================================================
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)
MAX_GPUS=4
if [ "$AVAILABLE_GPUS" -lt "$MAX_GPUS" ]; then
    MAX_GPUS=$AVAILABLE_GPUS
    echo "WARNING: System has $AVAILABLE_GPUS GPUs (expected 4). GPU scaling runs adjusted."
fi

# ============================================================================
# Display system info — useful for reproducibility when reading logs later
# ============================================================================
echo ""
echo "========================================================================"
echo "  H100 BENCHMARK SUITE — System Info"
echo "  $(date)"
echo "========================================================================"
echo ""
echo "=== GPU Hardware ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m
echo ""
echo "=== Cluster Info ==="
echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES:-1}  AVAILABLE_GPUS= ${AVAILABLE_GPUS}"
echo ""
echo "=== PyTorch / CUDA ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB, SM {p.major}.{p.minor}')
" 2>/dev/null || true
echo ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Print a section separator with a message and timestamp
sep() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "========================================================================"
    echo ""
}

# Run a single training experiment and report wall-clock time.
#
# Arguments:
#   $1  name        — Human-readable label for this run (printed in logs)
#   $2  ngpus       — Number of GPUs to use (1, 2, or 4)
#   $3  extra_args  — Additional CLI args for faster_train.py
#   $4  port        — Master port for torchrun (must be unique per concurrent run)
run_training() {
    local name="$1"
    local ngpus="$2"
    local extra_args="$3"
    local port="$4"

    sep "TRAINING: $name (${ngpus} GPU)"
    echo "Command:"
    echo "  torchrun --standalone --nproc_per_node=$ngpus --master_port=$port \\"
    echo "    $TRAIN_SCRIPT --yaml_config=$YAML_CONFIG --run_num=$RUN_NUM $extra_args"
    echo ""

    local start_time=$(date +%s)
    #local _run_log=$(mktemp /tmp/h100_run.XXXXXX)
    local _run_log=h100_run.$port

    # Run training. Stdout goes to both terminal (SLURM .out) and _run_log.
    # Stderr goes to SLURM .err as normal.
    torchrun \
        --standalone \
        --nproc_per_node=$ngpus \
        --master_port=$port \
        $TRAIN_SCRIPT \
        --yaml_config=$YAML_CONFIG \
        --run_num=$RUN_NUM \
        $extra_args | tee "$_run_log"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Extract benchmark metrics from this run's output
    local _secstep="N/A"
    local _samplesec="N/A"
    local _bline=$(grep "BENCHMARK_RESULT:" "$_run_log" | tail -1)
    if [ -n "$_bline" ]; then
        _secstep=$(echo "$_bline" | sed 's/.*sec_per_step=//;s/ .*//')
        _samplesec=$(echo "$_bline" | sed 's/.*samples_per_sec=//')
    fi
    rm -f "$_run_log"

    echo ""
    echo "--- RESULT: $name (${ngpus} GPU) ---"
    echo "  Exit code: $exit_code"
    echo "  Duration:  ${duration}s ($(echo "scale=1; $duration/60" | bc)m)"
    echo "  sec/step:  $_secstep"
    echo "  samples/s: $_samplesec"
    echo "---"
    echo ""

    # Capture result for summary table
    # Format: name|exit_code|duration|sec_per_step|samples_per_sec
    echo "${name} (${ngpus} GPU)|${exit_code}|${duration}|${_secstep}|${_samplesec}" >> "$RESULTS_FILE"

    # Increment RUN_NUM so the next run gets a fresh checkpoint directory
    RUN_NUM=$((RUN_NUM + 1))
}

# ============================================================================
# VALIDATE TRAIN SCRIPT EXISTS
# ============================================================================

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at: $TRAIN_SCRIPT"
    echo "Set TRAIN_SCRIPT=/path/to/faster_train.py or run from HPC_scripts/ directory."
    exit 1
fi

# ------------------------------------
# Common flags shared by ALL benchmark runs:
#   --ddp-static-graph   : Enable static DDP graph for better overlap of compute and comms
#   --max-grad-norm 1.0  : Gradient clipping for training stability
#   --log-every-n-steps  : How often to print loss (not too frequent to avoid overhead)
#   --metrics-every      : How often to compute RMSE metrics (expensive, so infrequent)
#   --accum-steps 1      : No gradient accumulation (measure raw per-step throughput)
# ------------------------------------
COMMON="--epochs 1 --max-steps 50 --fresh_start --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 100 --metrics-every 500 --accum-steps 1"

# ============================================================================
# TASK 1: PRECISION SWEEP
#
# Measures: Seconds per iteration across precision modes
# Config:   4x H100, batch_size=8 (default from YAML)
# Precisions tested:
#   - BF16     : Default mixed precision (fastest on H100)
#   - FP16     : FP16 mixed precision with loss scaling
#   - FP32+TF32: Full FP32 but TF32 hardware acceleration enabled
#   - FP32 pure: FP32 with TF32 disabled (slowest, reference baseline)
#
# NOTE: FP8 is NOT tested on H100. FP8 via Transformer Engine is only
#       meaningful on Blackwell (B300). The B300 script includes FP8.
# ============================================================================

# single run
sep "TASK 1: PRECISION SWEEP (4x H100, batch_size=12)"
run_training "Precision: BF16" \
    $MAX_GPUS \
    "--amp-dtype bf16 --ddp-bucket-cap-mb 256 --ddp-fp16-compress --batch-size-override 12 $COMMON" \
    29500


# ============================================================================
# DONE — Results Table
# ============================================================================

# Flush stdout so the .out file has all BENCHMARK_RESULT lines before we grep it
sep "ALL H100 BENCHMARKS COMPLETE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                          H100 BENCHMARK RESULTS TABLE                               ║"
echo "╠════════════════════════════════════╦════════╦══════════╦════════════╦════════════════╣"
echo "║ Run Name                           ║ Status ║ Wall (s) ║ sec/step   ║ samples/sec    ║"
echo "╠════════════════════════════════════╬════════╬══════════╬════════════╬════════════════╣"

# Read results from the temp file (populated by run_training)
# Format: name|exit_code|duration|sec_per_step|samples_per_sec
while IFS='|' read -r name exitcode duration secstep samplesec; do
    if [ "$exitcode" = "0" ]; then
        status="  OK  "
    else
        status=" FAIL "
        secstep="N/A"
        samplesec="N/A"
    fi
    printf "║ %-36s ║ %s ║ %8s ║ %10s ║ %14s ║\n" "$name" "$status" "$duration" "$secstep" "$samplesec"
done < "$RESULTS_FILE"

echo "╚════════════════════════════════════╩════════╩══════════╩════════════╩════════════════╝"
echo ""
echo "Notes:"
echo "  - sec/step = wall-clock seconds per training step (lower is better)"
echo "  - samples/sec = global_batch_size / sec_per_step (higher is better)"
echo "  - FAIL runs hit CUDA OOM or other errors (check .err file for details)"
echo "  - All runs used 50 training steps (--max-steps 50) for timing consistency"
echo ""
nvidia-smi 2>/dev/null || true
echo "=== H100 Benchmarks Complete ==="
