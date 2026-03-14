#!/bin/bash -l
# ============================================================================
#
# B300 (Blackwell) BENCHMARK SUITE  —  Comparison against H100 (Hopper)
#
# Runs faster_train.py on NVIDIA B300 GPUs across four benchmark dimensions,
# mirroring the H100 benchmark suite for direct comparison:
#
#   TASK 1 — Precision Sweep (4x B300, batch_size=8)
#            BF16, FP16, FP32 with TF32, FP32 pure (no TF32)
#            + FP8 via Transformer Engine (Blackwell-only, bonus comparison)
#            Measures: seconds per iteration for each precision mode.
#
#   TASK 2 — Batch Size Sweep (4x B300, BF16)
#            batch_size = 8, 16, 32
#            Measures: seconds per iteration to find memory saturation point.
#
#   TASK 3 — Data Location (4x B300, batch_size=8, BF16)
#            Local NVMe storage vs GPFS
#            Measures: seconds per iteration to isolate I/O bottleneck.
#
#   TASK 4 — GPU Strong scaling (B300, batch_size=16, BF16)
#            1, 2, 4, 8 GPUs
#            Measures: seconds per iteration to assess multi-GPU scaling.
#            NOTE: The 1-GPU run skips DDP wrapping (no allreduce overhead).
#
#   TASK 5 - GPU Weak scaling (B300, batch size = 2 per GPU) 
#            1, 2, 4, 8 GPUs
#            Measures: seconds per iteration to assess multi-GPU scaling.
#            NOTE: The 1-GPU run skips DDP wrapping (no allreduce overhead).
#
# Key differences from H100 script:
#   - DDP bucket size: 512 MB (vs 256 MB on H100) — Blackwell has NVLink 5
#     with higher bandwidth, so larger buckets reduce allreduce round-trips.
#   - FP8 precision test via NVIDIA Transformer Engine (Blackwell SM_100+).
#   - NCCL buffer/channel settings tuned for Blackwell's wider NVLink.
#
# Usage:
#   sbatch b300_training.sh                              # submit to SLURM
#   bash b300_training.sh 2>&1 | tee b300_results.log   # interactive run
#
# Output:
#   - SLURM .out/.err files (or b300_results.log if run interactively)
#   - Per-run timing printed to stdout: "Duration: Xs (Y.Zm)"
#   - profiler_traces/ directory if PyTorch Profiler tasks are enabled
#   - nsys_*.nsys-rep / ncu_*.ncu-rep if profiling sections are uncommented
#
# ============================================================================

#SBATCH --time=02:00:00
#SBATCH -p b300
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o bench_%x_%j.out
#SBATCH -e bench_%x_%j.err

###SBATCH --mail-type=ALL
###SBATCH --mail-user=youzhi@rcc.uchicago.edu

# Exit on undefined variables; propagate pipeline failures
set -uo pipefail

# ============================================================================
# CONFIGURATION — Edit these paths for your environment
# ============================================================================

# Path to faster_train.py (relative to this script's location or absolute)
#TRAIN_SCRIPT="${TRAIN_SCRIPT:-../faster_train.py}"
TRAIN_SCRIPT="../faster_train.py"

# YAML config for the model/data (must define batch_size, data_dir, etc.)
#YAML_CONFIG="${YAML_CONFIG:-../config/exp1_profiling.yaml}"
YAML_CONFIG="../config/exp1_dsai.yaml"

# Starting run number — incremented after each training run so checkpoints don't collide
RUN_NUM="${RUN_NUM:-99}"

# Temp file for collecting results
jobid=$SLURM_JOBID
#RESULTS_FILE=$(mktemp /tmp/b300_bench_results.$jobid)
RESULTS_FILE=results.b300.${jobid}.maxgpus4.batch_size8.all
#trap "rm -f $RESULTS_FILE" EXIT

#export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 >= 2 {print $2}' | head -n4 | paste -sd ',')

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
# NCCL optimizations for B300 (Blackwell, NVLink 5, SM_100+)
#
# Blackwell has 2x the NVLink bandwidth of Hopper (1.8 TB/s vs 900 GB/s),
# so we use larger buffers and more channels to saturate the link.
# These settings are otherwise identical to the H100 script for consistency.
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
export NCCL_BUFFSIZE=16777216           # 16 MB NCCL buffer
export NCCL_NTHREADS=512                # NCCL GPU threads
export NCCL_MAX_NCHANNELS=32            # Max NCCL channels
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF       # OFF for benchmarks (DETAIL for debugging)

# extract 4 full GPUs
#export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 >= 2 {print $2}' | head -4 | paste -sd ',')

# PCIe
#export NCCL_DEBUG=WARN
#export NCCL_P2P_LEVEL=2
#export NCCL_P2P_DISABLE=0
#export NCCL_SHM_DISABLE=0
#export NCCL_IB_DISABLE=1
#export NCCL_NET_GDR_LEVEL=0
#export NCCL_SOCKET_IFNAME=^lo,docker0


# CUDA optimizations for B300
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
#ml python
set +u
#source activate /project/pedramh/bing/env
module load anaconda3/2024.02-1
source activate /scratch/dzarzhi1/ext-ndtrung/s2s
set -u
export WANDB_MODE=offline

# ============================================================================
# GPU detection and validation
# ============================================================================
AVAILABLE_GPUS=$(nvidia-smi -L | grep "^GPU" | grep -v "MIG" 2>/dev/null | wc -l || echo 4)
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
echo "  B300 (Blackwell) BENCHMARK SUITE — System Info"
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
echo "=== PyTorch / CUDA / Transformer Engine ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB, SM {p.major}.{p.minor}')
try:
    import transformer_engine
    print(f'Transformer Engine: {transformer_engine.__version__} (FP8 supported)')
except ImportError:
    print('Transformer Engine: NOT installed (FP8 runs will fall back to BF16)')
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
    local _run_log=$(mktemp /tmp/b300_run.XXXXXX)

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
    #echo "  Duration:  ${duration}s ($(echo "scale=1; $duration/60" | bc)m)"
    echo "  Duration:  ${duration}s"
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

# ------------------------------------
# B300-specific: DDP bucket size is 512 MB (vs 256 MB on H100).
# Blackwell's NVLink 5 has ~2x the bandwidth, so larger buckets
# reduce the number of allreduce round-trips without stalling.
# ------------------------------------
B300_BUCKET_MB=512


# ============================================================================
# TASK 1: PRECISION SWEEP
#
# Measures: Seconds per iteration across precision modes
# Config:   4x B300, batch_size=8 (default from YAML)
# Precisions tested:
#   - BF16      : Default mixed precision (fast on Blackwell Tensor Cores)
#   - FP16      : FP16 mixed precision with loss scaling
#   - FP32+TF32 : Full FP32 but TF32 hardware acceleration enabled
#   - FP32 pure : FP32 with TF32 disabled (slowest, reference baseline)
#   - FP8 (bonus): FP8 via NVIDIA Transformer Engine (Blackwell-only feature)
#
# FP8 notes:
#   - Requires transformer_engine pip package.
#   - Uses te.fp8_autocast() around the forward pass.
#   - Full FP8 speedup requires replacing nn.Linear with te.Linear in the model.
#     This run tests correctness and provides a baseline for future TE integration.
#   - If Transformer Engine is not installed, this run falls back to BF16 automatically.
# ============================================================================

running_precision=0
running_batch_sizes=1
running_num_gpus_strong=0
running_num_gpus_weak=1

if [ $running_precision -eq 1 ]
then
    sep "TASK 1: PRECISION SWEEP ($MAX_GPUS x B300, batch_size=8)"

    global_batch=8

    bf16=1
    fp16=1
    tf32=1
    fp32=1
    fp8=1

    if [ $bf16 -eq 1 ]
    then
        # BF16 — the recommended precision for Blackwell.
        # Uses torch.amp autocast with bfloat16. No loss scaling needed.
        run_training "Precision: BF16" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29500
    fi

    # FP16 — uses torch.amp autocast with float16.
    # Requires GradScaler for loss scaling to avoid underflow.
    #fp16=1
    if [ $fp16 -eq 1 ]
    then
        run_training "Precision: FP16" \
        $MAX_GPUS \
        "--amp-dtype fp16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29501
    fi

    # FP32 with TF32 — full 32-bit accumulation, but matmuls use TF32 hardware paths.
    #tf32=1
    if [ $tf32 -eq 1 ]
    then
        run_training "Precision: FP32 with TF32" \
        $MAX_GPUS \
        "--amp-dtype fp32 --ddp-bucket-cap-mb $B300_BUCKET_MB --batch-size-override $global_batch $COMMON" \
        29502
    fi

    # FP32 pure (no TF32) — the slowest reference baseline.
    # --no-tf32 disables TF32 in matmul and cuDNN, sets float32_matmul_precision='highest'.
    #fp32=1
    if [ $fp32 -eq 1 ]
    then
        run_training "Precision: FP32 pure (no TF32)" \
        $MAX_GPUS \
        "--amp-dtype fp32 --no-tf32 --ddp-bucket-cap-mb $B300_BUCKET_MB --batch-size-override $global_batch $COMMON" \
        29503
    fi

    # FP8 via Transformer Engine — Blackwell-only bonus test.
    # If Transformer Engine is not installed, faster_train.py will log a warning
    # and fall back to BF16 automatically (check logs for "Falling back to BF16").
    #fp8=1
    if [ $fp8 -eq 1 ]
    then
        run_training "Precision: FP8 (Transformer Engine)" \
        $MAX_GPUS \
        "--amp-dtype fp8 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
        29504
    fi
fi

# ============================================================================
# TASK 2: BATCH SIZE SWEEP
#
# Measures: Seconds per iteration at different global batch sizes
# Config:   4x B300, BF16 precision
# Batch sizes: 4, 8, 12 (per-GPU: 1, 2, 3)
#
# NOTE: With num_ensemble_members=4, the model uses ~90 GB per GPU at
#       per-GPU batch=2 on H100. B300 has ~192 GB VRAM so larger batches
#       may fit. We start conservatively and match H100 sizes for comparison.
#       If B300 handles these, try uncommenting larger sizes below.
# ============================================================================

if [ $running_batch_sizes -eq 1 ]
then      	
    sep "TASK 2: BATCH SIZE SWEEP (4x B300, BF16)"

    # global_batch varies from 16 to max out device memory.
    for bsize in 4 #16 12 8 4
    do
	seed=`shuf -i10000-39999 -n1`
        run_training "BatchSize: $bsize" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $bsize $COMMON" \
        $seed
    done

fi

# ============================================================================
# TASK 3: DATA LOCATION — Local NVMe vs GPFS
#
# Measures: Seconds per iteration from different storage backends
# Config:   4x B300, batch_size=8, BF16
#
# Purpose: Isolate I/O bottleneck. If local NVMe is significantly faster
#          than GPFS, the training is I/O-bound and would benefit from
#          staging data to local scratch before training.
#
# NOTE: Set DATA_DIR_GPFS and DATA_DIR_LOCAL at the top of this script
#       to enable this section. Both must be non-empty.
# ============================================================================

if [ -n "$DATA_DIR_LOCAL" ] && [ -n "$DATA_DIR_GPFS" ]; then
    sep "TASK 3: DATA LOCATION — Local NVMe vs GPFS (4x B300, batch_size=8)"

    # GPFS — network-attached parallel filesystem (default for most HPC jobs)
    run_training "DataLoc: GPFS" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --data-dir-override $DATA_DIR_GPFS $COMMON" \
        29520

    # Local NVMe — node-local SSD (fastest possible I/O, but data must be staged)
    run_training "DataLoc: Local NVMe" \
        $MAX_GPUS \
        "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --data-dir-override $DATA_DIR_LOCAL $COMMON" \
        29521

else
    sep "TASK 3: DATA LOCATION — SKIPPED"
    echo "To enable, set DATA_DIR_GPFS and DATA_DIR_LOCAL at the top of this script."
    echo "  e.g. DATA_DIR_GPFS=/project/pedramh/h5data/h5data"
    echo "       DATA_DIR_LOCAL=/local/scratch/h5data"
fi


# ============================================================================
# TASK 4: GPU STRONG SCALING
#
# Measures: Seconds per iteration with 1, 2, 4 GPUs
# Config:   B300, BF16, global batch_size= 16 (kept constant)
#
# Purpose: Measure how well training scales across GPUs for a given global batch size.
#
# ============================================================================

# strong scaling: global batch = 16
if [ $running_num_gpus_strong -eq 1 ]
then
    sep "TASK 4: GPU STRONG SCALING (B300, global batch size = 16, BF16)"

    global_batch=16

    for num_gpus in 8 4 2 1
    do
	if [ "$MAX_GPUS" -ge $num_gpus ]; then
           seed=`shuf -i10000-39999 -n1`
	   run_training "GPUScale: $num_gpus GPU with global batch $global_batch" \
           $num_gpus \
           "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
           $seed
        fi
    done

    # 1 GPU — no DDP communication overhead.
    # global_batch=8 so per-GPU=8.
    #run_training "GPUScale: 1 GPU" \
    #1 \
    #"--amp-dtype bf16 --batch-size-override $global_batch $COMMON" \
    #29530

fi

# ============================================================================
# TASK 5: GPU WEAK SCALING
#
# Measures: Seconds per iteration with 1, 2, 4 GPUs
# Config:   B300, BF16, per-GPU batch_size=2 (kept constant)
#
# Purpose: Measure how well training scales across GPUs.
#   - 1 GPU:  global_batch=2 (per-GPU=2). Pure compute, no allreduce.
#   - 2 GPU:  global_batch=4 (per-GPU=2). Tests NVLink 5 scaling.
#   - 4 GPU:  global_batch=8 (per-GPU=2). Full node scaling.
#
# We keep per-GPU batch_size constant at 2 to isolate the effect of
# DDP communication overhead from batch size differences.
# ============================================================================


# weak scaling: 2 per GPU
if [ $running_num_gpus_weak -eq 1 ]
then
    sep "TASK 5: GPU WEAK SCALING (B300, per-GPU batch=2, BF16)"

    # keep batch size per-GPU=2
    batch_size_per_gpu=2

    for num_gpus in 1 #8 4 2 1
    do
        if [ "$MAX_GPUS" -ge $num_gpus ]; then
	   global_batch=$(( num_gpus * batch_size_per_gpu ))
	   seed=`shuf -i10000-39999 -n1`
	   
	   sep "Running with $num_gpus and global batch $global_batch.."
           
	   run_training "GPUScale: $num_gpus GPU" \
           $num_gpus \
           "--amp-dtype bf16 --ddp-bucket-cap-mb $B300_BUCKET_MB --ddp-fp16-compress --batch-size-override $global_batch $COMMON" \
           $seed
        fi
    done


fi

# ============================================================================
# OPTIONAL: NSYS PROFILING (uncomment to enable)
#
# Nsight Systems captures GPU kernel timelines, CUDA API calls, NCCL comms,
# and memory transfers. Output is a .nsys-rep file viewable in nsys-ui.
# ============================================================================

# Uncomment the block below to enable nsys profiling on B300:
#
# if command -v nsys &>/dev/null; then
#     sep "OPTIONAL: NSYS PROFILING (1 GPU, BF16)"
#     PROF_COMMON="--ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 --metrics-every 50 --accum-steps 1"
#     nsys profile \
#         --trace=cuda,nvtx,osrt,cudnn,cublas \
#         --cuda-memory-usage=true \
#         --gpu-metrics-device=all \
#         --output="nsys_b300_bf16_1gpu" \
#         --force-overwrite=true \
#         torchrun --standalone --nproc_per_node=1 --master_port=29550 \
#             $TRAIN_SCRIPT --yaml_config=$YAML_CONFIG --run_num=$RUN_NUM \
#             --amp-dtype bf16 \
#             --compile-max-autotune $PROF_COMMON
# fi


# ============================================================================
# OPTIONAL: PYTORCH PROFILER (uncomment to enable)
#
# The built-in PyTorch profiler generates Chrome/TensorBoard traces with
# per-operator FLOPs and memory allocation tracking.
# Output: <experiment_dir>/profiler_traces/*.json
# ============================================================================

# Uncomment the block below to enable PyTorch Profiler on B300:
#
# sep "OPTIONAL: PYTORCH PROFILER (1 GPU, BF16)"
# run_training "PyTorch Profiler: BF16 1GPU" \
#     1 \
#     "--amp-dtype bf16 \
#      --profiling --profile-wait-steps 5 --profile-warmup-steps 3 \
#      --profile-active-steps 10 --profile-with-flops --profile-memory \
#      --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 \
#      --metrics-every 50 --accum-steps 1" \
#     29560


# ============================================================================
# DONE — Results Table
# ============================================================================

sep "ALL B300 BENCHMARKS COMPLETE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                          B300 BENCHMARK RESULTS TABLE                               ║"
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
#nvidia-smi 2>/dev/null || true
echo "=== B300 Benchmarks Complete ==="
