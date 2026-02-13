#!/bin/bash -l
# ============================================================================
#
# B300 BENCHMARK SUITE
#
# Based on the H100 sbatch script. Same env vars, same NCCL tuning.
# Runs faster_train.py across multiple dimensions:
#   - GPU scaling (1, 2, 4 GPUs)
#   - Precision (BF16, FP16, FP8, FP32, FP32 no TF32)
#   - Batch size sweep (memory saturation)
#   - Data loader source (local NVMe vs GPFS)
#   - nsys / ncu profiling for FLOPs and memory bandwidth
#
# Usage:
#   bash b300_training.sh 2>&1 | tee b300_results.log
#   sbatch b300_training.sh
#
# ============================================================================
#SBATCH --account=pi-pedramh
#SBATCH --time=20:10:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -o b300_bench_%x_%j.out
#SBATCH -e b300_bench_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youzhi@rcc.uchicago.edu

set -uo pipefail

# ============================================================================
# CONFIGURATION — EDIT THESE
# ============================================================================

TRAIN_SCRIPT="${TRAIN_SCRIPT:-../faster_train.py}"
YAML_CONFIG="${YAML_CONFIG:-../config/exp1.yaml}"
RUN_NUM="${RUN_NUM:-99}"

# Data directories for local vs GPFS comparison
# Set these to the actual paths on your system
DATA_DIR_GPFS="${DATA_DIR_GPFS:-}"        # e.g. /project/pedramh/data
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-}"       # e.g. /local/scratch/data or /tmp/data

# Batch sizes for memory saturation sweep (adjust to your model)
BATCH_SIZES="${BATCH_SIZES:-}"            # e.g. "4 8 16 32 64" — leave empty to skip

# ============================================================================
# CRITICAL: Environment setup BEFORE any CUDA/Python imports
# ============================================================================

export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -l unlimited

# ============================================================================
# NCCL optimizations (matches original H100 sbatch exactly)
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=16777216
export NCCL_NTHREADS=512
export NCCL_MAX_NCHANNELS=32
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# ============================================================================
# CUDA optimizations
# ============================================================================
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export NVIDIA_TF32_OVERRIDE=1

# ============================================================================
# Python/PyTorch optimizations
# ============================================================================
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# ============================================================================
# Load environment — EDIT FOR YOUR SYSTEM
# ============================================================================
ml python
source activate /project/pedramh/bing/env
export WANDB_MODE=offline

# ============================================================================
# GPU detection
# ============================================================================
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)
MAX_GPUS=4
if [ "$AVAILABLE_GPUS" -lt "$MAX_GPUS" ]; then
    MAX_GPUS=$AVAILABLE_GPUS
    echo "NOTE: System has $AVAILABLE_GPUS GPUs (fewer than 4). Scaling runs adjusted."
fi

# ============================================================================
# Display system info
# ============================================================================
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""
nvidia-smi topo -m
echo ""
echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES:-1} AVAILABLE_GPUS= ${AVAILABLE_GPUS}"
echo ""
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB, SM {p.major}.{p.minor}')
try:
    import transformer_engine
    print(f'Transformer Engine: {transformer_engine.__version__}')
except ImportError:
    print('Transformer Engine: not installed (FP8 runs will fall back to BF16)')
" 2>/dev/null || true
echo ""

# ============================================================================
# HELPERS
# ============================================================================

sep() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date)"
    echo "========================================================================"
    echo ""
}

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

    torchrun \
        --standalone \
        --nproc_per_node=$ngpus \
        --master_port=$port \
        $TRAIN_SCRIPT \
        --yaml_config=$YAML_CONFIG \
        --run_num=$RUN_NUM \
        $extra_args

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "--- RESULT: $name (${ngpus} GPU) ---"
    echo "  Exit code: $exit_code"
    echo "  Duration:  ${duration}s ($(echo "scale=1; $duration/60" | bc)m)"
    echo "---"
    echo ""

    RUN_NUM=$((RUN_NUM + 1))
}

run_nsys() {
    local name="$1"
    local ngpus="$2"
    local extra_args="$3"
    local port="$4"
    local output_prefix="nsys_${name// /_}_${ngpus}gpu"

    sep "NSYS PROFILE: $name (${ngpus} GPU)"
    echo "Output: ${output_prefix}.nsys-rep"
    echo ""

    local start_time=$(date +%s)

    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --gpu-metrics-device=all \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --output="${output_prefix}" \
        --force-overwrite=true \
        torchrun \
            --standalone \
            --nproc_per_node=$ngpus \
            --master_port=$port \
            $TRAIN_SCRIPT \
            --yaml_config=$YAML_CONFIG \
            --run_num=$RUN_NUM \
            $extra_args

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "--- RESULT: NSYS $name (${ngpus} GPU) ---"
    echo "  Exit code: $exit_code"
    echo "  Duration:  ${duration}s"
    echo "  Report:    ${output_prefix}.nsys-rep"
    echo "---"
    echo ""

    # Print summary stats from the nsys report
    if [ $exit_code -eq 0 ] && command -v nsys &>/dev/null; then
        echo "--- NSYS Summary ---"
        nsys stats "${output_prefix}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null || true
        echo ""
        nsys stats "${output_prefix}.nsys-rep" --report cuda_gpu_mem_size_sum 2>/dev/null || true
        echo "---"
    fi

    RUN_NUM=$((RUN_NUM + 1))
}

run_ncu() {
    local name="$1"
    local extra_args="$2"
    local port="$3"
    local output_prefix="ncu_${name// /_}"

    sep "NCU PROFILE: $name (1 GPU only)"
    echo "Output: ${output_prefix}.ncu-rep"
    echo "NOTE: ncu runs single-GPU only and profiles a small number of kernels."
    echo ""

    local start_time=$(date +%s)

    # ncu with key metrics: FLOPs, memory throughput, occupancy
    ncu \
        --set full \
        --target-processes all \
        --launch-skip 50 \
        --launch-count 20 \
        --output "${output_prefix}" \
        --force-overwrite \
        torchrun \
            --standalone \
            --nproc_per_node=1 \
            --master_port=$port \
            $TRAIN_SCRIPT \
            --yaml_config=$YAML_CONFIG \
            --run_num=$RUN_NUM \
            $extra_args

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "--- RESULT: NCU $name ---"
    echo "  Exit code: $exit_code"
    echo "  Duration:  ${duration}s"
    echo "  Report:    ${output_prefix}.ncu-rep"
    echo "---"
    echo ""

    RUN_NUM=$((RUN_NUM + 1))
}

# ============================================================================
# CHECK TRAIN SCRIPT
# ============================================================================

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found at: $TRAIN_SCRIPT"
    echo "Set TRAIN_SCRIPT=/path/to/faster_train.py"
    exit 1
fi

COMMON="--ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 100 --metrics-every 500 --accum-steps 1"

# ============================================================================
# SECTION 1: PRECISION SWEEP (4 GPU, apples-to-apples)
# ============================================================================

sep "SECTION 1: PRECISION SWEEP (4 GPU)"

run_training "BF16 Baseline" \
    $MAX_GPUS \
    "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress $COMMON" \
    29500

run_training "FP16" \
    $MAX_GPUS \
    "--amp-dtype fp16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress $COMMON" \
    29501

run_training "FP8 via Transformer Engine" \
    $MAX_GPUS \
    "--amp-dtype fp8 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-fp16-compress $COMMON" \
    29502

run_training "FP32 with TF32" \
    $MAX_GPUS \
    "--amp-dtype fp32 --torch-compile True --compile-mode max-autotune --ddp-bucket-cap-mb 256 $COMMON" \
    29503

run_training "FP32 pure (no TF32)" \
    $MAX_GPUS \
    "--amp-dtype fp32 --no-tf32 --torch-compile True --compile-mode max-autotune --ddp-bucket-cap-mb 256 $COMMON" \
    29504

# ============================================================================
# SECTION 2: GPU SCALING (1, 2, 4 GPUs with BF16)
# ============================================================================

sep "SECTION 2: GPU SCALING"

run_training "BF16 1-GPU" \
    1 \
    "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune $COMMON" \
    29510

if [ "$MAX_GPUS" -ge 2 ]; then
    run_training "BF16 2-GPU" \
        2 \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress $COMMON" \
        29511
fi

if [ "$MAX_GPUS" -ge 4 ]; then
    run_training "BF16 4-GPU" \
        4 \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress $COMMON" \
        29512
fi

# ============================================================================
# SECTION 3: BATCH SIZE SWEEP (memory saturation)
# Set BATCH_SIZES="4 8 16 32 64" to enable.
# ============================================================================

if [ -n "$BATCH_SIZES" ]; then
    sep "SECTION 3: BATCH SIZE SWEEP (memory saturation)"
    PORT=29520
    for BS in $BATCH_SIZES; do
        run_training "BF16 batch_size=$BS" \
            $MAX_GPUS \
            "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress --batch-size-override $BS $COMMON" \
            $PORT
        PORT=$((PORT + 1))
    done
else
    sep "SECTION 3: BATCH SIZE SWEEP — SKIPPED"
    echo "Set BATCH_SIZES=\"4 8 16 32 64\" to enable."
fi

# ============================================================================
# SECTION 4: DATA LOADER — LOCAL vs GPFS
# Set DATA_DIR_LOCAL and DATA_DIR_GPFS to enable.
# ============================================================================

if [ -n "$DATA_DIR_LOCAL" ] && [ -n "$DATA_DIR_GPFS" ]; then
    sep "SECTION 4: DATA LOADER — LOCAL vs GPFS"

    run_training "BF16 data=GPFS" \
        $MAX_GPUS \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress --data-dir-override $DATA_DIR_GPFS $COMMON" \
        29530

    run_training "BF16 data=LOCAL" \
        $MAX_GPUS \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress --data-dir-override $DATA_DIR_LOCAL $COMMON" \
        29531
else
    sep "SECTION 4: DATA LOADER — SKIPPED"
    echo "Set DATA_DIR_GPFS and DATA_DIR_LOCAL to enable."
    echo "  e.g. DATA_DIR_GPFS=/project/pedramh/data DATA_DIR_LOCAL=/local/scratch/data"
fi

# ============================================================================
# SECTION 5: NSYS PROFILING (FLOPs, memory bandwidth, kernel analysis)
# ============================================================================

if command -v nsys &>/dev/null; then
    sep "SECTION 5: NSYS PROFILING"

    # Short run for profiling — limit steps via mode=test in the yaml,
    # or let it run a full epoch and kill after nsys captures enough.
    PROF_COMMON="--ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 --metrics-every 50 --accum-steps 1"

    # 1-GPU nsys (cleaner traces, no DDP noise)
    run_nsys "bf16_1gpu" \
        1 \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune $PROF_COMMON" \
        29540

    # 4-GPU nsys (captures DDP communication patterns)
    run_nsys "bf16_4gpu" \
        $MAX_GPUS \
        "--amp-dtype bf16 --torch-compile True --compile-mode max-autotune --compile-max-autotune --ddp-bucket-cap-mb 256 --ddp-fp16-compress $PROF_COMMON" \
        29541

    # FP16 nsys for precision comparison
    run_nsys "fp16_1gpu" \
        1 \
        "--amp-dtype fp16 --torch-compile True --compile-mode max-autotune --compile-max-autotune $PROF_COMMON" \
        29542

else
    sep "SECTION 5: NSYS PROFILING — SKIPPED"
    echo "nsys not found. Install NVIDIA Nsight Systems or use the NGC container."
fi

# ============================================================================
# SECTION 6: NCU PROFILING (per-kernel FLOPs + memory bandwidth)
# Single GPU, small number of kernels — this is slow.
# ============================================================================

if command -v ncu &>/dev/null; then
    sep "SECTION 6: NCU PROFILING (single GPU, slow)"

    PROF_COMMON="--ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 --metrics-every 50 --accum-steps 1"

    run_ncu "bf16" \
        "--amp-dtype bf16 --torch-compile False $PROF_COMMON" \
        29550

    run_ncu "fp16" \
        "--amp-dtype fp16 --torch-compile False $PROF_COMMON" \
        29551

else
    sep "SECTION 6: NCU PROFILING — SKIPPED"
    echo "ncu not found. Install NVIDIA Nsight Compute or use the NGC container."
    echo "NOTE: ncu is very slow. Run separately if needed."
fi

# ============================================================================
# SECTION 7: PYTORCH PROFILER (built-in, Chrome trace + FLOPs)
# Uses the --profiling flag in faster_train.py
# ============================================================================

sep "SECTION 7: PYTORCH PROFILER"

run_training "BF16 + PyTorch Profiler (1 GPU)" \
    1 \
    "--amp-dtype bf16 --torch-compile True --compile-mode reduce-overhead --profiling --profile-wait-steps 5 --profile-warmup-steps 3 --profile-active-steps 10 --profile-with-flops --profile-memory --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 --metrics-every 50 --accum-steps 1" \
    29560

run_training "BF16 + PyTorch Profiler (4 GPU)" \
    $MAX_GPUS \
    "--amp-dtype bf16 --torch-compile True --compile-mode reduce-overhead --ddp-bucket-cap-mb 256 --ddp-fp16-compress --profiling --profile-wait-steps 5 --profile-warmup-steps 3 --profile-active-steps 10 --profile-with-flops --profile-memory --ddp-static-graph --max-grad-norm 1.0 --log-every-n-steps 10 --metrics-every 50 --accum-steps 1" \
    29561

# ============================================================================
# DONE
# ============================================================================

sep "ALL BENCHMARKS COMPLETE"
echo "Paste this entire output for analysis."
echo ""
echo "Output files to collect:"
echo "  - This log (b300_results.log or SLURM .out file)"
echo "  - nsys_*.nsys-rep  (open with nsys-ui or 'nsys stats')"
echo "  - ncu_*.ncu-rep    (open with ncu-ui)"
echo "  - profiler_traces/ (Chrome traces from PyTorch profiler)"
echo ""
nvidia-smi 2>/dev/null || true
echo "=== Benchmarks Complete ==="