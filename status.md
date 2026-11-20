# Training Optimization Status

**Date**: 2026-06-01 — **updated 2026-07-13: Issues 1 & 2 implemented** (on `experiment` branch)  
**Branch**: experiment  
**Key files**: `faster_train.py`, `networks/pangu.py`, `utils/losses.py`, `config/exp1.yaml`

---

## Setup

- Model: `PanguModel_Plasim` — dual-encoder VAE-style architecture with 3D windowed attention (Swin-style)
- Target hardware: 8× H200 or B200 GPUs with NVLink switches
- Training config: `config/exp1.yaml`, section `S2S`
- Global batch size: 8 (= 1 per GPU with 8 GPUs)
- Ensemble: `num_ensemble_members: 4` (CRPS loss), which expands per-GPU batch to effective B=4 inside the model
- AMP: BF16 by default; FP8 via Transformer Engine supported but not enabled

---

## Issue 1 (Critical): Redundant Encoder Computation from Ensemble Expansion

**Status: ✅ IMPLEMENTED 2026-07-13** — see "Implemented fix" below; verified by `test/ensemble_latent_test.py`.

### Root cause

`_prepare_inputs_batch` (`faster_train.py:871`) calls `to_ensemble_batch(data, 4)` on **all inputs and targets** before passing them to the model. With 1 sample per GPU and 4 ensemble members, the model receives B=4 where all 4 inputs are byte-for-byte identical.

Because the encoder is deterministic, **3 of the 4 encoder forward passes produce identical `(mu, sigma)` pairs**. The ensemble diversity only comes from `torch.randn_like` inside `reparameterize`, which does not require running the encoder 4 times.

During training, two encoders run (`layer1/2/3` and `layer1_e2/2_e2/3_e3`), so the total wasted work is **6 out of 8 encoder forward passes per training step**.

This is also why `batch_size=3` OOMs on H100 — the effective model batch is 3×4=12.

### Memory budget (relative units)

| Config | Peak activation memory |
|---|---|
| Current (B=1, E=4, both encoders at B×E=4) | 8× encoder-unit |
| After latent-space fix (encoders at B=1, decoder at B×E=4) | ~3× encoder-unit |
| After latent-space fix + block checkpointing fix | ~1.5× encoder-unit |

### Implemented fix (2026-07-13) — ensemble sampling moved to the latent bottleneck

Note: the CRPS loss groups members as `input.view(B, E, ...)`, i.e. **members of a
sample must be contiguous** — so all expansions use `repeat_interleave` ordering
(`[s0m0, s0m1, ..., s1m0, ...]`), not `torch.cat`/`repeat` as originally sketched.

**`networks/pangu.py`**
- `PanguModel_Plasim.forward` gained a `num_ensemble_members=1` kwarg. With E>1 both
  encoders run at batch B; E members are sampled at the bottleneck
  (`reparameterize(mu.repeat_interleave(E, 0), sigma.repeat_interleave(E, 0))`); then
  `x`/`skip` are `repeat_interleave`d so only the decoder runs at B×E.
- `reparameterize` now draws `torch.randn(std.size(), ...)` instead of `randn_like(std)`:
  `randn_like` fills values in the input's *memory order*, so the noise realization
  silently depended on sigma's strides (conv output is permuted; `repeat_interleave`
  output is contiguous). Statistically identical, but layout-independent.
- Removed the unused `norm_e2 = self.reparameterize(mu_e2, sigma_e2)` draw (dead compute,
  one large randn per step).

**`faster_train.py`**
- `to_ensemble_batch` now uses `repeat_interleave` (same ordering as before, no
  `torch.ones` allocation + multiply).
- `_prepare_inputs_batch` expands **targets only** to [B*E] (the loss consumes them);
  inputs stay at [B].
- `cal_loss` passes `target[::E]` views to the model (encoder 2 at batch B) and
  `num_ensemble_members=E`; losses still compare [B*E] outputs vs [B*E] targets.
- `constant_boundary_data` stays at [B] for training; a `constant_boundary_data_ens`
  copy at [B*E] serves the validation loop, which is behaviorally unchanged (it still
  expands its inputs — optimizing the first rollout step is future work under Issue 3).
- The `predict_delta` metrics branch expands inputs locally before calling the
  integrator (outputs/targets are [B*E] there).

**Verification** — `test/ensemble_latent_test.py` (no dataset needed; run from repo root):
compares the new path (inputs at B, E sampled at bottleneck) against the legacy path
(all inputs pre-expanded to B*E) with identical seeds. Result on CPU, B=1, E=2:
mu/sigma/mu_e2/sigma_e2 **bitwise identical**, decoder outputs equal to float32
tolerance (max diff ~3e-7), ensemble members diverse, and the training forward is
already ~1.5× faster (13.5s vs 20.5s); savings grow with E=4 and free ~2.5× peak
activation memory for larger real batch sizes.

The CRPS loss handles ensemble grouping internally — **no changes to `losses.py`**.

---

## Issue 2 (Bug): Block-Level Checkpointing is Dead Code

**Status: ✅ IMPLEMENTED 2026-07-13** — `EarthSpecificLayer.forward` now checkpoints
per block when `checkpointing > 2` and the module is in training mode (the original
code was commented out because it referenced a `train` variable that doesn't exist in
that scope; `self.training` is the correct signal). Verified on a standalone
`EarthSpecificLayer`: outputs, input grads, and parameter grads are bitwise identical
with checkpointing on vs off.

**Not active by default**: current configs set `checkpointing: 2`, which preserves
today's behavior exactly. Set `checkpointing: 3` in the YAML to enable block-level
checkpointing (trades ~20-30% extra recompute in backward for a large activation-memory
reduction across `layer1/2/3/4` and the encoder-2 layers — combines with the Issue 1
fix to allow much larger per-GPU batch sizes).

### Root cause

`EarthSpecificLayer.forward` (`pangu.py:837–843`) has the per-block checkpointing call **commented out**:

```python
def forward(self, x):
    for blk in self.blocks:
        # if self.checkpointing > 2 and train:      ← DEAD
        #     x = checkpoint(blk, x, ...)           ← DEAD
        # else:                                     ← DEAD
        x = blk(x)   # always runs without checkpointing
    return x
```

The config sets `checkpointing: 2`, but this has no effect on the transformer blocks. Checkpointing is only active on `patchrecovery2d/3d` and the VAE `layer2_e2/layer3_e3` calls.

### Fix

```python
# networks/pangu.py:837
def forward(self, x):
    for blk in self.blocks:
        if self.checkpointing > 2 and self.training:
            x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
        else:
            x = blk(x)
    return x
```

Note: uses `self.training` instead of a `train` argument since `nn.Module` already tracks this via `.train()` / `.eval()`.

---

## Issue 3: Validation Loop Runs in Float32

`validate_one_epoch` (`faster_train.py:1027`) wraps the loop in `torch.inference_mode()` but **never enters the BF16 autocast context**. Every model forward during validation runs in float32, doubling validation time compared to BF16.

### Fix

Add an autocast context around the model call inside the validation loop:

```python
# faster_train.py ~line 1113
with autocast(device_type="cuda", dtype=self.amp_dtype,
              enabled=torch.cuda.is_available() and self.amp_dtype != torch.float32):
    val_output_surface, val_output_upper_air, ... = self.model(...)
```

---

## Issue 4: Per-Metric `all_reduce` in Diagnostics

`diagnostic_log_per_iter` (`faster_train.py:950–961`) issues one `dist.all_reduce` call per logged metric — potentially 50+ individual collectives per training step. Even with NVLink's high bandwidth, many small collectives add fixed per-call latency overhead.

### Fix

Stack all metric tensors into a single tensor, do one `all_reduce`, then unstack:

```python
keys = sorted(diagnostic_logs.keys())
vals = torch.stack([
    diagnostic_logs[k] if isinstance(diagnostic_logs[k], torch.Tensor)
    else torch.tensor(diagnostic_logs[k], device=self.device, dtype=torch.float32)
    for k in keys
])
dist.all_reduce(vals)
vals /= dist.get_world_size()
for i, k in enumerate(keys):
    diagnostic_logs[k] = float(vals[i])
```

---

## Issue 5: Earth Position Bias Recomputed Every Forward Pass

`EarthAttention3D.forward` (`pangu.py:1044–1050`) re-gathers and permutes `earth_position_bias_table` on every call. The table is a learned parameter (changes during training) but its gathered/permuted form is the same within a training step.

### Fix

Cache as a non-persistent buffer, invalidate after each optimizer step (or cache only during inference):

```python
# __init__
self.register_buffer("_cached_earth_bias", None, persistent=False)

# forward — replace lines 1044-1050
if self._cached_earth_bias is None:
    eb = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1] * self.window_size[2],
        self.window_size[0] * self.window_size[1] * self.window_size[2],
        self.type_of_windows, -1
    ).permute(3, 2, 0, 1).contiguous().unsqueeze(0)
    self._cached_earth_bias = eb
earth_position_bias = self._cached_earth_bias
```

Invalidate in the training loop after `optimizer.step()`:
```python
for m in self.model.modules():
    if hasattr(m, '_cached_earth_bias'):
        m._cached_earth_bias = None
```

---

## Lower-Priority / Launch Script Changes

These don't require code changes — they're flags and environment variables for the launch script.

### torch.compile (max-autotune)
```bash
torchrun ... faster_train.py \
    --torch-compile \
    --compile-max-autotune \
    --compile-loss \
    --amp-dtype bf16
```
`max-autotune` generates fused CUDA kernels and gives 10–25% throughput improvement after warmup. Currently the default is `reduce-overhead`.

### FP8 via Transformer Engine (H200/B200)
The model already conditionally uses `te.Linear` and `te.LayerNorm` when `use_transformer_engine: True` in the config. For H200 (SM 9.0) and B200 (SM 10.0):
```yaml
# config/exp1.yaml
use_transformer_engine: True
```
```bash
--amp-dtype fp8
```
Note: `DotProductAttention` from TE is present but commented out at `pangu.py:1021`. Enabling it would give additional speedup on the attention projections.

### NCCL tuning for NVLink
Add to the torchrun launch script or SLURM job:
```bash
export NCCL_NVLS_ENABLE=1        # NVLink SHARP for faster all-reduce
export NCCL_P2P_LEVEL=NVL        # Force NVLink peer-to-peer
export NCCL_NET_GDR_LEVEL=SYS    # GPU Direct RDMA
export NCCL_IB_DISABLE=1         # Skip InfiniBand probe if not present
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_MIN_NCHANNELS=32     # More channels for B200 NVLink 5.0
```

### DDP static graph
```python
# faster_train.py ~line 492 — safe when forward graph is identical each iteration
static_graph=True,
```
Currently `ddp_static_graph=False` is the default. Enabling reduces DDP's per-step bookkeeping overhead. Verify no dynamic control flow changes which parameters are used across iterations.

### num_data_workers
Config sets `num_data_workers: 8`. H200/B200 nodes have many CPU cores; increase to 16–32 if I/O is a bottleneck. Note: the script also sets `params['num_workers']` (a different key) which doesn't reach the DataLoader — only `num_data_workers` matters.

---

## Priority Order

1. ~~**Issue 1** (ensemble latent fix)~~ — ✅ done 2026-07-13
2. ~~**Issue 2** (checkpointing bug)~~ — ✅ done 2026-07-13 (opt-in: set `checkpointing: 3`)
3. **Issue 3** (validation BF16) — easy fix, ~2× validation speedup; also fold in
   latent-sampling for the first rollout step (validation still expands inputs to B*E)
4. **Issue 4** (batched all_reduce) — easy fix, reduces collective overhead under NVLink
5. **Issue 5** (bias cache) — moderate fix, reduces redundant memory bandwidth per step
6. **Launch script tuning** — NCCL vars, `--torch-compile --compile-max-autotune`, FP8

## Additional findings (2026-07-13 review)

- `num_levels: 17upper_air_boundary` — paste error in 8 of 10 configs (parses as a
  string). Latent: only consumed via `utils/integrate.py:6` when `predict_delta: True`.
  Should be `17`.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` in `h100_training.sh` / `b300_training_dsai.sh`
  forces all kernels into a single hardware queue and hurts DDP compute/comm overlap
  (it's a Megatron sequence-parallel trick). Recommend removing (default is 8).
- Benchmark scripts never pass `--torch-compile`; the nsys section passes
  `--compile-max-autotune` alone, which is a no-op (`should_compile` keys off
  `torch_compile`). 10–25% throughput untested.
- `NCCL_NVLS_ENABLE=1` missing on the Blackwell nodes (full NVSwitch mesh supports
  NVLink SHARP). Not applicable on the H100 NVL pairs.
