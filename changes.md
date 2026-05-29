# Changelog

All changes made to the training code and model for B300 (Blackwell) FP8 support and bug fixes.

---

## `faster_train.py`

### FP8 / Transformer Engine support (6 changes)

**1. Optional `transformer_engine` import**

Added a try/except import at module level:
```python
try:
    import transformer_engine.pytorch as te
    _te_available = True
except ImportError:
    te = None
    _te_available = False
```
Reason: TE is not always installed. Making it optional lets the script run normally on H100s or any system without TE, falling back to BF16 automatically.

**2. Added `"fp8"` to `--amp-dtype` choices**

Changed:
```python
choices=["auto", "bf16", "fp16", "fp32"]
```
To:
```python
choices=["auto", "bf16", "fp16", "fp32", "fp8"]
```
Reason: Without this, argparse rejected `--amp-dtype fp8` before any Python code ran, causing an immediate crash in the B300 benchmark scripts that pass that flag.

**3. FP8 branch in `Trainer.__init__` dtype selection**

Added a new `elif amp_choice == "fp8":` branch with three sub-cases:
- TE available + GPU SM ≥ 9.0 (Hopper/Blackwell): sets `amp_dtype = bfloat16`, `use_fp8 = True`, and `params['use_transformer_engine'] = True`
- TE available but GPU too old: falls back to BF16 with a warning
- TE not installed: falls back to BF16 with a warning

Also added `self.use_fp8 = False` as a default at the top of the selection block.

Reason: The model (`networks/pangu.py`) already has full Transformer Engine layer support behind the `use_transformer_engine` param flag — using `te.Linear` and `te.LayerNorm` throughout. Setting `use_transformer_engine = True` here activates those TE layers at model construction time so that `te.fp8_autocast()` can actually quantize them to FP8. Without this link, the FP8 context manager was wrapping plain `nn.Linear` layers that TE cannot accelerate.

**4. `GradScaler` disabled for FP8**

Changed:
```python
self.scaler = GradScaler(enabled=(... and self.amp_dtype == torch.float16))
```
To:
```python
self.scaler = GradScaler(enabled=(... and self.amp_dtype == torch.float16 and not getattr(self, 'use_fp8', False)))
```
Reason: When FP8 is used via Transformer Engine, TE manages its own internal loss scaling. Running PyTorch's `GradScaler` on top conflicts with that and causes NaN gradients.

**5. `te.fp8_autocast()` context in `cal_loss`**

Changed the forward pass from a single `autocast` context to:
```python
_fp8_cm = te.fp8_autocast() if (self.use_fp8 and _te_available and te is not None) else contextlib.nullcontext()
with _fp8_cm, autocast(...):
    ...
```
Reason: `te.fp8_autocast()` is the mechanism that tells TE layers to operate in FP8 during the forward pass. Without it, even TE-built layers fall back to BF16. The outer `contextlib.nullcontext()` fallback makes this a no-op when FP8 is not active, so non-FP8 runs are unaffected.

**6. `params['use_transformer_engine'] = True` in the FP8 activation path**

Set inside the `use_fp8 = True` branch (see point 3 above).

Reason: The `PanguModel_Plasim` constructor reads `params.use_transformer_engine` to decide whether to instantiate `te.Linear`/`te.LayerNorm` or plain `nn.Linear`/`nn.LayerNorm` for every sub-module (`DownSample`, `UpSample`, `EarthSpecificBlock`, `EarthAttention3D`, `Mlp`). This flag must be set before `trainer.setup_model()` is called. Without it, FP8 autocast runs over non-TE layers with no effect.

---

## `networks/pangu.py`

### Bug fix: encoder 2 received wrong input

**File:** `networks/pangu.py`, `PanguModel_Plasim.forward()`, `upper_air_boundary=True` branch

Changed:
```python
surface_target = self.patchembed2d(surface)   # BUG
```
To:
```python
surface_target = self.patchembed2d(surface_target)
```
Reason: One line above, `surface_target` was correctly built from `target_surface`. The next line then overwrote it by embedding the *input* surface instead of the target. Encoder 2 (the VAE second encoder) was therefore learning from the wrong field, corrupting the VAE latent. This bug only affects runs with `upper_air_boundary: True`; the default config has it `False`.

### Bug fix: ocean variables masked with land mask

**File:** `networks/pangu.py`, `PanguModel_Plasim.forward()`, two locations (predict-delta and non-delta branches)

Changed (both occurrences):
```python
if self.has_ocean and self.mask_output:
    output_surface[:, self.num_surface_vars + self.num_land_vars:] = \
        self.land_mask(...).to(output_surface.dtype)   # BUG
```
To:
```python
if self.has_ocean and self.mask_output:
    output_surface[:, self.num_surface_vars + self.num_land_vars:] = \
        self.ocean_mask(...).to(output_surface.dtype)
```
Reason: Ocean-specific output variables (e.g., sea surface temperature) were being masked using the *land* mask instead of `self.ocean_mask`. The result was that SST and similar fields were zeroed/filled over land points when they should have been zeroed/filled over ocean points — the exact opposite. This bug only activates when `mask_output: True`; the default config has it `False`.

---

## `utils/patch_embed.py`

### Bug fix: self-referential level padding

**File:** `utils/patch_embed.py`, `PatchEmbed3D.__init__` and `PatchEmbed3D_Cyclic.__init__`

Changed (both classes):
```python
padding_front = l_pad - padding_front   # BUG: reads itself (still 0)
```
To:
```python
padding_front = l_pad - padding_back
```
Reason: `padding_front` was still `0` at the point of assignment (it was just initialized). So the original expression computed `l_pad - 0 = l_pad` regardless of `padding_back`. For the current default config (`patch_size=[2,4,4]`, 17 levels), `l_pad=1` and `padding_back=0`, so both expressions happen to give `1` — the bug is invisible. For any other patch size where `l_pad ≥ 2` (e.g., `patch_size=[4,...]` with a non-multiple level count), the total padding would be `l_pad + l_pad//2` instead of `l_pad`, producing the wrong output size.

### Removed debug print statements

**File:** `utils/patch_embed.py`, `PatchEmbed3D.__init__`

Removed:
```python
print('Pad sizes in 3D patch embedding')
print((padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back))
print('Patch Size')
print(patch_size)
```
Reason: These are leftover debug lines that fire on every model construction (including every DDP worker, every resume from checkpoint, and every benchmark run). They produce noise in logs and slow down startup on distributed jobs.

---

## `utils/patch_recovery.py`

### Bug fix: `AttributeError` in `PolarPad2d` and `PolarPad3d`

**File:** `utils/patch_recovery.py`, `PolarPad2d.__init__` and `PolarPad3d.__init__`

Added to both constructors:
```python
self.grid_has_poles = grid_has_poles
```
Reason: Both `forward()` methods read `self.grid_has_poles` but neither constructor ever stored it as an instance attribute. Any call to `forward()` would immediately raise `AttributeError: 'PolarPad2d' object has no attribute 'grid_has_poles'`. This affects all runs with `polar_pad: True` in the config.

---

## `utils/losses.py`

### Performance fix: `CRPSSpread` O(M²) loops → broadcasting

**File:** `utils/losses.py`, `Latitude_weighted_CRPSLoss.CRPSSpread()`

Replaced:
```python
spread = torch.zeros_like(input[0])
for i in range(input.shape[0]):
    for j in range(i+1, input.shape[0]):
        spread += 2 * weighted_mae(input[i], target[j], self.latitudes, reduction='none')
prefactor = 1 / (self.num_ensemble_members * (self.num_ensemble_members - 1))
spread = prefactor * spread
return spread
```
With:
```python
M = input.shape[0]
abs_diff = torch.abs(input.unsqueeze(1) - input.unsqueeze(0))
pair_sum = abs_diff.sum(dim=(0, 1))
reshape_shape = tuple(1 if i != len(pair_sum.shape) - 2 else -1 for i in range(len(pair_sum.shape)))
weight = torch.reshape(latitude_weighting_factor_torch(self.latitudes), reshape_shape)
return torch.mean(weight * pair_sum) / (M * (M - 1))
```
Reason: The original used nested Python loops over ensemble members, executing `M*(M-1)/2` separate `weighted_mae` calls. For `num_ensemble_members=4` this is 6 serial calls per sample per batch; for larger ensembles it grows quadratically. The replacement computes all pairwise absolute differences in a single broadcast operation `(M,1,...) - (1,M,...)`, then sums them in one pass. Diagonal entries are exactly zero so no masking is needed. The result is mathematically identical and runs entirely as fused GPU kernels.
