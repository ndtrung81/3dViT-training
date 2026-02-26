## H100 NVL GPUs

`midway3-0423`: 4 H100 GPUs, each H100 GPU has 96 GB RAM, paired by NVLink: 0-1 and 2-3.

|      | GPU0  |  GPU1  |  GPU2   |  GPU3
|------|-------|--------|---------|--------
|GPU0  |   X   |   NV12 |   SYS  |  SYS |
|GPU1  |  NV12 |    X   |   SYS  |  SYS |
|GPU2  |  SYS |   SYS |    X    |  NV12 |
|GPU3  |  SYS |   SYS |   NV12  |   X   | 

* SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
* NV#  = Connection traversing a bonded set of # NVLinks

### Task 1 - Precision Sweep (4x H100, BF16)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| BF16 | 2.1497	| 276 | 3.72 |
| FP16 | 2.1858	| 287	| 3.66 |
| FP32 with TF32 | 3.3552 | 358 | 2.38 |
| FP32 pure (no TF32) | 4.8744 | 481 | 1.64 |

BF16 and FP16 perform nearly identically and are the fastest options. Enabling TF32 on top of FP32 recovers roughly 30% of the speed lost vs. mixed precision. Pure FP32 without TF32 is the slowest, taking more than twice as long per step as BF16.

### Task 2 - Batch Size Sweep (4x H100, BF16)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| Global batch = 4 (per-GPU = 1) | 1.2189	| 229	| 3.28 |
| Global batch = 8 (per-GPU = 2) | 2.2412 | 290 | 3.57 |
| Global batch = 12 (per-GPU = 3) | 3.5185 | 351 | 3.41 |

Throughput (samples/sec) peaks at batch size 8 and stays roughly flat at 12, indicating the GPUs are well-utilized at batch 8. Batch 4 underutilizes the hardware, yielding ~8% lower throughput.

### Task 4 - GPU scaling (BF16, per-GPU batch = 2)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| 1-GPU |	1.6366 | 510 | 1.22 |
| 2-GPU	| 1.8159 | 360 | 2.20 |
| 4-GPU	| 2.2045 | 286 | 3.63 |

Going from 1 to 4 GPUs delivers a 2.97X throughput improvement. The modest step-time increase with more GPUs reflects NVLink all-reduce overhead, which is well-contained - scaling efficiency is approximately 74% at 4 GPUs.

## B200 GPUs

DSAI b200: 4 B200 GPUs, each GPU has 148 SMs, 180 GB RAM. The 4 B200 GPUs are all connected with NVLink.

|      | GPU0  |  GPU1  |  GPU2   |  GPU3
|------|-------|--------|---------|--------
|GPU0  |   X   |   NV18 |   NV18  |  NV18 |
|GPU1  |  NV18 |    X   |   NV18  |  NV18 |
|GPU2  |  NV18 |   NV18 |    X    |  NV18 |
|GPU3  |  NV18 |   NV18 |   NV18  |   X   | 

* NV#  = Connection traversing a bonded set of # NVLinks

### Task 1 - Precision Sweep (4x B200, BF16)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| BF16 | 173|0.9577|4.18 |
| FP16 |167|0.7725|5.18 |
| FP32 with TF32 | 179 | 1.2354 | 3.24 |
| FP32 pure (no TF32) | 288 | 2.2017 | 1.8 |

FP16 is the fastest and 1.2X (5.18/4.18) faster than BF16. For this precision, B200 is 1.4X (5.18/3.66) faster than H100.

### Task 2 - Batch Size Sweep (4x B200, BF16)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| Global batch = 4 (per-GPU = 1) | 169 | 1.0191 | 3.92 |
| Global batch = 8 (per-GPU = 2) | 223 | 2.0552  | 3.89 |
| Global batch = 12 (per-GPU = 3) | 271 | 2.8422 | 4.22 |

### Task 4 - GPU scaling (BF16, per-GPU batch = 2)

| Run | sec/step | Wall Time | Samples/sec
|----------|---------|-------------|-------------|
| 1-GPU |	332 | 0.8975 | 2.23 |
| 2-GPU	| 246 | 1.2617 | 3.17 |
| 4-GPU	| 224 | 1.6209 | 4.94 |

A single B200 is 1.8X ( (2.23/1.22)) faster than H100 GPU.

## Summary

* B200 GPUs are more sensitive to FP16 vs BF16 than H100 GPUs.
* B200 GPUs are faster than H100 GPUs with the given workload and configurations (Note the NVLink mesh between B200 GPUs, whereas H100 GPUs are paired up.)
* A single B200 is 1.8X faster than a single H100 GPU.