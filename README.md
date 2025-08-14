## Problem Statement
Given the provided SimpleRNN implementation, analyze its performance characteristics and optimize it for efficient training and inference.

## Setup instructions

### Environment Setup

1. Create an Conda environment and install the required Python packages

```bash
conda create --name SimpleRNN python=3.12
pip3 install -r requirements.txt
```

### Benchmark Scripts

### Pipeline Scripts

## Key Findings

### Memory Optimsation #1

**Issue**
Frequent layer calls can become costly for long sequences.

**Resolution**:
- Checkpointing: Introduce a configurable flag to perform gradient checkpointing every K timesteps to reduce memory consumption.

### Training Optimisation #1

**Issue**
Per-timestep Python loops store intermediate tensors for each step, increasing memory usage and preventing efficient GPU utilization.

**Resolution**:
- Vectorized (batched) recurrence computation: Replace the loop with a vectorized prefix-scan formulation that computes all timesteps in parallel using cumulative products and sums, eliminating large intermediate lists and enabling fused GPU operations.

### Training Optimisation #2

**Issue**
Each timestep involves four separate Linear projection operations, resulting in considerable computational overhead.

**Resolution**:
- Projection Fusion: Combine all projections into a single matrix multiplication, then split the output into query, key, gate, and value components.

### Training Optimisation #3

**Issue**
The current forward method in SimpleRNN processes one data point at a time, leading to inefficient training.

**Resolution**:
- Batch Processing: Enable the model to process multiple sentences in parallel.
- Dynamic Batch Sizing: Adjust batch sizes according to the longest sequence within each batch to optimise resource usage.
- Dataloader: Spawn multiple background worker processes to prepare data while the GPU is busy training.

### Inference Optimisation #1:

**Issue**
The current forward function requires the full sequence at each timestep and only returns logits for the entire sequence, limiting efficiency during inference.

**Resolution**:
- Stepwise Inference: Modify the model to output the `cell_state` for a given prefix input (e.g., <en> I love apple <ja>), enabling re-use and incremental updates of `cell_state` for subsequent prediction steps.

## Performance metrics

### Memory & Training Efficiency

| S/N | Modifications.    | Improvement Cat. |Runtime duration | Peak memory usage | 
|-----|-------------------|------------------|------------------|-------------------|
| 1| Simple (Batch = 1)  | N/A         | 3.0147 ± 0.0362 seconds  | 20.51 MB          |
| 2| Fused Projection (Batch = 1)  | Runtime   | 0.0947 ± 0.0514 seconds                  |38.59 MB
| 3| 2 + Batch Processing (Batch = 32)   | Runtime                 |  0.0271 ± 0.0504 seconds                  | 929.57 MB|
| 4 | 2 + 3 + Gradient Checkpointing | Memory |  0.0791 ± 0.2113 seconds | 331.82 MB|

### Inference Efficiency

| S/N | Modifications.    | Improvement Cat. |Runtime duration | Peak memory usage | 
|-----|-------------------|------------------|------------------|-------------------|
| 1| Simple (Batch = 1)   | N/A         |               |         |
| 2| Stepwise Inference (Batch = 1) | Memory   |                  |                   |

### Benchmark Performance

### Demonstrations

#### Rotten Tomato

#### Daily Dialogue

## Discussions

### Architecture Comparison
Compare **Self-Attention** vs **SimpleRNN** in terms of:

- **Computational complexity** (Big-O analysis)  
- **Parallelization potential** 
- **Long-range dependency modeling**  
- **Memory scaling** with sequence length  
- **Training characteristics**  
- *(You may include other relevant factors)*

### Scaling Discussion
To scale SimpleRNN training across multiple GPUs effectively, we can consider doing the following:

1. **Data Parallelism**:
- Use `torch.nn.DistributedDataParallel (DDP)` for efficient and scalable training, as it minimizes overhead compared to `torch.nn.DataParallel`.
  - DDP split the input batch across GPUs, run forward and backward passes independently, and then synchronize gradients.

2. **Efficient Batch**:
- Pad sequences within each mini-batch to similar lengths to minimize idle time across GPUs.
  - Use bucketing to group similar-length sequences, improving computational efficiency.

3. **Model Sharding**:
- When the model is too large to fit on a single GPU, place different layers (or parts of the model) are placed on different GPUs.
- *Caveat: SimpleRNNs are typically lightweight; this method adds communication overhead and is less beneficial unless the model is very deep or memory-intensive.*

## Expected Deliverables
[X] **Optimized SimpleRNN** with documented performance improvements  
[X] **Benchmark results** comparing original vs optimized versions  
[X] **Training demonstration** on sample data showcasing improvements 
[X] **Technical analysis** covering architecture comparisons and scaling strategies  
[X] **Clean, well-documented code** with explanations for each optimization