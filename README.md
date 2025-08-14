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

## Key Findings

### Memory Optimisation #1

**Issue**
The intermediate tensors (i.e., query, key, gate, and value) are pre-allocated for the entire sequence. For long sequences, this approach can result in substantial memory usage.

**Resolution**:
- Sequential Processing: Process the RNN step-by-step to avoid storing all intermediate results at once.
- Intermediate Computations: Reduce simultaneous creation and storage of multiple intermediate tensors.

### Memory Optimisation #2

**Issue**
Each timestep involves four separate Linear projection operations, resulting in considerable computational overhead.

**Resolution**:
- Projection Fusion: Combine all projections into a single matrix multiplication, then split the output into query, key, gate, and value components.

### Memory Optimsation #3

**Issue**
Frequent layer calls can become costly for long sequences.

**Resolution**:
- Checkpointing: Introduce a configurable flag to perform gradient checkpointing every K timesteps to reduce memory consumption.

### Training Efficiency #1

**Issue**
The current forward method in SimpleRNN processes one data point at a time, leading to inefficient training.

**Resolution**:
- Batch Processing: Enable the model to process multiple sentences in parallel.
- Dynamic Batch Sizing: Adjust batch sizes according to the longest sequence within each batch to optimise resource usage.

### Training Efficiency #2

**Issue**
The value projection is linear, which can cause instability due to unbounded outputs.

**Resolution**
- Activation Adjustment: Use tanh for the value projection, allowing bounded, sign-preserving updates and improving stability.

### Training Efficiency #3: 

**Issue**
Exploding gradients can cause numerical instability and hinder the training process.

**Resolution**
- Gradient Clipping: Apply gradient clipping to prevent gradients from becoming excessively large.

```python
torch.nn.utils.clip_grad_norm
```

**References**
https://www.geeksforgeeks.org/deep-learning/gradient-clipping-in-pytorch-methods-implementation-and-best-practices/

### Training Efficiency #4: 

**Issue**
The model may struggle to capture long-range dependencies.

**Resolution**
- Luong Attention: Incorporate Luong attention to better handle long dependency relationships in the data.

### Inference Optimisation #1:

**Issue**
The current forward function requires the full sequence at each timestep and only returns logits for the entire sequence, limiting efficiency during inference.

**Resolution**:
- Stepwise Inference: Modify the model to output the `cell_state` for a given prefix input (e.g., <en> I love apple <ja>), enabling re-use and incremental updates of `cell_state` for subsequent prediction steps.

## Performance metrics

### Memory & Training Efficiency

### Inference Efficiency

### Benchmark Performance

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
[] **Optimized SimpleRNN** with documented performance improvements  
[] **Benchmark results** comparing original vs optimized versions  
[] **Training demonstration** on sample data showcasing improvements  
[] **Technical analysis** covering architecture comparisons and scaling strategies  
[] **Clean, well-documented code** with explanations for each optimization