#### 1.1 Memory Optimsation

**Optimisation #1**: The intermediate (i.e., query, key, gate and value) tensors are pre-allocated for the entire sequence, which can be memory-intensive for long sequences.

**Resolution**:
- Sequential processing: We can process the RNN step-by-step without storing all intermediate results.
- Intermediate computations: Multiple intermediate tensors are created and stored simultaneously.

**Optimisation #2**: There are 4 separate `Linear` projection per step, which incurs high overhead

**Resolution**:
- Fuse the projections into one matmul, and splitting them into query, key, gate and value subsequently

#### 1.2 Training efficiency

**Optimisation #2**: The current SimpleRNN's `forward` function implementation processes stochastically (i.e., one data point at a time).

**Resolution**:
- Batch processing: We can make the model process multiple sentences in parallel.
- Variable training batch size: We can adjust the size of each training batch based on the longest sequence found in each batch.

**Optimisation #3**: 

#### 1.3 Inference Optimization

**Optimisation #3**: The current SimpleRNN's `forward` function requires the entire sequence at each timestep as it only outputs the logits for the entire sequence

**Resolution**:
- Stepwise Inference: We can make the model output the `cell_state` for the prefix input (i.e., `<en> I love apple <ja>`), and re-use and update the `cell_state` for subsequent prediction step.