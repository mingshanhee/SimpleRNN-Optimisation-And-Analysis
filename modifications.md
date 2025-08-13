#### 1.1 Memory Optimsation

**Issue #1**: The intermediate (i.e., query, key, gate and value) tensors are pre-allocated for the entire sequence, which can be memory-intensive for long sequences.

**Resolution**:
- Sequential processing: We can process the RNN step-by-step without storing all intermediate results.
- Intermediate computations: Multiple intermediate tensors are created and stored simultaneously.
- Gradient checkpointing: For training, we can implement gradient checkpointing to trade computation for memory.

#### 1.2 Training efficiency

**Issue #2**: The current SimpleRNN's `forward` function implementation processes stochastically (i.e., one data point at a time).

**Resolution**:
- Batch processing: We can make the model process multiple sentences in parallel.