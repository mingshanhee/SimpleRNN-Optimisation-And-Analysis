#### 1.1 Memory Optimsation

**Issue #1**: Pre-allocated tensor storage: The output tensor is pre-allocated for the entire sequence, which can be memory-intensive for long sequences.

**Resolution**:
- Intermediate computations: Multiple intermediate tensors are created and stored simultaneously.

- In-place operations: We can use more in-place operations to reduce temporary tensor allocation.

- Sequential processing: We can process the RNN step-by-step without storing all intermediate results.

- Gradient checkpointing: For training, we can implement gradient checkpointing to trade computation for memory.