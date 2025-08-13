## 1.1 Memory Optimsation

### Optimisation #1

**Issue**
The intermediate (i.e., query, key, gate and value) tensors are pre-allocated for the entire sequence, which can be memory-intensive for long sequences.

**Resolution**:
- Sequential processing: We can process the RNN step-by-step without storing all intermediate results.
- Intermediate computations: Multiple intermediate tensors are created and stored simultaneously.

### **Optimisation #2** 

**Issue**
There are 4 separate `Linear` projection per step, which incurs high overhead

**Resolution**:
- Fuse the projections into one matmul, and splitting them into query, key, gate and value subsequently

### **Optimisation #3** 

**Issue**
layer calls tend to get expensive over long sequences

**Resolution**:
- Checkpoint every K steps of time.

## 1.2 Training efficiency

### **Optimisation #4** 

**Issue**
The current SimpleRNN's `forward` function implementation processes stochastically (i.e., one data point at a time).

**Resolution**:
- Batch processing: We can make the model process multiple sentences in parallel.
- Variable training batch size: We can adjust the size of each training batch based on the longest sequence found in each batch.

### **Optimisation #5**

**Issue**
The key projection is using sigmoid function, while the value projection is a linear function. The sigmoid function makes the key projection loses sign and saturates, while linear function (no activation) hurts training stability due to unbounded magnitute

**Resolution**
Add and change the activation to `tanh` function, allowing for bounded and sign-preserving updates 

### Optimisation #6**: 

**Issue**
Exploding gradient can lead to numerical instability and iompede the training process.

**Resolution**
Use gradient clipping to prevent the gradients from becoming excessively large during the training of neural networks.

```python
torch.nn.utils.clip_grad_norm
```

**References**
https://www.geeksforgeeks.org/deep-learning/gradient-clipping-in-pytorch-methods-implementation-and-best-practices/

#### 1.3 Inference Optimization

### **Optimisation #7**: 

**Issue**
The current SimpleRNN's `forward` function requires the entire sequence at each timestep as it only outputs the logits for the entire sequence

**Resolution**:
- Stepwise Inference: We can make the model output the `cell_state` for the prefix input (i.e., `<en> I love apple <ja>`), and re-use and update the `cell_state` for subsequent prediction step.