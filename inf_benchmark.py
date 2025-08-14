import argparse
import numpy as np

import torch
import time
import tracemalloc  # for CPU memory tracking
import torch.nn as nn
import torch.nn.functional as F

from modified_rnn import LM as ModifiedLM
from simple_rnn import LM as SimpleLM

# Dataloader
from torch.utils.data import DataLoader
from decode_utils import simple_inference, modified_inference

def main():
    vocab_size = 512
    hidden_dim = 128
    key_dim = 32
    value_dim = 64
    output_dim = 128
    num_layers = 2

    parser = argparse.ArgumentParser(description='Train SimpleRNN language model on BSD dataset')
    parser.add_argument('--model_name', type=str, required=True, choices=['simple', 'modified'], help='Model type to use')
    args = parser.parse_args()
    
    # Test with batch processing
    size, seq_length = 1, 256
    prefix_x = torch.randint(0, vocab_size, (size, seq_length))

    if args.model_name == 'simple':
        model = SimpleLM(
            vocab_size=vocab_size, 
            hidden_dim=hidden_dim, 
            key_dim=key_dim, 
            value_dim=value_dim, 
            output_dim=output_dim, 
            num_layers=num_layers
        )

    else:
        model = ModifiedLM(
            vocab_size, 
            hidden_dim, 
            key_dim, 
            value_dim, 
            output_dim, 
            num_layers,
            fused_projection=True,
            use_gradient_checkpointing=True
        )

    # Keep DataLoader the same for both models
    dataloader = DataLoader(prefix_x, batch_size=1)

        
    # Lists to store results
    runtimes = []
    peak_memories = []

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for benchmarking")
    else:
        device = torch.device("cpu")
        print("Using CPU for benchmarking")
    
    model = model.to(device)

    # Run benchmarking multiple times
    repeat = 10
    for i in range(repeat):
        print(f"Running iteration {i+1}/{repeat}...")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()  # Clear cache before each run
            
            start_time = time.time()
            
            for batched_input in dataloader:
                batched_input = batched_input.to(device)
                # Forward pass
                if args.model_name == 'simple':
                    batched_output = simple_inference(model, batched_input.squeeze(0)) 
                else:
                    batched_output = modified_inference(model, batched_input)

            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
            
        else:
            tracemalloc.start()
            start_time = time.time()

            for batched_input in dataloader:
                batched_input = batched_input.to(device)
                # Forward pass
                if args.model_name == 'simple':
                    batched_output = simple_inference(model, batched_input.squeeze(0)) 
                else:
                    batched_output = modified_inference(model, batched_input)

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024 ** 2)  # in MB
            tracemalloc.stop()
        
        runtime = end_time - start_time
        runtimes.append(runtime)
        peak_memories.append(peak_memory)
    
    # Calculate statistics
    avg_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    avg_memory = np.mean(peak_memories)
    std_memory = np.std(peak_memories)

    print(f"\nBenchmark Results (n={repeat}):")
    print(f"Batched input shape: {batched_input.shape}")
    print(f"Runtime - Average: {avg_runtime:.4f} ± {std_runtime:.4f} seconds")
    print(f"Peak Memory - Average: {avg_memory:.2f} ± {std_memory:.2f} MB")

    


if __name__ == "__main__":
    main()
