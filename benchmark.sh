
# echo "Running benchmark for SimpleRNN model..."
# python3 train_benchmark.py \
#     --model_name "simple"

# echo "Running benchmark for Modified SimpleRNN model..."
# echo "+ Fused Projection"
# python3 train_benchmark.py \
#     --model_name "modified" \
#     --batch_size 32 

echo "Running benchmark for Modified SimpleRNN model..."
echo "+ Batch Processing"
python3 train_benchmark.py \
    --model_name "modified" \
    --batch_size 32 \
    --fused_projection

echo "Running benchmark for Modified SimpleRNN model..."
echo "+ Batch Processing"
python3 train_benchmark.py \
    --model_name "modified" \
    --batch_size 32 \
    --fused_projection \
    --use_gradient_checkpointing

# # ##### INFERENCE BENCHMARKS #####
# echo "Running inference benchmark for SimpleRNN model..."
# python3 inf_benchmark.py \
#     --model_name "simple"

# echo "Running inference benchmark for Modified SimpleRNN model..."
# python3 inf_benchmark.py \
#     --model_name "modified"