CUDA_VISIBLE_DEVICES=4 python3 train.py \
    --dataset_name "dailydialog" \
    --model_name "modified" \
    --learning_rate 1e-3 \
    --num_layers 2 \
    --hidden_dim 128 \
    --key_dim 32 \
    --value_dim 64 \
    --output_dim 128 \
    --batch_size 32 \
    --num_epochs 20 \
    --save_path "models/dailydialog/ModifiedRNN_{params}.pt" \
    --use_wandb \
    --fused_projection > /dev/null &

pid1=$!

wait $pid1

