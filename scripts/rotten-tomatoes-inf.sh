
python3 test.py \
    --model_path models/rotten-tomatoes/SimpleRNN_2layers-128hidden-32key-64value.pt \
    --model_name "simple" \
    --dataset_name "rotten-tomatoes"

python3 test.py \
    --model_path models/rotten-tomatoes/ModifiedRNN_2layers-128hidden-32key-64value.pt \
    --model_name "modified" \
    --dataset_name "rotten-tomatoes"