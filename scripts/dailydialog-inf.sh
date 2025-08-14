
python3 test.py \
    --model_path models/dailydialog/SimpleRNN_2layers-128hidden-32key-64value.pt \
    --model_name "simple" \
    --dataset_name "dailydialog"

python3 test.py \
    --model_path models/dailydialog/ModifiedRNN_2layers-128hidden-32key-64value.pt \
    --model_name "modified" \
    --dataset_name "dailydialog"