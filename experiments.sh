
# Model experiments
ex=01
python main.py --save_root "./model-store/ex${ex}" --print_interval 0.2 \
    --n_layers 8 --epoch 512 --context_size 128 \
    --wb_flag --wb_project 'gpt' --wb_notes "base model" --wb_tag "base"

# Increase weights on special tokens
ex=02
python main.py --save_root "./model-store/ex${ex}" --print_interval 0.2 \
    --n_layers 8 --epoch 512 --context_size 128 --sp_weight 2.0 \
    --wb_flag --wb_project 'gpt' --wb_notes "Adjust class weights" --wb_tag "class_weight"
