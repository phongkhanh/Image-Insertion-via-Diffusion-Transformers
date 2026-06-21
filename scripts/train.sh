export XFL_CONFIG=experiments/config/insertanything.yaml

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1,2
accelerate launch --num_processes 2 --main_process_port 41353  -m src.train.train