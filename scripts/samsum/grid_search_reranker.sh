cd /mnt/wfs/mmchongqingwfssz/user_mqxincheng/selfmem/src
pretraind_model=$1

for temperature in 0.001 0.003 0.007 0.1 0.3 0.5 0.6 0.7 1.0
do
    python train_reranker.py \
        --config_path config/samsum/train_reranker.yaml \
        --candidate_dir ../candidates/samsum/concate/diverse \
        --pretrained_model_path ../pretrained_model/$pretraind_model \
        --max_epochs 20 \
        --default_root_dir ../results/reranker/samsum/concate_bart_large/diverse/$pretraind_model \
        --enable_progress_bar false \
        --cheat \
        --num_candidates 6 \
        --per_device_train_batch_size 2 \
        --early_stop_patience 5 \
        --temperature $temperature
done
