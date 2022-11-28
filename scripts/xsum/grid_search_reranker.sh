cd /mnt/wfs/mmchongqingwfssz/user_mqxincheng/selfmem/src
nvidia-smi
pretrained_model=../pretrained_model/$1


for temperature in 0.001 0.005 0.007 0.1 0.2 0.4 0.5 0.7 1.0
do
    python train_reranker.py \
        --config_path config/xsum/train_reranker.yaml \
        --candidate_dir ../candidates/xsum/concate_pegasus_xsum/diverse \
        --pretrained_model_path $pretrained_model \
        --default_root_dir ../results/reranker/xsum/concate/$1 \
        --enable_progress_bar false \
        --precision 16 \
        --accumulate_grad_batches 8 \
        --logging_steps 100 \
        --max_epochs 10 \
        --val_check_interval 0.5 \
        --num_candidates 12 \
        --cheat \
        --temperature $temperature
done
