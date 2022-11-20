cd $wkdir
dataset=cnndm
pretrained_model_path=../pretrained_model/bart_large_cnn
output_dir=../results/finetune_cnndm_memory_separate_bart

python train_generator.py \
    --data_dir ../data/$dataset \
    --memory_dir ../data/$dataset/memory/bm25 \
    --memory_encoding separate \
    --pretrained_model_path  $pretrained_model_path \
    --lr 5e-3 \
    --accelerator gpu \
    --max_epochs 20 \
    --label_smoothing 0.1 \
    --warmup_steps 6000 \
    --default_root_dir $output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 12 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 1.0 \
    --gen_max_len 140 \
    --gen_min_len 55 \
    --num_beams 4 \
    --train_max_trg_len 120 \
    --train_max_src_len 1024 \
    --length_penalty 2.0 \
    --weight_decay 0 \
    --no_repeat_ngram_size 3 \
    --early_stopping \
    --enable_progress_bar true \
    --early_stop_patience 5 \
    --val_check_interval 0.5

# CUDA_VISIBLE_DEVICES=0,1 python generate_hyps.py \
#     --default_root_dir /tmp \
#     --data_path ../data/samsum/test.jsonl \
#     --memory_path ../data/samsum/memory/bm25/test.txt \
#     --memory_encoding separate \
#     --output_path ../results/samsum/test.hyps \
#     --pretrained_model_path /home/hannibal046/project/selfmem/src/finetune_samsum_memory_separate/lightning_logs/version_2/bart_large_best_ckpt \
#     --num_beams 5 \
#     --train_max_src_len 512 \
#     --gen_max_len 69 \
#     --gen_min_len 9 \
#     --per_device_eval_batch_size 20 \
#     --accelerator gpu \
#     --train_max_trg_len 80 \


