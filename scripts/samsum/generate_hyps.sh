cd $wkdir
dataset=samsum
# dual_encoder
# pretrained_model_path=/home/hannibal046/project/selfmem/results/finetune_samsum_memory_separate/lightning_logs/version_2/bart_large_best_ckpt

## concate
pretrained_model_path=/home/hannibal046/project/selfmem/results/finetune_samsum/lightning_logs/version_0/bart_large_best_ckpt
output_path=/tmp/hyps.txt
memory="--memory_path /home/hannibal046/project/selfmem/src/lightning_logs/version_0/test_hyps.txt  --memory_encoding concate"

CUDA_VISIBLE_DEVICES=1 python generate_hyps.py \
    --default_root_dir /tmp \
    --data_path ../data/samsum/test.jsonl \
    --output_path $output_path \
    --pretrained_model_path  $pretrained_model_path \
    --num_beams 5 \
    --train_max_src_len 512 \
    --gen_max_len 69 \
    --gen_min_len 9 \
    --per_device_eval_batch_size 10 \
    --accelerator gpu \
    --train_max_trg_len 80 \
    $memory


