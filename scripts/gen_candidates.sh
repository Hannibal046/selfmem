wkdir=/home/hannibal046/project/selfmem/src
# wkdir=/wfs/mmchongqingwfssz/user_mqxincheng/selfmem/src
cd $wkdir

## samsum
for SPLIT in train dev test
do
    CUDA_VISIBLE_DEVICES=0,1 python $wkdir/gen_candidates.py \
        --data_path /data/samsum/${SPLIT}.jsonl \
        --memory_path /data/samsum/memory/bm25/${SPLIT}.txt \
        --output_dir ../data/samsum/raw/${SPLIT} \
        --train_max_src_len 512 \
        --accelerator gpu \
        --pretrained_model /home/hannibal046/project/selfmem/results/finetune_samsum/lightning_logs/version_4/bart_best_ckpt_huggingface \
        --num_return_sequences 128 \
        --num_beam_groups 16 \
        --diversity_penalty 0.1 \
        --num_beams 128 \
        --length_penalty 0.6 \
        --gen_max_len 69 \
        --gen_min_len 9 \
        --no_repeat_ngram_size 3 \
        --early_stopping \
        --per_device_eval_batch_size 2 
done
