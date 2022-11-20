
wkdir=/home/hannibal046/project/selfmem/src
# wkdir=/wfs/mmchongqingwfssz/user_mqxincheng/selfmem/src
cd $wkdir
dataset=wikitext2
dataset_path=/data
pretrained_model_path=/data/pretrained_model/bart_base

output_dir=../results/finetune_wikitext
CUDA_VISIBLE_DEVICES=0 python train_summarier_lighting.py \
    --data_dir $dataset_path/$dataset \
    --memory_dir $dataset_path/$dataset/memory/bm25 \
    --pretrained_model_path  $pretrained_model_path \
    --lr 6e-3 \
    --accelerator gpu \
    --max_epochs 10 \
    --label_smoothing 0.0 \
    --warmup_steps 4000 \
    --default_root_dir $output_dir \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 15 \
    --accumulate_grad_batches 1 \
    --gradient_clip_val 1.0 \
    --gen_max_len 65 \
    --gen_min_len 60 \
    --num_beams 5 \
    --length_penalty 1.0 \
    --train_max_trg_len 80 \
    --train_max_src_len 1024 \
    --weight_decay 0 \
    --no_repeat_ngram_size 3 \
    --early_stopping \
    --enable_progress_bar true \
    --src context \
    --trg target \
    --eval_metrics ppl 