wkdir=/home/hannibal046/project/selfmem/src
cd $wkdir

CUDA_VISIBLE_DEVICES=0,1 python train_brio.py \
    --data_dir ../data/xsum \
    --pretrained_model_path ../pretrained_model/pegasus_xsum \
    --default_root_dir ../results/finetune_brio_xsum \
    --warmup_steps 10000 \
    --lr 2e-3 \
    --accumulate_grad_batches 8 \
    --per_device_train_batch_size 1 \
    --margin 0.001 \
    --gradient_clip_val 0 \
    --gold_weight 0 \
    --gold_margin 0 \
    --mle_weight 0.1 \
    --rank_weight 10 \
    --scale 0.01 \
    --label_smoothing_factor 0.1 \
    --train_max_src_len 512 \
    --train_max_trg_len 80 \
    --gen_max_len 62 \
    --gen_min_len 11 \
    --adding 0 \
    --num_beams 8 \
    --warmup_steps 10000 \
    --length_penalty 0.6 \
    --accelerator gpu \
    --max_epochs 1 \
    --limit_train_batches 10 \
    --limit_val_batches 10 \
    --limit_test_batches 10 
    