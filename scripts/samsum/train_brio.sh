## base fine tune
python train_brio.py \
    --config_path ./config/samsum/train_brio.yaml \
    --candidate_dir ?

## base retrieval augmented
python train_brio.py \
    --config_path ./config/samsum/train_brio.yaml \
    --candidate_dir ../candidates/samsum/dual_encoder_bart_with_memory/dp_0.1 \
    --pretrained_model_path  ../results/finetune_samsum_memory_separate/lightning_logs/version_2/bart_large_best_ckpt \
    --memory_dir ../data/$dataset/memory/bm25/ \
    --memory_encoding separate \
    --default_root_dir ../results/brio_samsum_bart_dualencoder_test