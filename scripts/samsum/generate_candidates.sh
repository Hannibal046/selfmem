# cd $wkdir


# ## base bart generate on samsum test.jsonl
# python generate_hyps.py \
#     --default_root_dir /tmp \
#     --config_path ./config/samsum/generate_hyps.yaml \
#     --pretrained_model_path ? \
#     --output_path ? \
#     --

# ## retrieval-aug bart 
# python generate_hyps.py \
#     --default_root_dir /tmp \
#     --config_path ./config/samsum/generate_hyps.yaml \
#     --pretrained_model_path ? \
#     --output_path ? \
#     --memory_path ../data/samsum/memory/bm25/test.txt \
#     --memory_encoding separate \

# ## generate candidates
# python generate_hyps.py \
#     --default_root_dir /tmp \
#     --config_path ./config/samsum/generate_hyps_dbs.yaml \
#     --pretrained_model_path ? \
#     --output_path ? \
#     --memory_path ../data/${dataset}/memory/bm25/${_split}.txt \
#     --memory_encoding separate \

for _split in dev test train
do
    CUDA_VISIBLE_DEVICES=0 python generate_hyps.py \
        --default_root_dir /tmp \
        --data_path ../data/samsum/${_split}.jsonl \
        --config_path ./config/samsum/generate_hyps_dbs.yaml \
        --pretrained_model_path /home/hannibal046/project/selfmem/results/generator/samsum/concate_bart_large/lightning_logs/version_0/bart_large_best_ckpt \
        --output_path ../candidates/samsum/diverse/${_split}.candidates \
        --memory_path ../data/samsum/memory/bm25/${_split}.txt \
        --memory_encoding concate \

done
