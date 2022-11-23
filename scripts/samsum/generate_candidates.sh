for _split in dev test train
do
    python generate_hyps.py \
        --default_root_dir /tmp \
        --data_path ../data/samsum/${_split}.jsonl \
        --config_path ./config/samsum/generate_hyps_dbs.yaml \
        --pretrained_model_path $1 \
        --output_path ../candidates/samsum/$2/diverse/${_split}.candidates \
        --memory_path ../data/samsum/memory/bm25/${_split}.txt \
        --memory_encoding $2 \
        --per_device_eval_batch_size 2
done
