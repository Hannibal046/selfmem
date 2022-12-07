dataset=$1
memory_encoding=$2
pretrained_model_path=$3
output_dir=$4
metrics=$5

for _split in train dev test
do
    python generate_hyps.py \
        --config_path config/$dataset/generate_hyps.yaml \
        --data_path ../data/$dataset/${_split}.jsonl \
        --memory_path ../data/$dataset/memory/bm25/${_split}.txt \
        --memory_encoding $memory_encoding \
        --pretrained_model_path $pretrained_model_path \
        --output_path ../candidates/$dataset/$output_dir/beam/${_split}.candidates 
    
    python calculate_candidates_score.py \
        --refs_path ../data/$dataset/${_split}.jsonl \
        --candidates_path ../candidates/$dataset/$output_dir/beam/${_split}.candidates \
        --metrics $metrics
done