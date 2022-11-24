refs_dir=$1
candidates_dir=$2

for _split in train dev test
do
    python calculate_brio_score.py \
        --refs_path $refs_dir/${_split}.jsonl \
        --candidates_path $candidates_dir/${_split}.candidates 
        --num_workers 20
done

