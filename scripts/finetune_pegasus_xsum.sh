cd $wkdir
dataset=xsum
pretrained_model=pegasus_large 
output_dir=../results/finetune_pegasus_large

# CUDA_VISIBLE_DEVICES=0,1,2,3
python train_summarizer.py \
    --train_path ../data/$dataset/train.jsonl \
    --dev_path ../data/$dataset/dev.jsonl \
    --test_path ../data/$dataset/test.jsonl \
    --pretrained_model_path  ../pretrained_model/$pretrained_model \
    --learning_rate 1e-4 \
    --label_smoothing 0.1 \
    --max_train_steps 130000 \
    --warmup_steps 10000 \
    --output_dir $output_dir \
    --do_train \
    --do_predict \
    --train_batch_size 8 \
    --eval_batch_size 20 \
    --gradient_accumulation_steps 8 \ ## 4 gpu cards
    --gen_max_len 62 \
    --gen_min_len 11 \
    --num_beams 8 \
    --length_penalty 0.6 \
    --train_max_trg_len 80 \
    --train_max_src_len 512 \
    --weight_decay 0 \
    --no_repeat_ngram_size 3 \
    --logging_steps 100 \
    --early_stopping \

# CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir=../results/finetune_pegasus_large_with_memory
python train_summarizer.py \
    --train_path $dataset_path/$dataset/train.jsonl \
    --dev_path $dataset_path/$dataset/dev.jsonl \
    --test_path $dataset_path/$dataset/test.jsonl \
    --train_memory_path $dataset_path/$dataset/memory/bm25/train.txt \
    --dev_memory_path $dataset_path/$dataset/memory/bm25/dev.txt \
    --test_memory_path $dataset_path/$dataset/memory/bm25/test.txt \
    --pretrained_model_path  $pretrained_model_path \
    --learning_rate 1e-4 \
    --label_smoothing 0.1 \
    --max_train_steps 130000 \
    --warmup_steps 10000 \
    --output_dir $output_dir \
    --do_train \
    --do_predict \
    --train_batch_size 8 \
    --eval_batch_size 20 \
    --gradient_accumulation_steps 8 \ ## 4 gpu cards
    --gen_max_len 62 \
    --gen_min_len 11 \
    --num_beams 8 \
    --length_penalty 0.6 \
    --train_max_trg_len 80 \
    --train_max_src_len 512 \
    --weight_decay 0 \
    --no_repeat_ngram_size 3 \
    --logging_steps 100 \
    --early_stopping \

pretrained_model_path=../pretrained_model/pegasus_xsum 
# CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir=../results/finetune_pegasus_xsum_with_memory
python train_summarizer.py \
    --train_path $dataset_path/$dataset/train.jsonl \
    --dev_path $dataset_path/$dataset/dev.jsonl \
    --test_path $dataset_path/$dataset/test.jsonl \
    --train_memory_path $dataset_path/$dataset/memory/bm25/train.txt \
    --dev_memory_path $dataset_path/$dataset/memory/bm25/dev.txt \
    --test_memory_path $dataset_path/$dataset/memory/bm25/test.txt \
    --pretrained_model_path  $pretrained_model_path \
    --learning_rate 5e-5 \
    --label_smoothing 0.1 \
    --max_train_steps 130000 \
    --warmup_steps 10000 \
    --output_dir $output_dir \
    --do_train \
    --do_predict \
    --train_batch_size 8 \
    --eval_batch_size 20 \
    --gradient_accumulation_steps 8 \ ## 4 gpu cards
    --gen_max_len 62 \
    --gen_min_len 11 \
    --num_beams 8 \
    --length_penalty 0.6 \
    --train_max_trg_len 80 \
    --train_max_src_len 512 \
    --weight_decay 0 \
    --no_repeat_ngram_size 3 \
    --logging_steps 100 \
    --early_stopping \
    --zero_shot

