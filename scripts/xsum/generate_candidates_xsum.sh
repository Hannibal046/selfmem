cd $wkdir
dataset=xsum
pretrained_model_path=../pretrained_model/pegasus_xsum
# for _split in train dev test
for _split in test
do  
    diversity_penalty=0.1
    num_beams=128
    num_beam_groups=16
    for diversity_penalty in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python generate_hyps.py \
            --default_root_dir /tmp \
            --data_path ../data/${dataset}/${_split}.jsonl \
            --output_path ../candidates/${dataset}/pegasus_xsum/dp_${diversity_penalty}/candidates_${_split}.txt \
            --pretrained_model_path $pretrained_model_path \
            --num_return_sequences 128 \
            --num_beam_groups $num_beam_groups \
            --diversity_penalty $diversity_penalty \
            --num_beams $num_beams \
            --length_penalty 0.6 \
            --train_max_src_len 512 \
            --gen_max_len 64 \
            --per_device_eval_batch_size 1 \
            --accelerator gpu \

    done
done
