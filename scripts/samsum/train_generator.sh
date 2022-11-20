cd $wkdir

## base finetune
python train_generator.py \
    --config_path ./config/samsum/train_generator.yaml \
    --default_root_dir ?

## retrieval-aug dual encoder
python train_generator.py \
    --config_path ./config/samsum/train_generator.yaml \
    --memory_dir ../data/samsum/memory/bm25 \
    --memory_encoding separate \
    --default_root_dir ?

## retrieval-aug concate
python train_generator.py \
    --config_path ./config/samsum/train_generator.yaml \
    --memory_dir ../data/samsum/memory/bm25 \
    --memory_encoding concate \
    --default_root_dir ?