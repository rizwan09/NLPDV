#!/usr/bin/env bash
export GLUE_DIR='/home/rizwan/NLPDV/glue/'
task_name=MNLI
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 7 ./examples/run_sglue.py   \
CUDA_VISIBLE_DEVICES=1 python ./examples/run_sglue.py   \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name $task_name \
    --do_train   \
    --do_eval   \
    --do_lower_case   \
    --data_dir $GLUE_DIR/MNLI/   \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir /tmp/$task_name_output/ \
    --overwrite_output_dir   \
    --fp16 \
    --data_size 100


#    --overwrite_cache
#    &>> ./log/run_log.txt