import os, pdb

# ______________________________________NLPDV____________________________________
# _______________________________________________________________________
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import *
import _pickle as pkl
import shutil
import numpy as np
from tqdm import trange, tqdm

# _______________________________________________________________________
# ______________________________________NLPDV____________________________________
# Task param
train_task_name = 'SNLI'
eval_task_name = 'SNLI'
# CUDA gpus
CUDA_VISIBLE_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]




# Model params
GLUE_DIR = '/home/rizwan/NLPDV/glue/'
run_file = './examples/run_sglue.py'
# run_file= './examples/data_valuation.py'
model_type = 'bert'
train_model_name_or_path = 'bert-base-cased'
do_lower_case = True
num_train_epochs = 1.0
num_eval_epochs = 2.0
per_gpu_eval_batch_size = 8
per_gpu_train_batch_size = 8
learning_rate = 2e-5
max_seq_length = 128
fp16 = True
overwrite_cache = False



train_output_dir = 'temp/' + train_task_name + '_output/'  # +str(seed)+'/'
eval_output_dir = 'temp/' + eval_task_name + '_output/'  # +str(seed)+'/'


train_data_dir = GLUE_DIR
eval_data_dir = GLUE_DIR

# _______________________________________________________________________
# ______________________________________NLPDV____________________________________


run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])
for i in CUDA_VISIBLE_DEVICES[1:]:
    run_command += ',' + str(i)
run_command += ' python '

if len(CUDA_VISIBLE_DEVICES) > 1: run_command += '-m torch.distributed.launch --nproc_per_node ' \
                                                 + str(len(CUDA_VISIBLE_DEVICES))
run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
               ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
    per_gpu_eval_batch_size) + \
               ' --per_gpu_train_batch_size=' + str(per_gpu_train_batch_size) + ' --learning_rate ' + str(learning_rate) \
               + ' --overwrite_output_dir '
if do_lower_case: run_command += '--do_lower_case '
if fp16: run_command += ' --fp16 '

if overwrite_cache:
    run_command += ' --overwrite_cache '

# For training:
train_run_command = run_command + ' --do_train --task_name ' + train_task_name + \
                    ' --data_dir ' + train_data_dir + ' --output_dir ' + \
                    train_output_dir + ' --model_name_or_path ' + train_model_name_or_path


# For eval:
eval_run_command = run_command + ' --do_eval --task_name ' + eval_task_name + \
                   ' --data_dir ' + eval_data_dir + ' --output_dir ' + train_output_dir + \
                   ' --model_name_or_path ' + train_output_dir

command = train_run_command + ' --num_train_epochs ' + str(num_train_epochs) + ' --is_baseline_run '
print(command, flush=True)
os.system(command)

# initial Eval on whole dataset
command = eval_run_command + ' --is_baseline_run  '
print(command, flush=True)
os.system(command)
