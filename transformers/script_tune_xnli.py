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

#gpu 0,2 on NLP9 are culprit gpu 3,4 on nlp8
CUDA_VISIBLE_DEVICES = [5,6,7]

BASE_DATA_DIR = '/home/rizwan/NLPDV/XNLI'
run_file = './examples/run_sxnli.py'
model_type = 'bert'
train_model_name_or_path = 'bert-base-multilingual-cased'  # 'bert-large-uncased-whole-word-masking'
do_lower_case = False
num_train_epochs = 1.0
num_eval_epochs = 2.0
per_gpu_eval_batch_size = 32
per_gpu_train_batch_size = 32
learning_rate = 5e-5
max_seq_length = 128
fp16 = True
overwrite_cache = False


#batch sizes: 8, 16, 32, 64, 128 (for max seq 128, max batch size is 32)
#learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5
'''
Runs:
'''
best_learning_rate = 5e-5
best_per_gpu_train_batch_size = 32
best_acc = 0
all_acc_shapley = []
all_acc_baseline = []
is_tune=True

if not is_tune: num_train_epochs=3.0

ALL_EVAL_TASKS = 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split()

for eval_task_name in ALL_EVAL_TASKS:
    train_task_name = eval_task_name
    for is_Shapley in [False]:
        for learning_rate in [ 5e-5]:
            for per_gpu_train_batch_size in [32]:


                if is_Shapley=='LOO': train_output_dir = 'temp/' + train_task_name + '_output_LOO_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  # +str(seed)+'/'
                elif is_Shapley==True: train_output_dir = 'temp/' + train_task_name + '_output_Shapley_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  # +str(seed)+'/'
                elif is_Shapley == "BEST_SINGLE":
                    train_output_dir = 'temp/' + train_task_name + '_output_BEST_SINGLE_' + str(
                        per_gpu_train_batch_size) + '_' + str(learning_rate)  # +str(seed)+'/'

                else:
                    train_output_dir = 'temp/' + train_task_name + '_output_baseline_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  #

                eval_output_dir = train_output_dir

                train_data_dir = BASE_DATA_DIR
                eval_data_dir = BASE_DATA_DIR
                seed = 43

                directory = eval_output_dir

                if not os.path.exists(directory) :
                    os.makedirs(directory)
                    os.makedirs(os.path.join(directory, 'plots'))


                def write_indices_to_delete(indices_to_delete_file_path, ids):
                    with open(indices_to_delete_file_path, "w") as writer:
                        print(f"***** Writing ids to {str(indices_to_delete_file_path)}  *****", flush=True)
                        for id in ids:
                            writer.write("%s " % (id))

                indices_to_delete_file_path = directory + '/indices_to_delete_file_path_' + str(seed) + '.json'

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                ALL_BINARY_TASKS = 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split()


                DOMAIN_TRANSFER = True

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                if eval_task_name in ALL_BINARY_TASKS: ALL_BINARY_TASKS.remove(eval_task_name)







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
                train_run_command_full = run_command + ' --do_train --data_dir ' + train_data_dir + ' --output_dir ' + \
                                         train_output_dir + ' --model_name_or_path ' + train_model_name_or_path + ' --language ' + eval_task_name

                train_run_command = train_run_command_full #train_run_command_full + ' --data_size ' + str(train_data_size)

                # For eval:
                eval_run_command_full = run_command + ' --do_eval --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                        ' --model_name_or_path ' + train_output_dir

                eval_run_command = eval_run_command_full

                if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path

                command = train_run_command + ' --num_train_epochs ' + str(num_train_epochs)
                print(command, flush=True)



                os.system(command)

                # initial Eval on whole dataset
                command = eval_run_command
                print(command, flush=True)
                os.system(command)

                output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                with open(output_eval_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key in ['acc']:
                            acc = float(value)

                print('-'*100, flush=True)
                print("Task: ", train_task_name, flush=True)
                print("learning_rate: ", learning_rate, flush=True)
                print("per_gpu_train_batch_size: ", per_gpu_train_batch_size, flush=True)
                print("Acc: ", acc, flush=True)
                print("Shapely: ", str(is_Shapley), flush=True)
                print('-'*100, flush=True)
                if is_Shapley: all_acc_shapley.append(acc)
                else: all_acc_baseline.append(acc)
                if acc>best_acc:
                    best_per_gpu_train_batch_size = per_gpu_train_batch_size
                    best_learning_rate = learning_rate
                    best_acc=acc

        print('-'*100, flush=True)
        print('Task: ',eval_task_name, flush=True)
        print('best lr: ', best_learning_rate, 'bz sz: ', best_per_gpu_train_batch_size, \
              'best acc: ', best_acc, 'all_acc_shapley: ', all_acc_shapley, ' all_acc_baseline: ', all_acc_baseline,  flush=True)
        print('-'*100, flush=True)
#
