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
CUDA_VISIBLE_DEVICES = [0,1,2,3, 4,5,6,7]

BASE_DATA_DIR = '/local/rizwan/UDTree/'
run_file = './examples/run_multi_domain_pos.py'
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

ALL_EVAL_TASKS = [
        'UD_ARABIC',
        'UD_BASQUE',
        'UD_BULGARIAN',
        'UD_CATALAN',
        'UD_CHINESE',
        'UD_CROATIAN',
        'UD_CZECH',
        'UD_DANISH',
        'UD_DUTCH',
        'UD_ENGLISH',
        'UD_FINNISH',
        'UD_FRENCH',
        'UD_GERMAN',
        'UD_HEBREW',
        'UD_HINDI',
        'UD_INDONESIAN',
        'UD_ITALIAN',
        'UD_JAPANESE',
        'UD_KOREAN',
        'UD_NORWEGIAN',
        'UD_PERSIAN',
        'UD_POLISH',
        'UD_PORTUGUESE',
        'UD_ROMANIAN',
        'UD_RUSSIAN',
        'UD_SERBIAN',
        'UD_SLOVAK',
        'UD_SLOVENIAN',
        'UD_SPANISH',
        'UD_SWEDISH',
        'UD_TURKISH']

for eval_task_name in ALL_EVAL_TASKS[25:28][::-1]:
    train_task_name = eval_task_name
    for is_Shapley in [ True]:
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
                ALL_BINARY_TASKS =  [
                    'UD_ARABIC',
                    'UD_BASQUE',
                    'UD_BULGARIAN',
                    'UD_CATALAN',
                    'UD_CHINESE',
                    'UD_CROATIAN',
                    'UD_CZECH',
                    'UD_DANISH',
                    'UD_DUTCH',
                    'UD_ENGLISH',
                    'UD_FINNISH',
                    'UD_FRENCH',
                    'UD_GERMAN',
                    'UD_HEBREW',
                    'UD_HINDI',
                    'UD_INDONESIAN',
                    'UD_ITALIAN',
                    'UD_JAPANESE',
                    'UD_KOREAN',
                    'UD_NORWEGIAN',
                    'UD_PERSIAN',
                    'UD_POLISH',
                    'UD_PORTUGUESE',
                    'UD_ROMANIAN',
                    'UD_RUSSIAN',
                    'UD_SERBIAN',
                    'UD_SLOVAK',
                    'UD_SLOVENIAN',
                    'UD_SPANISH',
                    'UD_SWEDISH',
                    'UD_TURKISH']


                DOMAIN_TRANSFER = True

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                if eval_task_name in ALL_BINARY_TASKS: ALL_BINARY_TASKS.remove(eval_task_name)



                if is_Shapley == True and eval_task_name=='UD_ARABIC':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 3, 7, 8, 11, 16, 13, 21, 29]) # 11, 16, 29 also
                if is_Shapley == True and eval_task_name=='UD_BASQUE':
                    write_indices_to_delete(indices_to_delete_file_path, [17, 19]) # 17, 19 also
                if is_Shapley == True and eval_task_name == 'UD_BULGARIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [3, 17, 19]) # 13, 17 also
                if is_Shapley == True and eval_task_name == 'UD_CATALAN':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 3, 17, 20]) # 0, 19, 20 also
                if is_Shapley == True and eval_task_name == 'UD_CHINESE':
                    write_indices_to_delete(indices_to_delete_file_path, [5, 13, 20, 25, 26]) # 19 also
                if is_Shapley == True and eval_task_name == 'UD_CROATIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [13, 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_CZECH':
                    write_indices_to_delete(indices_to_delete_file_path, [13,  17, 19])
                if is_Shapley == True and eval_task_name == 'UD_DANISH':
                    write_indices_to_delete(indices_to_delete_file_path, [ 13]) # 17, 19 also
                if is_Shapley == True and eval_task_name == 'UD_DUTCH':
                    write_indices_to_delete(indices_to_delete_file_path, [ 17, 13]) # 13, 19 also To check: 13, None of 13, 19
                if is_Shapley == True and eval_task_name == 'UD_ENGLISH':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 3, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 29]) # shapley>0.01
                if is_Shapley == 'BEST_SINGLE' and eval_task_name == 'UD_ENGLISH':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) # shapley>0.01
                if is_Shapley == True and eval_task_name == 'UD_FINNISH':
                    write_indices_to_delete(indices_to_delete_file_path, [ 17, 19]) # 13, 17, 19 also To check: 17, 19, 13, etc.,
                if is_Shapley == True and eval_task_name == 'UD_FRENCH':
                    write_indices_to_delete(indices_to_delete_file_path, [17, 19] ) #  17, 19 also To check: 19
                if is_Shapley == True and eval_task_name == 'UD_GERMAN':
                    write_indices_to_delete(indices_to_delete_file_path, [3, 5, 16, 17, 19, 20]) #  17, 19 also To check: 19
                if is_Shapley == True and eval_task_name == 'UD_HEBREW':
                    write_indices_to_delete(indices_to_delete_file_path, [ 17]) #  17, 19 also To check: 19 (not important)
                if is_Shapley == True and eval_task_name == 'UD_HINDI':
                    write_indices_to_delete(indices_to_delete_file_path, [ 0, 17, 19]) #  0 also To check: No 0
                if is_Shapley == True and eval_task_name == 'UD_INDONESIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [ 0, 13, 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_ITALIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [ 5, 17, 19, 20])
                if is_Shapley == True and eval_task_name == 'UD_JAPANESE':
                    write_indices_to_delete(indices_to_delete_file_path, [ 19])
                if is_Shapley == True and eval_task_name == 'UD_KOREAN':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 13, 19])
                if is_Shapley == True and eval_task_name == 'UD_NORWEGIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [0, 13, 19])
                if is_Shapley == True and eval_task_name == 'UD_PERSIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [4, 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_POLISH':
                    write_indices_to_delete(indices_to_delete_file_path, [13, 19])
                if is_Shapley == True and eval_task_name == 'UD_PORTUGUESE':
                    write_indices_to_delete(indices_to_delete_file_path, [17])
                if is_Shapley == True and eval_task_name == 'UD_ROMANIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [ 13, 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_RUSSIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [ 13, 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_SERBIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [ 17, 19])
                if is_Shapley == True and eval_task_name == 'UD_SLOVAK':
                    write_indices_to_delete(indices_to_delete_file_path, [13, 19])
                if is_Shapley == True and eval_task_name == 'UD_SLOVENIAN':
                    write_indices_to_delete(indices_to_delete_file_path, [  17, ])
                if is_Shapley == True and eval_task_name == 'UD_SPANISH':
                    write_indices_to_delete(indices_to_delete_file_path, [ 5, 17, 19, 21])
                if is_Shapley == True and eval_task_name == 'UD_SWEDISH':
                    write_indices_to_delete(indices_to_delete_file_path, [  17])
                if is_Shapley == True and eval_task_name == 'UD_TURKISH':
                    write_indices_to_delete(indices_to_delete_file_path, [ 19])


                # else is_Shapley: continue



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
                if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path


                # For eval:
                eval_run_command = run_command + ' --do_eval --do_predict --task_name ' + eval_task_name + \
                                   ' --data_dir ' + eval_data_dir + ' --output_dir ' + train_output_dir + \
                                   ' --model_name_or_path ' + train_output_dir + ' --seed '+str(seed)

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
