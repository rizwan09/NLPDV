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
CUDA_VISIBLE_DEVICES = [0,1,2,3,4,5,6,7]

BASE_DATA_DIR = '/home/rizwan/NLPDV/glue/'
run_file = './examples/run_sglue.py'
model_type = 'bert'
train_model_name_or_path = 'bert-base-cased' # 'bert-large-uncased-whole-word-masking'
do_lower_case = False
num_train_epochs = 4.0
num_eval_epochs = 1.0
per_gpu_eval_batch_size = 100
per_gpu_train_batch_size = 32
learning_rate = 5e-5
max_seq_length = 128
fp16 = True
overwrite_cache = False
evaluate_during_training = True

#batch sizes: 8, 16, 32, 64, 128 (for max seq 128, max batch size is 32)
#learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5
'''
Runs:
'''


ALL_EVAL_TASKS = ['MNLI-mm', "QNLI"]


shpley_removals = {
    'MNLI-mm': [2],
    'QNLI': [0,2,3,4,5,6],
    'SNLI': [0,1],
    'QQP': [1]
}

all_acc_shapley = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}
all_acc_baseline = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}
all_acc_baseline_s = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}
is_tune=True

BASELINES_S = 'baseline-s'

if not is_tune: num_train_epochs=4.0



for eval_task_name in list(shpley_removals.keys()):
    eval_data_size = 1200 if eval_task_name=="QNLI" else 2000

    for i in range(1):
        seed = 43
        np.random.seed(seed)
        for is_few_shot in [False]:
            for is_Shapley in [  True, False]:
                best_learning_rate = None
                best_per_gpu_train_batch_size = None
                best_acc = 0

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                ALL_BINARY_TASKS = ['snli', 'qqp', 'qnli', 'mnli-fiction', 'mnli-travel', 'mnli-slate',
                                    'mnli-government',
                                    'mnli-telephone']

                DOMAIN_TRANSFER = True

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                if eval_task_name in ALL_BINARY_TASKS: ALL_BINARY_TASKS.remove(eval_task_name)

                if is_Shapley==BASELINES_S:
                    raddom_domains = np.random.choice(np.arange(len(ALL_BINARY_TASKS)), \
                                                  len(shpley_removals[eval_task_name]), replace=False)

                if model_type=='bert': learning_rates = [ 2e-5 ]
                elif  model_type=='xlm': learning_rates = [ 5e-4, 2e-4 ]
                bz_szs = [  32]

                for learning_rate in learning_rates:
                    for per_gpu_train_batch_size in bz_szs:

                        # if eval_task_name == 'UD_BULGARIAN' and is_Shapley==True: continue


                        train_task_name = eval_task_name
                        if is_Shapley=='LOO': train_output_dir = 'temp/' + train_task_name + '_output_LOO_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate) #+str(seed)+'/'
                        elif is_Shapley==True:

                            train_output_dir = 'temp/' + train_task_name + '_output_Shapley_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate) #+str(seed)+'/'
                        elif is_Shapley == BASELINES_S:
                            train_output_dir = 'temp/' + train_task_name + '_output_baseline-s_' + str(
                                per_gpu_train_batch_size) + '_' + str(learning_rate)  #+str(seed)+'/'
                        else:
                            train_output_dir = 'temp/' + train_task_name + '_output_baseline_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  #+str(seed)+'/'

                        eval_output_dir = train_output_dir +'/best'

                        train_data_dir = BASE_DATA_DIR
                        eval_data_dir = BASE_DATA_DIR


                        directory = eval_output_dir

                        if not os.path.exists(train_output_dir) :
                            os.makedirs(directory)
                            os.makedirs(os.path.join(directory, 'plots'))

                        if not os.path.exists(directory) :
                            os.makedirs(directory)
                            os.makedirs(os.path.join(directory, 'plots'))



                        def write_indices_to_delete(indices_to_delete_file_path, ids):
                            with open(indices_to_delete_file_path, "w") as writer:
                                print(f"***** Writing ids to {str(indices_to_delete_file_path)}  *****", flush=True)
                                for id in ids:
                                    writer.write("%s " % (id))

                        indices_to_delete_file_path = directory + '/indices_to_delete_file_path' + '.json'



                        if is_Shapley == True and eval_task_name != 'UD_TURKISH':
                            write_indices_to_delete(indices_to_delete_file_path, shpley_removals[eval_task_name])

                        if is_Shapley == BASELINES_S and eval_task_name != 'UD_TURKISH':

                            print('-eval_task_name: ', eval_task_name, flush=True)
                            print('raddom_removal_domains: ', raddom_domains,\
                                  'shapley removals: ', shpley_removals[eval_task_name], flush=True)
                            write_indices_to_delete(indices_to_delete_file_path, raddom_domains )



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
                        if evaluate_during_training: run_command += ' --evaluate_during_training '


                        # For training:
                        train_run_command = run_command + ' --do_train --task_name ' + train_task_name + \
                                            ' --data_dir ' + train_data_dir + ' --output_dir ' + \
                                            train_output_dir + ' --model_name_or_path ' + train_model_name_or_path


                        if is_few_shot : train_run_command += ' --is_few_shot'
                        if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path

                        # For eval:
                        run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])

                        run_command += ' python '


                        run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
                                       ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
                            per_gpu_eval_batch_size) + \
                                       ' --per_gpu_train_batch_size=' + str(
                            per_gpu_train_batch_size) + ' --learning_rate ' + str(learning_rate) \
                                       + ' --overwrite_output_dir '
                        if do_lower_case: run_command += '--do_lower_case '
                        if fp16: run_command += ' --fp16 '

                        if overwrite_cache:
                            run_command += ' --overwrite_cache '
                        if evaluate_during_training: run_command += ' --evaluate_during_training '

                        eval_run_command = run_command + ' --do_eval  --task_name ' + eval_task_name + \
                                           ' --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                           ' --model_name_or_path ' + eval_output_dir

                        command = train_run_command + ' --num_train_epochs 1'
                        print(command, flush=True)

                        # if not os.path.exists(os.path.join(directory, 'pytorch_model.bin')):\
                        #     os.system(command)

                        # initial Eval on whole dataset
                        command = eval_run_command
                        if eval_data_size: command += ' --data_size ' + str(eval_data_size)
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
                        if is_Shapley==True:
                            all_acc_shapley[eval_task_name].append(acc)
                        elif is_Shapley==False:
                            all_acc_baseline[eval_task_name].append(acc)
                        else:
                            all_acc_baseline_s[eval_task_name].append(acc)
                        if acc>best_acc:
                            best_per_gpu_train_batch_size = per_gpu_train_batch_size
                            best_learning_rate = learning_rate
                            best_acc=acc

                print('-' * 100, flush=True)
                print('-Task: ', eval_task_name, flush=True)
                print('-is_Shapley: ', is_Shapley, flush=True)
                print('-best lr: ', best_learning_rate, '\n-bz sz: ', best_per_gpu_train_batch_size, \
                      '\n-best acc: ', best_acc, '\n-all_acc_shapley: ', all_acc_shapley, \
                      '\n-all_acc_shapley_baseline_s: ', all_acc_baseline_s, '\n- all_acc_baseline: ', all_acc_baseline,
                      flush=True)
                print('-' * 100, flush=True)

                # For Test:

                train_task_name = eval_task_name
                if is_Shapley == 'LOO':
                    train_output_dir = 'temp/' + train_task_name + '_output_LOO_' + str(
                        best_per_gpu_train_batch_size) + '_' + str(
                        best_learning_rate)  # +str(seed)+'/'
                elif is_Shapley == True:
                    train_output_dir = 'temp/' + train_task_name + '_output_Shapley_' + str(
                        best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  # +str(seed)+'/'
                elif is_Shapley == BASELINES_S:
                    train_output_dir = 'temp/' + train_task_name + '_output_baseline-s_' + str(
                        best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  # +str(seed)+'/'
                else:
                    train_output_dir = 'temp/' + train_task_name + '_output_baseline_' + str(
                        best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  # +str(seed)+'/'

                eval_output_dir = train_output_dir + '/best/'

                run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])
                for i in CUDA_VISIBLE_DEVICES[1:]:
                    run_command += ',' + str(i)
                run_command += ' python '

                if len(CUDA_VISIBLE_DEVICES) > 1: run_command += '-m torch.distributed.launch --nproc_per_node ' \
                                                                 + str(len(CUDA_VISIBLE_DEVICES))

                run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
                               ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
                    per_gpu_eval_batch_size) + \
                               ' --per_gpu_train_batch_size=' + str(
                    best_per_gpu_train_batch_size) + ' --learning_rate ' + str(
                    best_learning_rate) \
                               + ' --overwrite_output_dir '
                if do_lower_case: run_command += '--do_lower_case '
                if fp16: run_command += ' --fp16 '

                if overwrite_cache:
                    run_command += ' --overwrite_cache '

                if evaluate_during_training: run_command += ' --evaluate_during_training '

                train_run_command = run_command + ' --do_train --task_name ' + train_task_name + \
                                    ' --data_dir ' + train_data_dir + ' --output_dir ' + \
                                    train_output_dir + ' --model_name_or_path ' + train_output_dir

                # For eval:

                run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])
                run_command += ' python '

                run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
                               ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
                    per_gpu_eval_batch_size) + \
                               ' --per_gpu_train_batch_size=' + str(
                    best_per_gpu_train_batch_size) + ' --learning_rate ' + str(
                    best_learning_rate) \
                               + ' --overwrite_output_dir '
                if do_lower_case: run_command += '--do_lower_case '
                if fp16: run_command += ' --fp16 '

                if overwrite_cache:
                    run_command += ' --overwrite_cache '
                eval_run_command = run_command + ' --do_eval  --task_name ' + eval_task_name + \
                                   ' --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                   ' --model_name_or_path ' + eval_output_dir

                indices_to_delete_file_path = eval_output_dir + '/indices_to_delete_file_path' + '.json'
                if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path

                command = train_run_command + ' --num_train_epochs ' + str(num_train_epochs-1)
                print(command, flush=True)

                # os.system(command)

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

                print('-' * 100, flush=True)
                print("Task: ", train_task_name, flush=True)
                print("best_learning_rate: ", best_learning_rate, flush=True)
                print("best_per_gpu_train_batch_size: ", best_per_gpu_train_batch_size, flush=True)
                print("BEST TEST Acc: ", acc, flush=True)
                print("Shapely: ", str(is_Shapley), flush=True)
                print('-' * 100, flush=True)

                if is_Shapley == True:
                    best_shapley_learning_rate = best_learning_rate
                    best_shapley_per_gpu_train_batch_size = best_per_gpu_train_batch_size
                elif is_Shapley == BASELINES_S:
                    best_baseline_s_learning_rate = best_learning_rate
                    best_baseline_s_per_gpu_train_batch_size = best_per_gpu_train_batch_size
                else:
                    best_baseline_learning_rate = best_learning_rate
                    best_baseline_per_gpu_train_batch_size = best_per_gpu_train_batch_size