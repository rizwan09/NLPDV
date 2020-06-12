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
CUDA_VISIBLE_DEVICES = [1,3,4,5,6,7]
train_data_size = 100
BASE_DATA_DIR = '/home/rizwan/NLPDV/XNLI'
run_file = './examples/run_sxnli.py'
model_type = 'bert'
# model_type = 'xlm'
train_model_name_or_path = 'bert-base-multilingual-cased'  # 'bert-large-uncased-whole-word-masking'
# train_model_name_or_path = 'xlm-mlm-xnli15-1024'  # 'bert-large-uncased-whole-word-masking'
do_lower_case = False
num_train_epochs = 2.0
num_eval_epochs = 1.0
per_gpu_eval_batch_size = 200
per_gpu_train_batch_size = 32
learning_rate = 5e-5
max_seq_length = 128
fp16 = True
overwrite_cache = False

evaluate_during_training = False

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

# if not is_tune: num_train_epochs=3.0

ALL_EVAL_TASKS = 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split()

# import os
# for eval_task_name in 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split():cmd = 'cp -r temp/'+eval_task_name+'_output_baseline_32_5e-05/* temp/'+eval_task_name+'_output_baseline_32_5e-05/best/';os.system(cmd)

BASELINE_S = "baseline-s"


shpley_removals = {
    # 'fr': [8,10,11,12]

    #
    # 'ar':[12],
    'th':[0,6],



    # 'zh': [0,5,8,12], #good
    # 'hi':[12], # good

    # 'sw': [10],
    # 'ur':[3,11],
    # 'en': [2, 5,8.10,11,12],
    #'

}


do_train = True


def get_global_step(train_output_dir):
    global_step = None
    n_points_file = os.path.join(train_output_dir, "training_results" + ".txt")
    try:
        with open(n_points_file, "r") as reader:
            for line in reader:
                line = line.strip().split()
                key = line[0]
                value = line[-1]
                if key == 'n_points':
                    n_points2 = int(value)
                if key == 'global_step':
                    global_step = int(value)
    except:
        pass
    if not global_step:
        if n_points2:
            global_step = int(n_points2 / (len(CUDA_VISIBLE_DEVICES) * per_gpu_train_batch_size))
        else:
            global_step = 0
    return global_step


# BASLINE_DEV = {"tr": 0.6386, "ar": 0.682, "th": 0.562650602409638, "zh": 0.7314}

all_acc_shapley = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}
all_acc_baseline = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}
all_acc_baseline_s = {eval_task_name:[] for eval_task_name in list(shpley_removals.keys())}

for eval_task_name in list(shpley_removals.keys())[:1]:
    print("--eval task name: " , eval_task_name)
    for is_Shapley in [True]:
        best_learning_rate = None
        best_per_gpu_train_batch_size = None
        best_acc = 0

        if model_type == 'bert':
            learning_rates = [ 3e-5]
        elif model_type == 'xlm':
            learning_rates = [5e-4, 2e-4]

        for learning_rate in learning_rates:
            for per_gpu_train_batch_size in [32]:
                print('--eval_task_name: ', eval_task_name, flush=True)
                best_acc_for_this_setting = 0
                train_task_name = eval_task_name
                if is_Shapley=='LOO': train_output_dir = 'temp/' + train_task_name + '_output_LOO_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  # +str(seed)+'/'
                elif is_Shapley==True: train_output_dir = 'temp/' + train_task_name + '_output_Shapley_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  # +str(seed)+'/'
                elif is_Shapley == BASELINE_S:
                    train_output_dir = 'temp/' + train_task_name + '_output_baseline-s_' + str(
                        per_gpu_train_batch_size) + '_' + str(learning_rate)  # +str(seed)+'/'
                else:
                    train_output_dir = 'temp/' + train_task_name + '_output_baseline_'+str(per_gpu_train_batch_size) + '_'+str(learning_rate)  #

                eval_output_dir = train_output_dir
                best_eval_output_dir = eval_output_dir+'/best'

                train_data_dir = BASE_DATA_DIR
                eval_data_dir = BASE_DATA_DIR
                seed = 43

                directory = eval_output_dir

                if not os.path.exists(best_eval_output_dir) :
                    print("creating directory", flush=True)
                    os.makedirs(best_eval_output_dir)
                    os.makedirs(os.path.join(best_eval_output_dir, 'plots'))


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

                if is_Shapley == BASELINE_S:
                    raddom_domains = np.random.choice(np.arange(len(ALL_BINARY_TASKS)), len(shpley_removals[eval_task_name]), replace=False)
                    print('-eval_task_name: ', eval_task_name, flush=True)
                    print('raddom_removal_domains: ', raddom_domains,  'shapley removals: ', shpley_removals[eval_task_name], flush=True)
                    write_indices_to_delete(indices_to_delete_file_path, raddom_domains)

                if is_Shapley == True:
                    write_indices_to_delete(indices_to_delete_file_path, shpley_removals[eval_task_name])




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
                train_run_command_full = run_command + ' --do_train --data_dir ' + train_data_dir + ' --output_dir ' + \
                                         train_output_dir \
                                         + ' --language ' + eval_task_name

                train_run_command = train_run_command_full #+ ' --data_size ' + str(train_data_size)

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

                eval_run_command_full = run_command + ' --do_eval --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                        ' --model_name_or_path ' + eval_output_dir + ' --language ' + eval_task_name #+ ' --test'

                eval_run_command = eval_run_command_full

                if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path

                copy_command = "cp " + train_output_dir + "/*  " + best_eval_output_dir
                pre_eval = None


                found = False
                command = train_run_command
                if  is_Shapley or not os.path.exists(os.path.join(directory, 'pytorch_model.bin')):
                    command += ' --model_name_or_path ' + train_model_name_or_path

                else:
                    found = True
                    command += ' --model_name_or_path ' + train_output_dir



                command  += ' --num_train_epochs ' + str(1)
                print(command, flush=True)
                if do_train: os.system(command)



                # initial Eval on whole dataset
                command = eval_run_command
                print(command, flush=True)
                if not pre_eval:
                    os.system(command)

                    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                    with open(output_eval_file, "r") as reader:
                        for line in reader:
                            line = line.strip().split()
                            key = line[0]
                            value = line[-1]
                            if key in ['acc']:
                                acc = float(value)
                else:
                    acc = pre_eval


                if acc>best_acc_for_this_setting:
                    print("At epoch 1(2)/4 best_acc_for_this_setting is updating from: ", best_acc_for_this_setting,
                          flush=True)
                    best_acc_for_this_setting = acc
                    print("To: ", best_acc_for_this_setting, flush=True)
                    os.system(copy_command)

                #second epoch
                # ---------------------------------------------------------------------------------------------------------
                command = train_run_command+ ' --model_name_or_path ' + train_output_dir + ' --num_train_epochs ' + str(1)
                print(command, flush=True)
                if do_train: os.system(command)

                # initial Eval on whole dataset
                command = eval_run_command
                print(command, flush=True)
                if not pre_eval:
                    os.system(command)

                    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                    with open(output_eval_file, "r") as reader:
                        for line in reader:
                            line = line.strip().split()
                            key = line[0]
                            value = line[-1]
                            if key in ['acc']:
                                acc = float(value)
                else:
                    acc = pre_eval

                if acc>best_acc_for_this_setting:
                    print("At epoch 2(3)/4 best_acc_for_this_setting is updating from: ", best_acc_for_this_setting,
                          flush=True)
                    best_acc_for_this_setting = acc
                    print("To: ", best_acc_for_this_setting, flush=True)
                    os.system(copy_command)

                # # Third epoch
                # # ---------------------------------------------------------------------------------------------------------

                if not found:
                    # ---------------------------------------------------------------------------------------------------------
                    command = train_run_command + ' --model_name_or_path ' + train_output_dir + ' --num_train_epochs ' + str(
                        1)
                    print(command, flush=True)
                    if do_train:os.system(command)

                    # initial Eval on whole dataset
                    command = eval_run_command
                    print(command, flush=True)
                    if not pre_eval:
                        os.system(command)

                        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                        with open(output_eval_file, "r") as reader:
                            for line in reader:
                                line = line.strip().split()
                                key = line[0]
                                value = line[-1]
                                if key in ['acc']:
                                    acc = float(value)
                    else:
                        acc = pre_eval

                    if acc > best_acc_for_this_setting:
                        print("At epoch (4)/4 best_acc_for_this_setting is updating from: ", best_acc_for_this_setting,
                              flush=True)
                        best_acc_for_this_setting = acc
                        print("To: ", best_acc_for_this_setting, flush=True)
                        os.system(copy_command)





                print('-'*100, flush=True)
                print("Task: ", train_task_name, flush=True)
                print("learning_rate: ", learning_rate, flush=True)
                print("per_gpu_train_batch_size: ", per_gpu_train_batch_size, flush=True)
                print("Acc: ", acc, flush=True)
                print("Shapely: ", str(is_Shapley), flush=True)
                print('-'*100, flush=True)
                if is_Shapley == True:
                    all_acc_shapley[eval_task_name].append(acc)
                elif is_Shapley == False:
                    all_acc_baseline[eval_task_name].append(acc)
                else:
                    all_acc_baseline_s[eval_task_name].append(acc)
                if acc>best_acc:
                    best_per_gpu_train_batch_size = per_gpu_train_batch_size
                    best_learning_rate = learning_rate
                    best_acc=acc

        print('-'*100, flush=True)
        print('Task: ',eval_task_name, flush=True)
        print('best lr: ', best_learning_rate, 'bz sz: ', best_per_gpu_train_batch_size, \
              'best acc: ', best_acc, 'all_acc_shapley: ', all_acc_shapley, ' all_acc_baseline: ', all_acc_baseline,  flush=True)
        print('-'*100, flush=True)

        # For Test:

        train_task_name = eval_task_name
        if is_Shapley == 'LOO':
            train_output_dir = 'temp/' + train_task_name + '_output_LOO_' + str(best_per_gpu_train_batch_size) + '_' + str(
                best_learning_rate)  # +str(seed)+'/'
        elif is_Shapley == True:
            train_output_dir = 'temp/' + train_task_name + '_output_Shapley_' + str(
                best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  # +str(seed)+'/'
        elif is_Shapley == "baseline-s":
            train_output_dir = 'temp/' + train_task_name + '_output_baseline-s_' + str(
                best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  # +str(seed)+'/'
        else:
            train_output_dir = 'temp/' + train_task_name + '_output_baseline_' + str(
                best_per_gpu_train_batch_size) + '_' + str(best_learning_rate)  #

        eval_output_dir = train_output_dir + '/best'

        run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])
        # for i in CUDA_VISIBLE_DEVICES[1:]:
        #     run_command += ',' + str(i)
        run_command += ' python '

        # if len(CUDA_VISIBLE_DEVICES) > 1: run_command += '-m torch.distributed.launch --nproc_per_node ' \
        #                                                  + str(len(CUDA_VISIBLE_DEVICES))
        run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
                       ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
            per_gpu_eval_batch_size) + \
                       ' --per_gpu_train_batch_size=' + str(best_per_gpu_train_batch_size) + ' --learning_rate ' + str(
            best_learning_rate) \
                       + ' --overwrite_output_dir '
        if do_lower_case: run_command += '--do_lower_case '
        if fp16: run_command += ' --fp16 '

        if overwrite_cache:
            run_command += ' --overwrite_cache '

        if evaluate_during_training: run_command += ' --evaluate_during_training '


        # For eval:
        best_eval_run_command_full = run_command + ' --do_eval --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                ' --model_name_or_path ' + best_eval_output_dir + ' --language ' + eval_task_name  + ' --test'



        # initial Eval on whole dataset
        command = best_eval_run_command_full
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
