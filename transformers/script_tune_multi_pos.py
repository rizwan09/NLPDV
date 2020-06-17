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
CUDA_VISIBLE_DEVICES = [0,1,3,4,5,6,7]

BASE_DATA_DIR = '/local/rizwan/UDTree/'
run_file = './examples/run_multi_domain_pos.py'
model_type = 'bert'
train_model_name_or_path = 'bert-base-multilingual-cased'  # 'bert-large-uncased-whole-word-masking'
do_lower_case = False
num_train_epochs = 4.0
num_eval_epochs = 1.0
per_gpu_eval_batch_size = 32
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


shpley_removals = {
    'UD_ARABIC': [0, 3, 7, 8, 11, 13, 16, 21, 29],
    'UD_BASQUE': [17, 19],
    'UD_BULGARIAN': [3, 19], #[3, 13, 17, 19],
    'UD_CATALAN': [ 0, 3, 17, 19, 20],
    'UD_CHINESE':[5, 13, 20, 25, 26],
    'UD_CROATIAN': [13, 17, 19],
    'UD_CZECH': [13, 17, 19],
    'UD_DANISH': [13],
    'UD_DUTCH': [17,19],
    'UD_ENGLISH': [0, 3, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 29],
    'UD_FINNISH': [13, 17, 19],
    'UD_FRENCH': [17],
    'UD_GERMAN': [ 17, 19], # Try with [3, 5, 16, 17, 19, 20]
    'UD_HEBREW': [17],
    'UD_HINDI': [ 0, 17, 19],
    'UD_INDONESIAN': [ 0, 13, 17, 19],
    'UD_ITALIAN': [ 5, 17, 19, 20],
    'UD_JAPANESE': [19],
    'UD_KOREAN': [ 0, 13, 19],
    'UD_NORWEGIAN': [0, 13, 19],
    'UD_PERSIAN': [4, 17, 19],
    'UD_POLISH': [13, 17, 19],
    'UD_PORTUGUESE': [17],
    'UD_ROMANIAN': [ 13, 17, 19],
    'UD_RUSSIAN': [ 13, 17, 19],
    'UD_SERBIAN': [ 13, 17, 19],
    'UD_SLOVAK': [ 13, 17, 19],
    'UD_SLOVENIAN': [17],
    'UD_SPANISH':[5, 17, 19],
    'UD_SWEDISH':[17],
    'UD_TURKISH': [13, 17, 19]

}

all_acc_shapley = {eval_task_name:[] for eval_task_name in ALL_EVAL_TASKS}
all_acc_baseline = {eval_task_name:[] for eval_task_name in ALL_EVAL_TASKS}
all_acc_baseline_s = {eval_task_name:[] for eval_task_name in ALL_EVAL_TASKS}
is_tune=True

BASELINES_S = 'baseline-s'

if not is_tune: num_train_epochs=4.0


for eval_task_name in ['UD_FINNISH']:
    if len(shpley_removals[eval_task_name])<1: continue
    for i in range(1):
        seed = 43
        np.random.seed(seed)
        for is_few_shot in [False]:
            best_shapley_learning_rate = None
            best_shapley_per_gpu_train_batch_size = None

            best_baseline_learning_rate = None
            best_baseline_per_gpu_train_batch_size = None

            best_baseline_s_learning_rate = None
            best_baseline_s_per_gpu_train_batch_size = None

            BEST_BASELINE_ACC = None
            BEST_SHAPLEY_ACC = None

            for is_Shapley in [ BASELINES_S,]:

                best_learning_rate = None
                best_per_gpu_train_batch_size = None
                best_acc = -1

                if BEST_BASELINE_ACC and BEST_SHAPLEY_ACC and BEST_BASELINE_ACC > BEST_SHAPLEY_ACC: continue

                # _______________________________________________________________________
                # ______________________________________NLPDV____________________________________
                ALL_BINARY_TASKS = [
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

                if is_Shapley==BASELINES_S:
                    raddom_domains = np.random.choice(np.arange(len(ALL_BINARY_TASKS)), \
                                                  len(shpley_removals[eval_task_name]), replace=False)

                learning_rates = [  2e-5, 3e-5, 5e-5]
                bz_szs = [    16, 32]

                for learning_rate in learning_rates:
                    for per_gpu_train_batch_size in bz_szs:

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

                        # if is_Shapley == False and eval_task_name == 'UD_ENGLISH':
                        #     write_indices_to_delete(indices_to_delete_file_path,\
                        #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29])

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
                        if evaluate_during_training: run_command += ' --evaluate_during_training '


                        # For training:
                        train_run_command = run_command + ' --do_train --task_name ' + train_task_name + \
                                            ' --data_dir ' + train_data_dir + ' --output_dir ' + \
                                            train_output_dir + ' --model_name_or_path ' + train_model_name_or_path



                        if is_Shapley: train_run_command += ' --indices_to_delete_file_path ' + indices_to_delete_file_path

                        if is_few_shot : train_run_command += ' --is_few_shot'


                        command = train_run_command + ' --num_train_epochs 1'
                        print(command, flush=True)

                        if not os.path.exists(os.path.join(eval_output_dir,"pytorch_model.bin")):
                            os.system(command)

                        # initial Eval on whole dataset
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

                        command = eval_run_command
                        print(command, flush=True)
                        os.system(command)

                        try:
                            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                            with open(output_eval_file, "r") as reader:
                                for line in reader:
                                    line = line.strip().split()
                                    key = line[0]
                                    value = line[-1]
                                    if key in ['acc']:
                                        acc = float(value)
                        except:
                            acc = 0

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

                print('-'*100, flush=True)
                print('-Task: ', eval_task_name, flush=True)
                print('-is_Shapley: ', is_Shapley, flush=True)
                print('-best lr: ', best_learning_rate, '\n-bz sz: ', best_per_gpu_train_batch_size, \
                      '\n-best acc: ', best_acc, '\n-all_acc_shapley: ', all_acc_shapley, \
                       '\n-all_acc_shapley_baseline_s: ',all_acc_baseline_s,'\n- all_acc_baseline: ', all_acc_baseline,  flush=True)
                print('-'*100, flush=True)


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
                                    train_output_dir + ' --model_name_or_path ' + train_model_name_or_path

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
                eval_run_command = run_command + ' --do_predict  --task_name ' + eval_task_name + \
                                   ' --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                                   ' --model_name_or_path ' + eval_output_dir

                indices_to_delete_file_path = eval_output_dir + '/indices_to_delete_file_path' + '.json'
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

                print('-' * 100, flush=True)
                print("Task: ", train_task_name, flush=True)
                print("best_learning_rate: ", best_learning_rate, flush=True)
                print("best_per_gpu_train_batch_size: ", best_per_gpu_train_batch_size, flush=True)
                print("BEST TEST Acc: ", acc, flush=True)
                print("Shapely: ", str(is_Shapley), flush=True)
                print('-' * 100, flush=True)

                if is_Shapley==True:
                    best_shapley_learning_rate = best_learning_rate
                    best_shapley_per_gpu_train_batch_size = best_per_gpu_train_batch_size
                    BEST_SHAPLEY_ACC = acc
                elif is_Shapley==BASELINES_S:
                    best_baseline_s_learning_rate = best_learning_rate
                    best_baseline_s_per_gpu_train_batch_size = best_per_gpu_train_batch_size
                else:
                    best_baseline_learning_rate = best_learning_rate
                    best_baseline_per_gpu_train_batch_size = best_per_gpu_train_batch_size
                    BEST_BASELINE_ACC = acc


            best_shapley_dir  = 'temp/'+eval_task_name+'_output_Shapley_'+str(best_shapley_per_gpu_train_batch_size)+'_'+\
                   str(best_shapley_learning_rate)+'/best/'
            gold = best_shapley_dir+'test_gold.txt'
            shapley = best_shapley_dir+'test_predictions.txt'
            baseline = 'temp/'+eval_task_name+'_output_baseline_'+str(best_baseline_per_gpu_train_batch_size)+'_'+\
                   str(best_baseline_learning_rate)+'/best/'+'test_predictions.txt'
            baseline_s = 'temp/'+eval_task_name+'_output_baseline-s_'+str(best_baseline_s_per_gpu_train_batch_size)+'_'+\
                   str(best_baseline_s_learning_rate)+'/best/'+'test_predictions.txt'

            print('-'*100, flush=True)
            print('Boostrap paired test of Shapley woth baseline!', flush=True)
            command = "python script_t_test.py "+ gold + ' '+ shapley + ' ' + baseline
            print(command, flush=True)
            print('-' * 50, flush=True)
            os.system(command)
            print('-' * 50, flush=True)
            print('-' * 50, flush=True)
            print('Boostrap paired test of Shapley woth baseline-s!', flush=True)
            command = "python script_t_test.py " + gold + ' ' + shapley + ' ' + baseline_s
            print(command, flush=True)
            print('-' * 50, flush=True)
            os.system(command)
            print('-' * 100, flush=True)
