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

# For full_training set None

# _______________________________________________________________________
# ______________________________________NLPDV____________________________________

# Model params
BASE_DATA_DIR = '/local/rizwan/NLPDV/XNLI'
run_file = './examples/run_sxnli.py'
# run_file= './examples/data_valuation.py'
model_type = 'bert'
train_model_name_or_path = 'bert-base-multilingual-cased'  # 'bert-large-uncased-whole-word-masking'
do_lower_case = False
num_train_epochs = 1.0
num_eval_epochs = 2.0
per_gpu_eval_batch_size = 32
per_gpu_train_batch_size = 32
learning_rate = 5e-5
max_seq_length = 128
save_steps = -1
fp16 = True
overwrite_cache = False



# CUDA gpus
CUDA_VISIBLE_DEVICES = [0,1,2,3]
# _______________________________________________________________________
# ______________________________________NLPDV____________________________________

# Debug data size
train_data_size = 180000
eval_data_size = 100000000
cluster_size = 10
cluster_num = train_data_size // cluster_size
# Seed
seed = 43
max_iter = 10
save_every = 1
# total _iter = max_iter x save_every
tolerance = 0.2
err = 0.3



# _______________________________________________________________________
# ______________________________________NLPDV____________________________________

# Shapley params
metric = 'acc'
sources = np.array([i // cluster_size for i in range(train_data_size)])
loo_run = True
tmc_run = True
g_run = False
load_removing_performance_plot = load_adding_performance_plot = True

overwrite_directory = False
load_shapley = False # True when just load shapley's when to plot only

ALL_EVAL_TASKS = 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split()

small_performance_dict = {
    'es': {'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7738955823293173,
           'el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7734939759036145,
           'de_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7763052208835342,
           'de_el_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7742971887550201,
           'de_el_bg_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7738955823293173,
           'de_el_bg_ru_ar_vi_th_zh_hi_sw_ur_en_fr': 0.770281124497992,
           'de_el_bg_ru_tr_vi_th_zh_hi_sw_ur_en_fr': 0.7682730923694779,
           'de_el_bg_ru_tr_ar_th_zh_hi_sw_ur_en_fr': 0.7751004016064257,
           'de_el_bg_ru_tr_ar_vi_zh_hi_sw_ur_en_fr': 0.7763052208835342,
           'de_el_bg_ru_tr_ar_vi_th_hi_sw_ur_en_fr': 0.772289156626506,
           'de_el_bg_ru_tr_ar_vi_th_zh_sw_ur_en_fr': 0.7694779116465863,
           'de_el_bg_ru_tr_ar_vi_th_zh_hi_ur_en_fr': 0.7799196787148595,
           'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_en_fr': 0.7714859437751004,
           'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_fr': 0.7650602409638554,
           'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en': 0.7626506024096386},
                          'de': {'el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7421686746987952},
                          'el': {'de_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7016064257028113},
                          'bg': {'de_el_ru_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7224899598393574},
                          'ru': {'de_el_bg_tr_ar_vi_th_zh_hi_sw_ur_en_fr': 0.7216867469879518},
                          'tr': {'de_el_bg_ru_ar_vi_th_zh_hi_sw_ur_en_fr': 0.6654618473895583},
                          'ar': {'de_el_bg_ru_tr_vi_th_zh_hi_sw_ur_en_fr': 0.6839357429718875},
                          'vi': {'de_el_bg_ru_tr_ar_th_zh_hi_sw_ur_en_fr': 0.7393574297188755},
                          'th': {'de_el_bg_ru_tr_ar_vi_zh_hi_sw_ur_en_fr': 0.5923694779116466},
                          'zh': {'de_el_bg_ru_tr_ar_vi_th_hi_sw_ur_en_fr': 0.7465863453815261},
                          'hi': {'de_el_bg_ru_tr_ar_vi_th_zh_sw_ur_en_fr': 0.6622489959839357},
                          'sw': {'de_el_bg_ru_tr_ar_vi_th_zh_hi_ur_en_fr': 0.5381526104417671},
                          'ur': {'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_en_fr': 0.6365461847389559},
                          'en': {'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_fr': 0.8036144578313253},
                          'fr': {'de_el_bg_ru_tr_ar_vi_th_zh_hi_sw_ur_en': 0.7586345381526104}}


print('small_performance_dict: ', small_performance_dict, flush=True)

full_performance_dict = {eval_task_name: {} for eval_task_name in ALL_EVAL_TASKS}

for eval_task_name in ALL_EVAL_TASKS[1:]:

    train_task_name = eval_task_name
    print('-' * 50, flush=True)
    print("Task name: ", eval_task_name, flush=True)
    print('-' * 50, flush=True)
    train_output_dir = 'temp/' + train_task_name + '_XNLI/'  # +str(seed)+'/'
    eval_output_dir = 'temp/' + eval_task_name + '_XNLI/'  # +str(seed)+'/'
    loo_run = True

    directory = train_output_dir
    indices_to_delete_file_path = directory + '/indices_to_delete_file_path_' + str(seed) + '.json'

    n_points_file = os.path.join(train_output_dir, "training_results" + ".txt")

    ALL_BINARY_TASKS = 'es de el bg ru tr ar vi th zh hi sw ur en fr'.split()

    DOMAIN_TRANSFER = True

    # _______________________________________________________________________
    # ______________________________________NLPDV____________________________________
    if eval_task_name in ALL_BINARY_TASKS: ALL_BINARY_TASKS.remove(eval_task_name)
    if DOMAIN_TRANSFER:
        sources = np.arange(len(ALL_BINARY_TASKS))

    if load_shapley:
        tmc_run = False

    train_data_dir = eval_data_dir = BASE_DATA_DIR

    name = train_task_name + '_' + eval_task_name

    tol = None
    mean_score = None

    np.random.seed(seed)

    run_command = "CUDA_VISIBLE_DEVICES=" + str(CUDA_VISIBLE_DEVICES[0])
    for i in CUDA_VISIBLE_DEVICES[1:]:
        run_command += ',' + str(i)
    run_command += ' python '

    if len(CUDA_VISIBLE_DEVICES) > 1: run_command += '-m torch.distributed.launch --nproc_per_node ' \
                                                     + str(len(CUDA_VISIBLE_DEVICES))
    run_command += ' ' + run_file + ' ' + ' --model_type ' + model_type + \
                   ' --max_seq_length ' + str(max_seq_length) + ' --per_gpu_eval_batch_size=' + str(
        per_gpu_eval_batch_size) + \
                   ' --per_gpu_train_batch_size=' + str(per_gpu_train_batch_size) + ' --learning_rate ' + str(
        learning_rate) \
                   + ' --overwrite_output_dir ' + ' --save_steps ' + str(save_steps)
    if do_lower_case: run_command += '--do_lower_case '
    if fp16: run_command += ' --fp16 '

    if overwrite_cache:
        run_command += ' --overwrite_cache '

    # For training:
    train_run_command_full = run_command + ' --do_train --data_dir ' + train_data_dir + ' --output_dir ' + \
                             train_output_dir + ' --model_name_or_path ' + train_model_name_or_path + ' --language ' + eval_task_name

    train_run_command = train_run_command_full + ' --data_size ' + str(train_data_size)

    # For eval:
    eval_run_command_full = run_command + ' --do_eval --data_dir ' + eval_data_dir + ' --output_dir ' + eval_output_dir + \
                            ' --model_name_or_path ' + train_output_dir

    eval_run_command = eval_run_command_full

    n_points = None


    # _______________________________________________________________________
    # ______________________________________NLPDV____________________________________

    def _which_parallel(directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                   for name in previous_results if 'mem_g' in name]
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0'
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0'
        return tmc_number, g_number


    def write_indices_to_delete(indices_to_delete_file_path, ids):
        with open(indices_to_delete_file_path, "w") as writer:
            print(f"***** Writing ids to {str(indices_to_delete_file_path)}  *****", flush=True)
            for id in ids:
                writer.write("%s " % (id))


    def _create_results_placeholder(directory, tmc_number, mem_tmc, idxs_tmc, g_number=None, mem_g=None, idxs_g=None):
        tmc_dir = os.path.join(
            directory,
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        pkl.dump({'mem_tmc': mem_tmc, 'idxs_tmc': idxs_tmc},
                 open(tmc_dir, 'wb'))
        if mem_g and idxs_g and g_number:
            g_dir = os.path.join(
                directory,
                'mem_g_{}.pkl'.format(g_number.zfill(4))
            )
            pkl.dump({'mem_g': mem_g, 'idxs_g': idxs_g}, open(g_dir, 'wb'))


    def _calculate_loo_vals(sources, baseline_value, n_points, eval_output_dir, train_run_command, eval_run_command,
                            n_points_file):
        """Calculated leave-one-out values for the given metric.

        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Leave-one-out scores
        """

        print('Starting LOO score calculations!', flush=True)
        vals_loo = np.zeros(n_points)
        counter = 0
        for i in tqdm(sources.keys()):
            ''' write the ids in a file so that model codes can access it. 
            train model on the dataset exclusive the sources ids 
            parse result and calculate the result_score or removed_value'''
            counter += 1
            print('=' * 50, flush=True)
            print(f'Calculating LOO score for {counter}/{len(sources)}!', flush=True)
            print('=' * 50, flush=True)

            data_combination = sorted(list(sources.keys()))
            data_combination.remove(i)
            data_combination_name = '_'.join([ALL_BINARY_TASKS[id] for id in data_combination])
            removed_value = None
            if data_combination_name not in small_performance_dict[eval_task_name]:

                write_indices_to_delete(indices_to_delete_file_path, sources[i])
                command = train_run_command + ' --LOO --seed ' + str(
                    seed) + ' --indices_to_delete_file_path ' + indices_to_delete_file_path
                print(command, flush=True)
                os.system(command)

                # parse file and set n_points
                with open(n_points_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key == 'n_points':
                            n_points2 = int(value)

                for eval_task in ALL_EVAL_TASKS:
                    if eval_task in data_combination_name: continue
                    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                    print('For Eval_task: ', eval_task, ' Removing: ', output_eval_file, flush=True)
                    os.system('rm ' + output_eval_file)
                    command = eval_run_command + ' --seed ' + str(seed) + ' ' + ' --language ' + eval_task
                    print(command, flush=True)
                    os.system(command)

                    # parse file and set n_points
                    new_score2 = None

                    with open(output_eval_file, "r") as reader:
                        for line in reader:
                            line = line.strip().split()
                            key = line[0]
                            value = line[-1]
                            if key in ['acc']:
                                new_score2 = float(value)
                    assert new_score2
                    small_performance_dict[eval_task].update({data_combination_name: new_score2})




            removed_value = small_performance_dict[eval_task_name][data_combination_name]
            if not removed_value: pdb.set_trace()
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])

        print("After Loo: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
              full_performance_dict, flush=True)

        return vals_loo


    def save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc, \
                     g_number=None, mem_g=None, idxs_g=None, sources=None, overwrite=False, n_points=None, tol=None, \
                     baseline_value=None, random_score=None, mean_score=None):
        """Saves results computed so far."""
        if directory is None:
            return
        loo_dir = os.path.join(directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            data_dict = {'loo': vals_loo, 'n_points': n_points, 'tol': tol, 'sources': sources, \
                         'baseline_value': baseline_value, 'mean_score': mean_score, 'random_score': random_score}
            pkl.dump(data_dict, open(loo_dir, 'wb'))

        performance_dir = os.path.join(directory, 'perf.pkl')
        pkl.dump({'small_performance_dict': small_performance_dict, 'full_performance_dict': full_performance_dict}, \
                 open(performance_dir, 'wb'))

        tmc_dir = os.path.join(
            directory,
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        pkl.dump({'mem_tmc': mem_tmc, 'idxs_tmc': idxs_tmc},
                 open(tmc_dir, 'wb'))
        if g_number and mem_g and idxs_g:
            g_dir = os.path.join(
                directory,
                'mem_g_{}.pkl'.format(g_number.zfill(4))
            )
            pkl.dump({'mem_g': mem_g, 'idxs_g': idxs_g},
                     open(g_dir, 'wb'))


    def error(mem, min_convergence_iter=50):
        if len(mem) < min_convergence_iter:
            return 1.0
        # if min_convergence_iter>50: min_convergence_iter= len(mem)
        all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[
                   -min_convergence_iter:]  # (100 or min_convergence_iter last iterations, train size)
        errors = np.mean(np.abs(all_vals[-min_convergence_iter:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12),
                         -1)  # (100 or min_convergence_iter last iterations)

        return np.max(errors)


    def _merge_parallel_results(key, directory, n_points, sources, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(directory)
                   if 'mem_{}'.format(key) in name]
        mem = np.zeros((0, n_points))
        n_sources = n_points if sources is None else len(sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(n_points)
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                directory,
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir, flush=True)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            directory,
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs},
                 open(merged_dir, 'wb'))
        return mem, idxs, vals


    def merge_results(directory, n_points, sources, max_samples=None, g_run=False):
        """Merge all the results from different runs.

        Returns:
            combined marginals, sampled indexes and values calculated
            using the two algorithms. (If applicable)
        """
        tmc_results = _merge_parallel_results('tmc', directory, n_points, sources, max_samples)
        mem_tmc, idxs_tmc, vals_tmc = tmc_results

        mem_g, idxs_g, vals_g = None, None, None
        if g_run:
            g_results = _merge_parallel_results('g', directory, max_samples)
            mem_g, idxs_g, vals_g = g_results
        return mem_tmc, idxs_tmc, vals_tmc, mem_g, idxs_g, vals_g


    def performance_plots(directory, vals, train_task_name, eval_task_name, train_run_command, eval_run_command, \
                          random_score, n_points, name=None, num_plot_markers=20, sources=None, rnd_iters=2,
                          length=None):
        """Plots the effect of removing valuable points.

        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8, 8
        plt.rcParams['font.size'] = 25
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]])
                                  for i in range(len(sources.keys()))])
                        for val in vals]  # ([TMC_shapley, LOO_vals, .. ] x train_size)
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0,
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys()) // num_plot_markers, 1)
        )
        removing_performance_dir = os.path.join(directory, train_task_name + '_' + eval_task_name + '_' + str(
            n_points) + '_' + name + '_removing_performance.pkl')
        if os.path.exists(removing_performance_dir) and load_removing_performance_plot:
            print(f'loading {removing_performance_dir}', flush=True)
            data = pkl.load(open(removing_performance_dir, 'rb'))
            perfs = data['perfs']
            rnd = data['rnd']
        else:
            perfs = []
            rnd = []
            if not length:

                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Removal LOO/Shapley: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we remove points in descending {val_}", flush=True)
                    print("=" * 60, flush=True)

                    perfs.append(_portion_performance(np.argsort(vals_source)[::-1], \
                                                      plot_points, train_run_command, eval_run_command, random_score,
                                                      n_points, sources=sources, val_=val_))
                    if val_ == "TMC_Shapley":
                        val_ = "LOO"
                    else:
                        val_ = "GShap"

                for itr in trange(rnd_iters, desc='Ploting Removal Random: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we remove points in random iter {itr}", flush=True)
                    print("=" * 60, flush=True)

                    rnd.append(_portion_performance(np.random.permutation( \
                        np.argsort(vals_sources[0])[::-1]), plot_points, \
                        train_run_command, eval_run_command, random_score, n_points, sources=sources, val_='rnd',
                        itr=itr,
                        max_itr=rnd_iters))

            else:

                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Removal LOO/Shapley: '):
                    print("=" * 40, flush=True)
                    print(f"Calulating performace if we remove points in descending {val_}", flush=True)
                    print("=" * 40, flush=True)

                    perfs.append(_portion_performance(np.argsort(vals_source)[::-1][:length], \
                                                      plot_points, train_run_command, eval_run_command, random_score,
                                                      n_points, sources=sources, val_=val_))
                    if val_ == "TMC_Shapley":
                        val_ = "LOO"
                    else:
                        val_ = "GShap"

                for itr in trange(rnd_iters, desc='Ploting Removal Random: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we remove points in random iter {itr}", flush=True)
                    print("=" * 60, flush=True)

                    rnd.append(_portion_performance(np.random.permutation( \
                        np.argsort(vals_sources[0])[::-1][:length]), plot_points[:length], \
                        train_run_command, eval_run_command, random_score, n_points, sources=sources, val_='rnd',
                        itr=itr,
                        max_itr=rnd_iters))

            rnd = np.mean(rnd, 0)
            data_dict = {'perfs': perfs, 'rnd': rnd}
            pkl.dump(data_dict, open(removing_performance_dir, 'wb'))

        plt.plot(plot_points / n_points * 100, perfs[0] * 100,
                 '-', lw=5, ms=10, color='b')
        if len(vals) == 3:
            plt.plot(plot_points / n_points * 100, perfs[1] * 100,
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals) == 2:
            legends = ['TMC-Shapley ', 'LOO', 'Random']
        else:
            legends = ['TMC-Shapley ', 'Random']
        plt.plot(plot_points / n_points * 100, perfs[-1] * 100,
                 '-.', lw=5, ms=10, color='g')
        plt.plot(plot_points / n_points * 100, rnd * 100,
                 ':', lw=5, ms=10, color='r')
        plt.legend(legends)
        if directory is not None and name is not None:
            plt.savefig(os.path.join(
                directory, 'plots', '{}.png'.format(name + '_removing_')),
                bbox_inches='tight')
            plt.close()


    def _portion_performance(idxs, plot_points, train_run_command, eval_run_command, random_score, n_points,
                             sources=None,
                             val_='TMC_SHAPLEY', itr=None, max_itr=None):
        """Given a set of indexes, starts removing points from
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        scores = []
        all_ids = np.array(range(n_points))
        for i in trange(len(plot_points), 0, -1,
                        desc='Inside ' + val_ + ' _portion_performance: removing sources in descending'):

            keep_idxs = np.concatenate([sources[idx] for idx
                                        in idxs[plot_points[i - 1]:]], -1)

            new_score = None

            data_combination = sorted(idxs[plot_points[i - 1]:])
            data_combination_name = '_'.join([ALL_BINARY_TASKS[id] for id in data_combination])
            if data_combination_name in full_performance_dict[eval_task_name]:
                new_score = full_performance_dict[eval_task_name][data_combination_name]
            else:
                write_indices_to_delete(indices_to_delete_file_path, np.setdiff1d(all_ids, keep_idxs))  # ids to remove

                command = train_run_command + ' --seed ' + str(
                    seed) + ' --indices_to_delete_file_path ' + indices_to_delete_file_path
                print('=' * 100, flush=True)
                print(
                    f'Training _portion_performance for {val_} progress {i}/{len(plot_points)} training on {len(keep_idxs)} dataset',
                    flush=True)
                if itr:  print(f'iteration {itr}/{max_itr}', flush=True)
                print(command, flush=True)
                print('=' * 100, flush=True)
                os.system(command)

                # parse file and set n_points
                n_points2 = None
                with open(n_points_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key == 'n_points' and value != "None":
                            n_points2 = int(value)

                ''' #Below code is not for domain sleection
                if not n_points2:
                    scores.append(random_score)
                    continue


                elif n_points2 != len(keep_idxs):
                    print(f'n_points2({n_points2}) != len(keep_idxs)({len(keep_idxs)} in removing plots)', flush=True)
                    continue 
                '''

                command = eval_run_command + ' --seed ' + str(seed) + ' --language ' + eval_task_name
                print('=' * 100, flush=True)
                print(f'Evaluating _portion_performance for {val_} progress {i}/{len(plot_points)}', flush=True)
                print(command, flush=True)
                print('=' * 100, flush=True)
                os.system(command)

                # parse file and set n_points

                output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                with open(output_eval_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key in ['acc']:
                            new_score = float(value)
                assert new_score
                full_performance_dict[eval_task_name].update({data_combination_name: new_score})
                # print("After _portion_performance small_performance_dict: ", small_performance_dict,
                #       'full_performance_dict:',
                #       full_performance_dict, flush=True)

            scores.append(new_score)

        return np.array(scores)[::-1]


    def shapley_value_plots(directory, vals, n_points, name=None, num_plot_markers=20, sources=None):
        """Plots the effect of removing valuable points.

        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 25, 25
        plt.rcParams['font.size'] = 25
        plt.xlabel('\t \t'.join([ALL_BINARY_TASKS[i] for i in sources.keys()]))
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]])
                                  for i in range(len(sources.keys()))])
                        for val in vals]  # (['TMC-Shapley ', 'G-Shapley ', 'LOO'] x train_size)
        if isinstance(num_plot_markers, str):
            num_plot_markers = len(sources.keys())
        elif len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys())
        plot_points = np.arange(
            0,
            max(len(sources.keys()), num_plot_markers),
            max(len(sources.keys()) // num_plot_markers, 1)
        )

        plt.plot(plot_points / n_points * 100, vals_sources[0] * 100,
                 '-', lw=5, ms=10, color='b')
        if len(vals) == 3:
            plt.plot(plot_points / n_points * 100, vals_sources[1] * 100,
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO']
        elif len(vals) == 2:
            legends = ['TMC-Shapley ', 'LOO']
        else:
            legends = ['TMC-Shapley ']
        plt.plot(plot_points / n_points * 100, vals_sources[-1] * 100,
                 '-.', lw=5, ms=10, color='g')

        plt.legend(legends)
        if directory is not None and name is not None:
            plt.savefig(os.path.join(
                directory, 'plots', '{}.png'.format(name + '_shapley_values_')),
                bbox_inches='tight')
            plt.close()


    def performance_plots_adding(directory, vals, train_task_name, eval_task_name, train_run_command, eval_run_command,
                                 random_score, n_points, name=None, num_plot_markers=20,
                                 sources=None, rnd_iters=1, length=None):
        """Plots the effect of removing valuable points.

        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8, 8
        plt.rcParams['font.size'] = 25
        plt.xlabel('Fraction of train data added (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]])
                                  for i in range(len(sources.keys()))])
                        for val in vals]  # ([TMC_shapley, LOO_vals, .. ] x train_size)

        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys())
        plot_points = np.arange(
            0,
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys()) // num_plot_markers, 1)
        )

        adding_performance_dir = os.path.join(directory, train_task_name + '_' + eval_task_name + '_' + str(
            n_points) + '_' + name + '_adding_performance.pkl')

        if os.path.exists(adding_performance_dir) and load_adding_performance_plot:
            print(f'loading {adding_performance_dir}', flush=True)
            data = pkl.load(open(adding_performance_dir, 'rb'))
            perfs = data['perfs']
            rnd = data['rnd']
            try:
                rnd_prmtn = data['rnd_prmtn']
            except:
                rnd_prmtn = np.random.permutation( \
                    np.argsort(vals_sources[0])[::-1])
        else:
            print(f'len(source): {len(sources)}', flush=True)
            # pdb.set_trace()
            perfs = []
            rnd = []
            if not length:

                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Adding LOO/Shapley: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we add points in descending {val_}", flush=True)
                    print("=" * 60, flush=True)
                    perfs.append(_portion_performance_addition(np.argsort(vals_source)[::-1], \
                                                               plot_points, train_run_command, eval_run_command,
                                                               random_score, n_points, sources=sources, val_=val_))
                    if val_ == "TMC_Shapley":
                        val_ = "LOO"
                    else:
                        val_ = "GShap"

                # Random part
                for itr in trange(rnd_iters, desc='Ploting Adding Random: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we add points in random iter {itr}", flush=True)
                    print("=" * 60, flush=True)
                    rnd_prmtn = np.random.permutation( \
                        np.argsort(vals_sources[0])[::-1])
                    rnd.append(_portion_performance_addition(rnd_prmtn, plot_points, \
                                                             train_run_command, eval_run_command, random_score,
                                                             n_points, sources=sources, val_='rnd', itr=itr,
                                                             max_itr=rnd_iters))

            else:
                perfs = []
                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Adding LOO/Shapley: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we add points in descending {val_}", flush=True)
                    print("=" * 60, flush=True)

                    perfs.append(_portion_performance_addition(np.argsort(vals_source)[::-1][:length], \
                                                               plot_points, train_run_command, eval_run_command,
                                                               random_score,
                                                               n_points, sources=sources, val_=val_))

                    if val_ == "TMC_Shapley":
                        val_ = "LOO"
                    else:
                        val_ = "GShap"
                # Random Part
                rnd = []
                for itr in trange(rnd_iters, desc='Ploting Adding Random: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we add points in random iter {itr}", flush=True)
                    print("=" * 60, flush=True)
                    rnd_prmtn = np.random.permutation(np.argsort(vals_sources[0])[::-1])
                    rnd.append(_portion_performance_addition(rnd_prmtn[:length], plot_points[:length], \
                                                             train_run_command, eval_run_command, random_score,
                                                             n_points, sources=sources, val_='rnd', itr=itr,
                                                             max_itr=rnd_iters))

            rnd = np.mean(rnd, 0)

            data_dict = {'perfs': perfs, 'rnd': rnd, 'rnd_prmtn': rnd_prmtn}
            pkl.dump(data_dict, open(adding_performance_dir, 'wb'))

        plt.plot(plot_points / n_points * 100, perfs[0] * 100,
                 '-', lw=5, ms=10, color='b')
        if len(vals) == 3:
            plt.plot(plot_points / n_points * 100, perfs[1] * 100,
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals) == 2:
            legends = ['TMC-Shapley ', 'LOO', 'Random']
            legends = ['TMC-Shapley ', 'LOO']
        else:
            legends = ['TMC-Shapley ', 'Random']
        plt.plot(plot_points / n_points * 100, perfs[-1] * 100,
                 '-.', lw=5, ms=10, color='g')

        plt.plot(plot_points / n_points * 100, rnd * 100,
                 ':', lw=5, ms=10, color='r')

        plt.legend(legends)

        if directory is not None and name is not None:
            plt.savefig(os.path.join(
                directory, 'plots', '{}.png'.format(name + '_adding_new')),
                bbox_inches='tight')
            plt.close()
        print('rnd_pmtn: ', rnd_prmtn, flush=True)
        return perfs


    def _portion_performance_addition(idxs, plot_points, train_run_command, eval_run_command, random_score, n_points,
                                      sources=None, val_="TMC_SHAPELY", itr=None, max_itr=None):
        """Given a set of indexes, starts adding points from
        the first elemnt and evaluates the new model after
        removing each point."""
        # pdb.set_trace()

        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        scores = []
        all_ids = np.array(range(n_points))
        for i in trange(1, len(plot_points) + 1, 1,
                        desc='Inside ' + val_ + ' _portion_performance_addition: adding sources in descending'):

            if i == len(plot_points):
                keep_idxs = np.concatenate([sources[idx] for idx in idxs], -1)
                data_combination = sorted(idxs)
            else:
                keep_idxs = np.concatenate([sources[idx] for idx
                                            in idxs[:plot_points[i]]], -1)
                data_combination = sorted(idxs[:plot_points[i]])
            new_score = None

            data_combination_name = '_'.join([ALL_BINARY_TASKS[id] for id in data_combination])
            if data_combination_name in full_performance_dict[eval_task_name]:
                new_score = full_performance_dict[eval_task_name][data_combination_name]
            else:
                write_indices_to_delete(indices_to_delete_file_path, np.setdiff1d(all_ids, keep_idxs))  # ids to remove

                command = train_run_command + ' --seed ' + str(
                    seed) + ' --indices_to_delete_file_path ' + indices_to_delete_file_path
                print('=' * 100, flush=True)
                print(
                    f'Training _portion_performance_adding for {val_} progress {i}/{len(plot_points)} training on {len(keep_idxs)} dataset iter',
                    flush=True)
                if itr:  print(f'iteration {itr}/{max_itr}', flush=True)
                print(command, flush=True)
                print('=' * 100, flush=True)
                os.system(command)

                # parse file and set n_points
                n_points2 = None
                with open(n_points_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key == 'n_points' and value != "None":
                            n_points2 = int(value)
                '''#Below code is not for domain sleection 
                if not n_points2:
                    scores.append(random_score)
                    continue
                elif n_points2 != len(keep_idxs):
                    print(f'n_points2({n_points2}) != len(keep_idxs)({len(keep_idxs)})', flush=True)
                    continue
                '''
                command = eval_run_command + ' --seed ' + str(seed) + ' --language ' + eval_task_name
                print('=' * 100, flush=True)
                print(f'Evaluating _portion_performance_adding for {val_} progress {i}/{len(plot_points)}', flush=True)
                print(command, flush=True)
                print('=' * 100, flush=True)
                os.system(command)

                # parse file and set n_points

                output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                with open(output_eval_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key in ['acc']:
                            new_score = float(value)
                assert new_score
                full_performance_dict[eval_task_name].update({data_combination_name: new_score})
                print("After _portion_performance_addition: small_performance_dict: ", small_performance_dict,
                      'full_performance_dict:',
                      full_performance_dict, flush=True)

            scores.append(new_score)

        return np.array(scores)


    def _tmc_shap(train_run_command, eval_run_command, eval_output_dir, n_points_file, mem_tmc, idxs_tmc, random_score,
                  mean_score, n_points, iterations, seed, iter_counter, max_iter, tolerance=None, sources=None,
                  print_step=1):
        """Runs TMC-Shapley algorithm.

        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """

        for iteration in trange(iterations, desc='Inside one iter/save_every'):
            # if print_step * (iteration + 1) / iterations % 1 == 0:
            if (iteration + 1) % print_step == 0:
                print('=' * 100, flush=True)
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iter_counter * (iterations) + iteration + 1, max_iter * iterations), flush=True)
                print('=' * 100, flush=True)
            # one iteration
            idxs = np.random.permutation(len(sources))
            marginal_contribs = np.zeros(n_points)
            truncation_counter = 0
            new_score = random_score
            Selected_IDS = []
            all_ids = {id for id in range(n_points)}
            for n, idx in enumerate(tqdm(idxs, desc='Inside one iter/save_every idxs')):
                old_score = new_score
                Selected_IDS += list(sources[idx])  # ids to keep
                new_score = None

                data_combination = sorted(idxs[:n + 1])
                data_combination_name = '_'.join([ALL_BINARY_TASKS[id] for id in data_combination])
                if data_combination_name not in small_performance_dict[eval_task_name]:

                    write_indices_to_delete(indices_to_delete_file_path,
                                            list(all_ids.difference(set(Selected_IDS))))  # ids to remove

                    command = train_run_command + ' --seed ' + str(
                        seed) + ' --indices_to_delete_file_path ' + indices_to_delete_file_path
                    print('=' * 100, flush=True)
                    print('{}/{} TMC_Shapley iterations.'.format(
                        (iter_counter * (iterations) + iteration) * len(sources) + n + 1,
                        (max_iter * iterations) * len(sources)), flush=True)
                    print('=' * 100, flush=True)
                    print(command, flush=True)
                    print('=' * 100, flush=True)
                    os.system(command)

                    # parse file and set n_points
                    n_points2 = None
                    with open(n_points_file, "r") as reader:
                        for line in reader:
                            line = line.strip().split()
                            key = line[0]
                            value = line[-1]
                            if key == 'n_points' and value != "None":
                                n_points2 = int(value)

                    if not n_points2:
                        continue

                    '''#Below code is not for domain sleection
                    elif n_points2 != len(Selected_IDS):
                        print(f'n_points2({n_points2}) != len(Selected_IDS)({len(Selected_IDS)})', flush=True)
                        pdb.set_trace()
                    '''

                    for eval_task in ALL_EVAL_TASKS:
                        if eval_task in data_combination_name: continue
                        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
                        print('Removing: ', output_eval_file, flush=True)
                        os.system('rm ' + output_eval_file)
                        command = eval_run_command + ' --seed ' + str(seed) + ' ' + ' --language ' + eval_task
                        print(command, flush=True)
                        os.system(command)

                        # parse file and set n_points
                        new_score2 = None
                        with open(output_eval_file, "r") as reader:
                            for line in reader:
                                line = line.strip().split()
                                key = line[0]
                                value = line[-1]
                                if key in ['acc']:
                                    new_score2 = float(value)
                        assert new_score2
                        small_performance_dict[eval_task].update({data_combination_name: new_score2})




                new_score = small_performance_dict[eval_task_name][data_combination_name]

                marginal_contribs[sources[idx]] = (new_score - old_score)
                marginal_contribs[sources[idx]] /= len(sources[idx])
                distance_to_full_score = np.abs(new_score - mean_score)
                if distance_to_full_score <= tolerance * mean_score:
                    truncation_counter += 1
                    if truncation_counter > 3:
                        print('=' * 50, flush=True)
                        print('Truncation condition reached for this epoch! ', flush=True)
                        print('=' * 50, flush=True)
                        break
                else:
                    truncation_counter = 0

            mem_tmc = np.concatenate([
                mem_tmc,
                np.reshape(marginal_contribs, (1, -1))
            ])
            idxs_tmc = np.concatenate([
                idxs_tmc,
                np.reshape(idxs, (1, -1))
            ])



        return mem_tmc, idxs_tmc



    # _______________________________________________________________________
    # ______________________________________NLPDV____________________________________

    # Create Shapley Directory
    if overwrite_directory and os.path.exists(directory):
        print('deleting recursive all previous files', flush=True)
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, 'plots'))

    # get baseline_value, random_score, n_points = (len train_dataset), sources etc.,

    print(f'max_iter: {max_iter} save_every {save_every}, total {max_iter * save_every}', flush=True)

    loo_dir = os.path.join(directory, 'loo.pkl')
    vals_loo = None
    if os.path.exists(loo_dir):
        print(f'loading {loo_dir}', flush=True)
        data = pkl.load(open(loo_dir, 'rb'))
        vals_loo = data['loo']
        try:
            n_points = data['n_points']
            baseline_value = data['baseline_value']
            random_score = data['random_score']
            tol = data['tol']
            mean_score = data['mean_score']
        except:
            print('n_points, etc were not stored ', flush=True)

    performance_dir = os.path.join(directory, 'perf.pkl')
    if os.path.exists(performance_dir):
        print(f'loading {performance_dir}', flush=True)
        data = pkl.load(open(performance_dir, 'rb'))
        small_performance_dict = data['small_performance_dict']
        full_performance_dict = data['full_performance_dict']

    # print("Begining of run: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
    #       full_performance_dict, flush=True)

    if sources is None:
        sources = {i: np.array([i]) for i in range(n_points)}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    n_sources = len(sources)

    data_combination = sorted(list(sources.keys()))
    data_combination_name = '_'.join([ALL_BINARY_TASKS[id] for id in data_combination])

    try:
        baseline_value
    except:
        if data_combination_name in small_performance_dict[eval_task_name]:
            baseline_value = small_performance_dict[eval_task_name][data_combination_name]
        elif not n_points:


            print('=' * 50, flush=True)
            print('Running baseline runs', flush=True)
            print('=' * 50, flush=True)

            # initial Training on whole dataset
            command = train_run_command + ' --num_train_epochs ' + str(
                num_train_epochs) + ' --is_baseline_run --seed ' + str(
                seed) + ' '
            print(command, flush=True)
            os.system(command)

            # parse file and set n_points
            if not ALL_BINARY_TASKS:
                with open(n_points_file, "r") as reader:
                    for line in reader:
                        line = line.strip().split()
                        key = line[0]
                        value = line[-1]
                        if key == 'n_points':
                            n_points = int(value)

            else:
                n_points = len(sources)

            # initial Eval on whole dataset
            command = eval_run_command + ' --is_baseline_run --seed ' + str(seed) + ' ' + ' --language '+ eval_task_name
            print(command, flush=True)
            os.system(command)

            # parse file
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "r") as reader:
                for line in reader:
                    line = line.strip().split()
                    key = line[0]
                    value = line[-1]
                    if key in ['acc']:
                        baseline_value = float(value)
                    elif key == 'random_init_result':
                        random_score = float(value)
                    elif key == 'tol':
                        tol = float(value)
                    elif key == 'mean_score':
                        mean_score = float(value)



    if not n_points: n_points = len(sources)
    try:
        random_score
    except:

        random_score = baseline_value #0.5
        mean_score = baseline_value
    if eval_task_name=='es':random_score = baseline_value

    single_mems = {lang: small_performance_dict[eval_task_name][lang] for lang in ALL_BINARY_TASKS if
                   lang in small_performance_dict[eval_task_name]}

    loo_mems = { '_'.join([id for id in ALL_BINARY_TASKS if id!= lang]): small_performance_dict[eval_task_name]['_'.join([id for id in ALL_BINARY_TASKS if id!= lang])]  for lang in ALL_BINARY_TASKS }

    random_score =   np.array(list(single_mems.values()) + [baseline_value]    ).mean()


    ''' for de 
    
    # with random_score = baseline:
    # [5.29390288e-03
    # 7.30193501e-05 - 3.68747718e-03 - 3.68747718e-03
    # -3.13983206e-03
    # 2.81124498e-03
    # 1.49689668e-03
    # 4.52719971e-03
    # 4.49069003e-03 - 2.59218693e-03 - 1.91675794e-02 - 1.09163928e-02
    # 3.65096751e-03
    # 4.52719971e-03]

    # with random_score =  np.array(list(single_mems.values()) + [baseline_value] +   list(loo_mems.values())   ).mean()
    # [0.00516351  0.00129327 - 0.00232119 - 0.00262718 - 0.00226382  0.00235227
    #  0.00259371  0.00434117  0.00407344 - 0.00144148 - 0.01758941 - 0.0102577
    #  0.00384395  0.00582568]

    #random_score =  np.array(list(single_mems.values()) + [baseline_value]   ).mean()
    # Done Shapely Values [ 0.0052939   0.00611902  0.00235853  0.00235853  0.00290617  0.00281124
    #   0.0075429   0.0045272   0.00449069  0.00345382 -0.00707558 -0.00487039
    #   0.00365097  0.0105732 ], Loo vals [ 0.00803213  0.00682731  0.          0.0064257   0.01084337  0.0124498
    #   0.01084337  0.00843373  0.01325301  0.00562249  0.0124498   0.00763052
    #  -0.00040161  0.00963855] Perfs None
    
    
    
    
    
    for es:
    # with random_score =  np.array(list(single_mems.values()) + [baseline_value]   ).mean()
    Done Shapely Values [ 0.00310332  0.01043812  0.00262505  0.00160277  0.00010588  0.00372399
  0.00576488  0.00346842  0.00328587  0.00262505 -0.00858708 -0.00869295
  0.00854326  0.01401606], Loo vals [ 0.00040161 -0.00240964 -0.00040161  0.          0.00361446  0.00562249
 -0.00120482 -0.00240964  0.00160643  0.00441767 -0.0060241   0.00240964
  0.00883534  0.01124498] Perfs None 
  
    # with random_score = np.array(list(single_mems.values()) + [baseline_value] +   list(loo_mems.values())    ).mean()
    Done Shapely Values [ 0.00310332  0.00681058 -0.00100249 -0.00202477 -0.00352166  0.00372399
    #   0.00213734  0.00346842  0.00328587 -0.00100249 -0.01584216 -0.01232049
    #   0.00854326  0.01038852], Loo vals [ 0.00040161 -0.00240964 -0.00040161  0.          0.00361446  0.00562249
    #  -0.00120482 -0.00240964  0.00160643  0.00441767 -0.0060241   0.00240964
    #   0.00883534  0.01124498] Perfs None 
    
    # with baseline:
    Done Shapely Values [ 0.00310332  0.00405257 -0.0037605  -0.00478277 -0.00627966  0.00372399
 -0.00062066  0.00346842  0.00328587 -0.0037605  -0.02135816 -0.0150785
  0.00854326  0.00763052], Loo vals [ 0.00040161 -0.00240964 -0.00040161  0.          0.00361446  0.00562249
 -0.00120482 -0.00240964  0.00160643  0.00441767 -0.0060241   0.00240964
  0.00883534  0.01124498] Perfs None 

    '''




        # pdb.set_trace()
    mem_tmc = np.zeros((0,
                        n_points))  # (n_iter x n_points) #n_iter is basically n_(save_every) which can be a high value and it's not epoch
    idxs_tmc = np.zeros((0, n_sources), int)
    if g_run:
        mem_g = np.zeros((0, n_points))
        idxs_g = np.zeros((0, n_sources), int)
    else:
        mem_g = None
        idxs_g = None


    small_performance_dict[eval_task_name][data_combination_name] = baseline_value

    tmc_number, g_number = _which_parallel(directory)
    if not g_run: g_number = None
    if not load_shapley:
        _create_results_placeholder(directory, tmc_number, mem_tmc, idxs_tmc, g_number, mem_g, idxs_g)

    if loo_run:
        try:
            len(vals_loo)
            if len(vals_loo) != n_points:
                vals_loo = None
            else:
                loo_run = False
        except:
            loo_run = True

    if loo_run:
        try:
            len(vals_loo)
        except:
            print('=' * 50, flush=True)
            print('Beginning LOO  runs', flush=True)
            print('=' * 50, flush=True)
            if 'full_training' in train_output_dir:
                vals_loo = _calculate_loo_vals(sources, baseline_value, n_points, \
                                               eval_output_dir,
                                               train_run_command_full + ' --num_train_epochs ' + str(num_train_epochs), \
                                               eval_run_command + ' --data_size ' + str(eval_data_size), n_points_file)
            else:
                vals_loo = _calculate_loo_vals(sources, baseline_value, n_points, \
                                               eval_output_dir,
                                               train_run_command + ' --num_train_epochs ' + str(num_train_epochs), \
                                               eval_run_command + ' --data_size ' + str(eval_data_size), n_points_file)
            print('LOO values are being saved!', flush=True)
            save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc, g_number, mem_g, idxs_g, sources,
                         overwrite=True,
                         n_points=n_points, tol=tol, \
                         baseline_value=baseline_value, random_score=random_score, mean_score=mean_score)
        print('LOO values calculated and saved!', flush=True)

    if not tolerance:
        pdb.set_trace()


    def run_routine(tmc_run, train_run_command, eval_run_command, eval_data_size, num_train_epochs, eval_output_dir,
                    n_points_file, mem_tmc, idxs_tmc, random_score, mean_score, n_points, save_every, seed, max_iter,
                    tolerance, sources):

        for iter_counter in trange(max_iter, desc='Overall Shpley Value Calculation'):
            if g_run:
                pdb.set_trace()
            if tmc_run:
                # pdb.set_trace()
                error_ = error(mem_tmc, min_convergence_iter=30)
                print('=' * 50, flush=True)
                print(f'Error now {error_}, and err is {err}', flush=True)
                print('=' * 50, flush=True)
                if error_ < err:
                    tmc_run = False
                    print(f'error_{error_}< err{err}: {error_ < err}')
                else:
                    # sets mem_tmc (save_every or #iterations x train_size)
                    print('=' * 120, flush=True)
                    print(f'Start TMC Shapley Runs {iter_counter}/{max_iter}', flush=True)
                    print('=' * 120, flush=True)
                    old_shape = mem_tmc.shape[0]
                    if 'full_training' in train_output_dir:
                        mem_tmc, idxs_tmc = _tmc_shap(
                            train_run_command_full + ' --num_train_epochs ' + str(num_train_epochs), \
                            eval_run_command, eval_output_dir, \
                            n_points_file, mem_tmc, idxs_tmc, random_score, mean_score, \
                            n_points, save_every, seed, iter_counter, max_iter, tolerance,
                            sources,
                            print_step=1)
                    else:
                        mem_tmc, idxs_tmc  = _tmc_shap(
                            train_run_command + ' --num_train_epochs ' + str(num_train_epochs), \
                            eval_run_command, eval_output_dir, \
                            n_points_file, mem_tmc, idxs_tmc, random_score, mean_score, \
                            n_points, save_every, seed, iter_counter, max_iter, tolerance, sources,
                            print_step=1)


                    # sets vals_tmc (train_size,) 1d array

                    if mem_tmc.shape[0] - old_shape != save_every:
                        pdb.set_trace()
                        # break
                    else:
                        vals_tmc = np.mean(mem_tmc, 0)

            if directory is not None:
                if tmc_run:
                    print(f'Saving TMC Shapley after iteration {iter_counter}', flush=True)
                    save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc)
                if g_run: pdb.set_trace()

        print(f'error now{error(mem_tmc)}< err{err}: {err}')
        return mem_tmc, idxs_tmc, vals_tmc


    if not load_shapley: mem_tmc, idxs_tmc, vals_tmc  = run_routine(tmc_run,
                                                                    train_run_command,
                                                                    eval_run_command,
                                                                    eval_data_size,
                                                                    num_train_epochs,
                                                                    eval_output_dir,
                                                                    n_points_file,
                                                                    mem_tmc,
                                                                    idxs_tmc,
                                                                    random_score,
                                                                    mean_score,
                                                                    n_points,
                                                                    save_every,
                                                                    seed, max_iter,
                                                                    tolerance,
                                                                    sources)

    mem_tmc, idxs_tmc, vals_tmc, mem_g, idxs_g, vals_g = merge_results(directory, n_points, sources)

    vals_tmc = np.mean(np.concatenate([mem_tmc, vals_loo.reshape(1, -1)]), 0)


    shapley_value_plots(directory, [vals_tmc, vals_loo], n_points, num_plot_markers=20, sources=sources,
                        name=name + '_' + str(n_points))

    single_mems = {lang: small_performance_dict[eval_task_name][lang] for lang in ALL_BINARY_TASKS if
                   lang in small_performance_dict[eval_task_name]}
    dict = {k: v for k, v in small_performance_dict[eval_task_name].items()}
    best_single_comb = {k: v for k, v in single_mems.items() if v == max(single_mems.values())}


    perfs = None



    print("=" * 50, f'\nDone Shapely Values {vals_tmc}, Loo vals {vals_loo} Perfs {perfs}', "\n", "=" * 50, flush=True)

    pdb.set_trace()

    try:
        del baseline_value
    except:
        pdb.set_trace()
