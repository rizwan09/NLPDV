
from flair.data import Corpus
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import MultiCorpus
from pathlib import Path
import torch
import pdb, _pickle as pkl
import os, numpy as np, random
from tqdm import trange, tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

ALL_POS_DOMAINS = 'wsj_emails_newsgroups_answers_reviews_weblogs'.split('_')
ALL_POS_ALL_CORPUS = {}
target_domain = 'newsgroups'

loo_run = True
tmc_run = True
g_run = False
load_removing_performance_plot = load_adding_performance_plot = True

overwrite_directory = False  # True when just load shapley's when to plot only
load_shapley = False

train_data_size = 2000 # For full_training set None
num_train_epochs = 1
seed = 43
err = 0.2
tolerance = 0.1
max_iter = 50
save_every = 1

data_dir = '/home/rizwan/NLPDV/SANCL/POS/gweb_sancl/parse/'
columns = {1: 'text', 3: 'pos'}
tag_type = 'pos'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


for domain in ALL_POS_DOMAINS:
    corpus: Corpus = ColumnCorpus(data_dir + domain, columns, \
                                  dev_file='dev.conll',
                                  train_file='train.conll',
                                  test_file='test.conll')
    ALL_POS_ALL_CORPUS.update({domain: corpus})
    print('-' * 150, flush=True)
    print(" - Domain: ", domain, flush=True)
    print(" - Corpus Train Size: ", len(corpus.train), flush=True)
    print(" - Corpus Dev Size: ", len(corpus.dev), flush=True)
    print(" - Corpus Test Size: ", len(corpus.test), flush=True)
    print('-' * 150, flush=True)



small_performance_dict = { domain: {} for domain in ALL_POS_DOMAINS }
full_performance_dict = { domain: {} for domain in ALL_POS_DOMAINS }



for domain in ALL_POS_DOMAINS:
    target_domain =domain
    print('-'*50, flush=True)
    print("Target name: ", target_domain, flush=True)
    print('-' * 50, flush=True)





    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('extvec'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    multi_all_corpus = MultiCorpus(ALL_POS_ALL_CORPUS.values())
    # 3. make the tag dictionary from the corpus
    tag_dictionary = multi_all_corpus.make_tag_dictionary(tag_type=tag_type)

    # initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)


    train_output_dir = 'resources/taggers/example-pos/'+target_domain+'_shapley'

    def train_model(current_sources, target_domain, train_data_size=train_data_size, data_combination_name=None, eval_on_dev=True, ALL_POS_DOMAINS=ALL_POS_DOMAINS, small_performance_dict=small_performance_dict):
        source_target_corpus = ALL_POS_ALL_CORPUS[target_domain]
        if eval_on_dev: source_target_corpus._test = source_target_corpus._dev
        source_target_corpus._train = []
        source_names = [ALL_POS_DOMAINS[i] for i in current_sources]
        for domain in source_names:
            temp_data = ALL_POS_ALL_CORPUS[domain]._train
            if train_data_size:
                if len(ALL_POS_ALL_CORPUS[domain].train) > train_data_size:
                    temp_data = random.choices(ALL_POS_ALL_CORPUS[domain]._train, k=train_data_size)
            source_target_corpus._train += temp_data

        if not data_combination_name: data_combination_name = '_'.join(source_names)

        trainer: ModelTrainer = ModelTrainer(tagger, source_target_corpus, use_tensorboard=True)
        result_dict = trainer.train(train_output_dir, max_epochs=num_train_epochs)
        small_performance_dict[target_domain].update({data_combination_name: result_dict['test_score']})
        for domain in ALL_POS_DOMAINS:
            print('-'*50, flush=True)
            print(' - ')
            if domain != target_domain: # Already done
                temp_corpus = ALL_POS_ALL_CORPUS[target_domain]
                trainer: ModelTrainer = ModelTrainer(tagger, temp_corpus, use_tensorboard=True)
                score = trainer.final_test(Path(train_output_dir), eval_mini_batch_size=32)
                small_performance_dict[target_domain].update({data_combination_name:score})

        return result_dict['test_score']

    def get_baseline_results(current_sources, target_domain, train_data_size=train_data_size, num_bags=20, eval_on_dev=True):
        source_target_corpus = ALL_POS_ALL_CORPUS[target_domain]
        if eval_on_dev: source_target_corpus._test = source_target_corpus._dev
        baseline_value = train_model(current_sources, target_domain, train_data_size, eval_on_dev)
        scores = []
        for _ in range(num_bags):
            temp_corpus=source_target_corpus
            bag_idxs = np.random.choice(len(source_target_corpus.test), len(source_target_corpus.test))
            temp_corpus._test = [ source_target_corpus._test[i] for i in  bag_idxs]
            trainer: ModelTrainer = ModelTrainer(tagger, temp_corpus, use_tensorboard=True)
            scores.append(trainer.final_test(Path(train_output_dir), eval_mini_batch_size=32))
        tol =  np.std(scores)
        mean_score = np.mean(scores)
        dist = source_target_corpus.get_tag_distribution(corpus_name='test') # dev, test are same if eval_on_dev==True
        random_score = max(dist.values()) / sum(dist.values())
        return baseline_value, random_score, tol, mean_score



    source_names = ALL_POS_DOMAINS.copy()
    source_names.remove(target_domain)
    sources = np.arange(len(source_names))
    n_points = len(sources)
    directory = train_output_dir
    # baseline_value, tol, mean_score = get_baseline_results(sources, target_domain=target_domain, train_data_size=train_data_size)


    # print('Target: ', target_domain, "baseline result trained with full source train datasets: ", baseline_value, tol, mean_score, flush=True)

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


    def _calculate_loo_vals(sources, baseline_value, n_points, target_domain, train_data_size):
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
            data_combination_name = '_'.join([ALL_POS_DOMAINS[id] for id in data_combination])
            print('-' * 150, flush=True)
            print(" - Loo for Domain: ", ALL_POS_DOMAINS[i], flush=True)
            print('-' * 150, flush=True)
            if data_combination_name in small_performance_dict[target_domain]:
                removed_value = small_performance_dict[target_domain][data_combination_name]
            else:
                removed_value = train_model(data_combination, target_domain, train_data_size, data_combination_name, ALL_POS_DOMAINS, small_performance_dict)

            if not removed_value: pdb.set_trace()
            print('baseline_value: ', baseline_value, 'removed value: ', removed_value, 'baseline_value - removed_value: ', baseline_value - removed_value, flush=True)
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


    def performance_plots(directory, vals, random_score, n_points, target_domain=target_domain, train_data_size=train_data_size, \
                           name=None, num_plot_markers=20, sources=None, rnd_iters=2, length=None):
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
        removing_performance_dir = os.path.join(directory, target_domain+ '_' + str(
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
                                                      plot_points, train_data_size, random_score,
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
                        np.argsort(vals_sources[0])[::-1]), plot_points, target_domain, train_data_size,
                        random_score, n_points, sources=sources, val_='rnd', itr=itr,
                        max_itr=rnd_iters))

            else:

                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Removal LOO/Shapley: '):
                    print("=" * 40, flush=True)
                    print(f"Calulating performace if we remove points in descending {val_}", flush=True)
                    print("=" * 40, flush=True)

                    perfs.append(_portion_performance(np.argsort(vals_source)[::-1][:length], \
                                                      plot_points, target_domain, train_data_size, random_score,
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
                        target_domain, train_data_size, random_score, n_points, sources=sources, val_='rnd', itr=itr,
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


    def _portion_performance(idxs, plot_points, target_domain, train_data_size, random_score, n_points, sources=None,
                             val_='TMC_SHAPLEY', itr=None, max_itr=None):
        """Given a set of indexes, starts removing points from
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(n_points)}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        scores = []
        for i in trange(len(plot_points), 0, -1,
                        desc='Inside ' + val_ + ' _portion_performance: removing sources in descending'):
            data_combination = sorted(idxs[plot_points[i - 1]:])
            data_combination_name = '_'.join([ALL_POS_DOMAINS[id] for id in data_combination])
            if data_combination_name in full_performance_dict:
                new_score = full_performance_dict[data_combination_name]
            else:
                new_score = train_model(data_combination, target_domain, train_data_size)
                full_performance_dict.update({data_combination_name: new_score})
                print("After _portion_performance small_performance_dict: ", small_performance_dict,
                      'full_performance_dict:',
                      full_performance_dict, flush=True)

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
        plt.xlabel('\t \t'.join([ALL_POS_DOMAINS[i] for i in sources.keys()]))
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
            max(len(sources.keys()) - 1, num_plot_markers),
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


    def performance_plots_adding(directory, vals, target_domain, train_data_size,
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

        adding_performance_dir = os.path.join(directory, target_domain + '_' + str(
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
                                                               plot_points, target_domain, train_data_size,
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
                        target_domain, train_data_size, random_score, n_points, sources=sources, val_='rnd', itr=itr,
                        max_itr=rnd_iters))

            else:
                perfs = []
                val_ = "TMC_Shapley"
                for vals_source in tqdm(vals_sources, desc='Ploting Adding LOO/Shapley: '):
                    print("=" * 60, flush=True)
                    print(f"Calulating performace if we add points in descending {val_}", flush=True)
                    print("=" * 60, flush=True)

                    perfs.append(_portion_performance_addition(np.argsort(vals_source)[::-1][:length], \
                                                               plot_points, target_domain, train_data_size,
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
                        target_domain, train_data_size, random_score, n_points, sources=sources, val_='rnd', itr=itr,
                        max_itr=rnd_iters))

            rnd = np.mean(rnd, 0)

            data_dict = {'perfs': perfs, 'rnd': rnd, 'rnd_prmtn':rnd_prmtn}
            pkl.dump(data_dict, open(adding_performance_dir, 'wb'))

        plt.plot(plot_points / n_points * 100, perfs[0] * 100,
                 '-', lw=5, ms=10, color='b')
        if len(vals) == 3:
            plt.plot(plot_points / n_points * 100, perfs[1] * 100,
                     '--', lw=5, ms=10, color='orange')
            legends = ['TMC-Shapley ', 'G-Shapley ', 'LOO', 'Random']
        elif len(vals) == 2:
            # legends = ['TMC-Shapley ', 'LOO', 'Random']
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


    def _portion_performance_addition(idxs, plot_points, target_domain, train_data_size, random_score, n_points,
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
        for i in trange(1, len(plot_points) + 1, 1,
                        desc='Inside ' + val_ + ' _portion_performance_addition: adding sources in descending'):

            if i == len(plot_points):
                data_combination = sorted(idxs)
            else:
                data_combination = sorted(idxs[:plot_points[i]])

            data_combination_name = '_'.join([ALL_POS_ALL_CORPUS[id] for id in data_combination])
            if data_combination_name in full_performance_dict:
                new_score = full_performance_dict[data_combination_name]
            else:
                new_score = train_model(data_combination, target_domain, train_data_size)
                full_performance_dict.update({data_combination_name: new_score})
                print("After _portion_performance_addition: small_performance_dict: ", small_performance_dict,
                      'full_performance_dict:',
                      full_performance_dict, flush=True)

            scores.append(new_score)

        return np.array(scores)


    def _tmc_shap( mem_tmc, idxs_tmc, random_score,
                  mean_score, n_points, iterations, iter_counter, max_iter, tolerance=None, sources=None,
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
            for n, idx in enumerate(tqdm(idxs, desc='Inside one iter/save_every idxs')):
                old_score = new_score
                Selected_IDS += list(sources[idx])  # ids to keep

                data_combination = sorted(idxs[:n + 1])
                data_combination_name = '_'.join([ALL_POS_DOMAINS[id] for id in data_combination])
                print('-' * 150, flush=True)
                print(" - Source Domains: ", data_combination_name, flush=True)
                print('-' * 150, flush=True)
                if data_combination_name in small_performance_dict:
                    new_score = full_performance_dict[data_combination_name]
                else:
                    new_score = train_model(data_combination, target_domain, train_data_size)
                    small_performance_dict.update({data_combination_name: new_score})
                    print("After tmc round: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
                          full_performance_dict, flush=True)

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

    print("Begining of run: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
          full_performance_dict, flush=True)

    try:
        baseline_value
    except:
        print('-' * 150, flush=True)
        print(" - Running Baseline with source sources: ", sources, 'ALL_POS_DOMAINS: ', ALL_POS_DOMAINS, flush=True)
        print('-' * 150, flush=True)
        baseline_value, random_score, tol, mean_score = get_baseline_results(sources, target_domain=target_domain,train_data_size=train_data_size)

    if sources is None:
        sources = {i: np.array([i]) for i in range(n_points)}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    n_sources = len(sources)

    mem_tmc = np.zeros((0,
                        n_points))  # (n_iter x n_points) #n_iter is basically n_(save_every) which can be a high value and it's not epoch
    idxs_tmc = np.zeros((0, n_sources), int)
    if g_run:
        mem_g = np.zeros((0, n_points))
        idxs_g = np.zeros((0, n_sources), int)
    else:
        mem_g = None
        idxs_g = None


    data_combination = sorted(list(sources.keys()))
    data_combination_name = '_'.join([ALL_POS_DOMAINS[id] for id in data_combination])
    small_performance_dict[data_combination_name] = baseline_value

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
            vals_loo = _calculate_loo_vals(sources, baseline_value, n_points, target_domain, train_data_size)

            print('LOO values are being saved!', flush=True)
            save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc, g_number, mem_g, idxs_g, sources,
                         overwrite=True,
                         n_points=n_points, tol=tol, \
                         baseline_value=baseline_value, random_score=random_score, mean_score=mean_score)
        print('LOO values calculated!', flush=True)

    if not tolerance: tolerance = tol


    def run_routine(tmc_run, mem_tmc, idxs_tmc, random_score, mean_score, n_points, save_every, seed, max_iter,
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
                    mem_tmc, idxs_tmc = _tmc_shap(mem_tmc, idxs_tmc, random_score, mean_score, \
                                                      n_points, save_every, iter_counter, max_iter, tolerance,
                                                      sources)

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


    if not load_shapley: mem_tmc, idxs_tmc, vals_tmc = run_routine(tmc_run, mem_tmc, idxs_tmc, random_score,
                                                                   mean_score, n_points, save_every, seed, max_iter,
                                                                   tolerance, sources)

    mem_tmc, idxs_tmc, vals_tmc, mem_g, idxs_g, vals_g = merge_results(directory, n_points, sources)

    vals_tmc = np.mean(np.concatenate([mem_tmc, vals_loo.reshape(1, -1)]), 0)

    shapley_value_plots(directory, [vals_tmc, vals_loo], n_points, num_plot_markers=20, sources=sources,
                        name=target_domain + '_' + str(n_points))

    perfs = None

    print("=" * 50, f'\nDone Shapely Values {vals_tmc}, Loo vals {vals_loo} Perfs {perfs}', "\n", "=" * 50, flush=True)
    del baseline_value


