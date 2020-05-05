
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
target_domains = ['emails']

loo_run = True
tmc_run = True
g_run = False
load_removing_performance_plot = load_adding_performance_plot = True

overwrite_directory = False  # True when just load shapley's when to plot only
load_shapley = False

train_data_size = 2000 # For full_training set None
num_train_epochs = 150
seed = 43
err = 0.1
tolerance = 0.05
max_iter = 30
save_every = 5

data_dir = '/home/rizwan/NLPDV/SANCL/POS/gweb_sancl/parse/'
columns = {1: 'text', 3: 'pos'}
tag_type = 'pos'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


for domain in ALL_POS_DOMAINS :
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



def make_empty_corpus(corpus, empty_train=True, empty_dev=False, empty_test=True ):
    if empty_train: corpus._train = []
    if empty_dev: corpus._dev = []
    if empty_test: corpus._test = []

print('-'*50, flush=True)
print("- Creating empty corpus for target_domain: ", target_domains[0], flush=True)

source_target_corpus: Corpus = ColumnCorpus(data_dir + target_domains[0], columns, \
                                  dev_file='dev.conll',
                                  train_file='dev.conll',
                                  test_file='dev.conll')
print('- Created' * 50, flush=True)
print('-' * 50, flush=True)


for target_domain in target_domains:
    print('-'*50, flush=True)
    print("Target name: ", target_domain, flush=True)
    print('-' * 50, flush=True)


    multi_all_corpus = MultiCorpus(ALL_POS_ALL_CORPUS.values())
    # 3. make the tag dictionary from the all corpora
    tag_dictionary = multi_all_corpus.make_tag_dictionary(tag_type=tag_type)


    train_output_dir = 'resources/taggers/example-pos/'+target_domain+'_shapley'
    if not train_data_size: train_output_dir+='_full_train'

    def train_model(data_combination_name, target_domain, train_data_size=train_data_size,
                    small_performance_dict=small_performance_dict, eval_on_dev=True, \
                    ALL_POS_DOMAINS=ALL_POS_DOMAINS, is_baeline_run = False, \
                    source_target_corpus=source_target_corpus, ALL_POS_ALL_CORPUS=ALL_POS_ALL_CORPUS):
        # make empty the temp corpus
        make_empty_corpus(source_target_corpus)

        # prepare target data
        source_target_corpus._dev = ALL_POS_ALL_CORPUS[target_domain]._dev
        if eval_on_dev:
            source_target_corpus._test = ALL_POS_ALL_CORPUS[target_domain]._dev
        else:
            source_target_corpus._test = ALL_POS_ALL_CORPUS[target_domain]._test

        print('-' * 50, flush=True)
        print(' - Trianing for Target: ', target_domain, flush=True)
        print(' - Initial Corpus Train size: ', len(source_target_corpus.train), flush=True)
        print(' - Initial Corpus Dev size: ', len(source_target_corpus.dev), flush=True)
        print(' - Initial Corpus Test size: ', len(source_target_corpus.test), flush=True)
        print('-' * 50, flush=True)

        # prepare source data
        for domain in data_combination_name.split('_'):
            if train_data_size:
                temp_data = list(np.random.choice(ALL_POS_ALL_CORPUS[domain]._train, train_data_size, replace=False))
            else:
                temp_data = [item for item in ALL_POS_ALL_CORPUS[domain]._train]
            source_target_corpus._train += temp_data
            print('-' * 50, flush=True)
            print(' - Adding domian into training data: ', domain, flush=True)
            print(' - Adding Train size: ', len(temp_data), flush=True)
            print(' - After Adding Corpus Train Size: ', len(source_target_corpus.train), flush=True)
            print(' - After Adding Corpus Dev size: ', len(source_target_corpus.dev), flush=True)
            print(' - After Adding Corpus Test size: ', len(source_target_corpus.test), flush=True)
            print('-' * 50, flush=True)
        print('-' * 50, flush=True)
        print(' - Training for Target: ', target_domain, flush=True)
        print(' - Final Corpus Train size: ', len(source_target_corpus.train), flush=True)
        print(' - Final Corpus Dev size: ', len(source_target_corpus.dev), flush=True)
        print(' - Final Corpus Test size: ', len(source_target_corpus.test), flush=True)
        print('-' * 50, flush=True)

        # init model
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('extvec'),
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type)
        trainer: ModelTrainer = ModelTrainer(tagger, source_target_corpus, use_tensorboard=True)
        # Train modle and get result
        result_dict = trainer.train(train_output_dir, embeddings_storage_mode='gpu', max_epochs=num_train_epochs, save_final_model=False)
        small_performance_dict[target_domain].update({data_combination_name: result_dict['test_score']})


        if is_baeline_run: return result_dict['test_score'], tagger
        else: result_dict['test_score']

    def get_baseline_results(source_names, target_domain, train_data_size=train_data_size, small_performance_dict=small_performance_dict, num_bags=20, eval_on_dev=True):
        source_target_corpus = ALL_POS_ALL_CORPUS[target_domain]
        baseline_value, tagger = train_model('_'.join(source_names), target_domain, train_data_size, small_performance_dict=small_performance_dict, eval_on_dev=eval_on_dev, is_baeline_run=True)
        scores = []
        for _ in range(num_bags):
            temp_corpus=source_target_corpus
            bag_idxs = np.random.choice(len(source_target_corpus.test), len(source_target_corpus.test))
            temp_corpus._test = [ source_target_corpus._test[i] for i in  bag_idxs]
            trainer: ModelTrainer = ModelTrainer(tagger, temp_corpus, use_tensorboard=True)
            scores.append(trainer.final_test(Path(train_output_dir), eval_mini_batch_size=32))
        tol =  np.std(scores)
        mean_score = np.mean(scores)
        dist = source_target_corpus.get_tag_distribution(corpus_name='dev' if eval_on_dev else 'test') # dev, test are same if eval_on_dev==True
        random_score = max(dist.values()) / sum(dist.values())


        return baseline_value, random_score, tol, mean_score


    # dep copy is needed in case of a multiple target in one run setting
    source_names = ALL_POS_DOMAINS.copy()
    source_names.remove(target_domain)
    sources = np.arange(len(source_names))
    n_points = len(sources)
    directory = train_output_dir

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


    def _calculate_loo_vals(source_names, sources, baseline_value, n_points, target_domain, train_data_size, small_performance_dict=small_performance_dict):
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
            data_combination_name = '_'.join([source_names[id] for id in data_combination])
            print('-' * 150, flush=True)
            print(" - Target ", target_domain, " and Loo for Domain: ", source_names[i], flush=True)
            print('-' * 150, flush=True)
            if data_combination_name not in small_performance_dict[target_domain]:
                removed_value = train_model(data_combination_name, target_domain, train_data_size, small_performance_dict)
            removed_value = small_performance_dict[target_domain][data_combination_name]
            print('baseline_value: ', baseline_value, 'removed value: ', removed_value, 'baseline_value - removed_value: ', baseline_value - removed_value, flush=True)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
            print("After Loo: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
                  full_performance_dict, flush=True)
        return vals_loo


    def save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc, \
                     g_number=None, mem_g=None, idxs_g=None, sources=sources, source_names = source_names, overwrite=False, n_points=None, tol=None, \
                     baseline_value=None, random_score=None, mean_score=None):
        """Saves results computed so far."""
        if directory is None:
            return
        loo_dir = os.path.join(directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            data_dict = {'loo': vals_loo, 'n_points': n_points, 'tol': tol, 'sources': sources, 'source_names': source_names,\
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
            data_combination_name = '_'.join([source_names[id] for id in data_combination])
            if data_combination_name in full_performance_dict:
                new_score = full_performance_dict[data_combination_name]
            else:
                new_score = train_model(data_combination_name, target_domain, None, full_performance_dict)
                full_performance_dict.update({data_combination_name: new_score})
                print("After _portion_performance small_performance_dict: ", small_performance_dict,
                      'full_performance_dict:',
                      full_performance_dict, flush=True)

            scores.append(new_score)

        return np.array(scores)[::-1]


    def shapley_value_plots(directory, vals, n_points, name=None, num_plot_markers=20, source_names=source_names, sources=None):
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
        plt.xlabel('      '.join([source_names[i] for i in sources.keys()]))
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
        scores = []
        for i in trange(1, len(plot_points) + 1, 1,
                        desc='Inside ' + val_ + ' _portion_performance_addition: adding sources in descending'):

            if i == len(plot_points):
                data_combination = sorted(idxs)
            else:
                data_combination = sorted(idxs[:plot_points[i]])

            data_combination_name = '_'.join([source_names[id] for id in data_combination])
            if data_combination_name in full_performance_dict:
                new_score = full_performance_dict[data_combination_name]
            else:
                new_score = train_model(data_combination_name, target_domain, None, full_performance_dict)
                full_performance_dict.update({data_combination_name: new_score})
                print("After _portion_performance_addition: small_performance_dict: ", small_performance_dict,
                      'full_performance_dict:',
                      full_performance_dict, flush=True)

            scores.append(new_score)

        return np.array(scores)


    def _tmc_shap( mem_tmc, idxs_tmc, random_score,
                  mean_score, n_points, iterations, iter_counter, max_iter, tolerance=tolerance, source_names=source_names,
                  sources=sources, small_performance_dict= small_performance_dict, print_step=1):
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
            assert len(sources) ==len(source_names)
            idxs = np.random.permutation(len(source_names))
            marginal_contribs = np.zeros(n_points)
            truncation_counter = 0
            new_score = random_score

            for n, idx in enumerate(tqdm(idxs, desc='Inside one iter/save_every idxs')):
                old_score = new_score

                data_combination = sorted(idxs[:n + 1])
                data_combination_name = '_'.join([source_names[id] for id in data_combination])
                print('-' * 150, flush=True)
                print(" - Source Domains: ", data_combination_name.split('_'), flush=True)
                print('-' * 150, flush=True)
                if data_combination_name not in small_performance_dict[target_domain]:
                    new_score_temp = train_model(data_combination_name, target_domain, train_data_size, small_performance_dict)
                    print("After tmc round: small_performance_dict: ", small_performance_dict, 'full_performance_dict:',
                          full_performance_dict, flush=True)
                new_score = small_performance_dict[target_domain][data_combination_name]
                marginal_contribs[sources[idx]] = (new_score - old_score)
                marginal_contribs[sources[idx]] /= len(sources[idx])
                distance_to_full_score = np.abs(new_score - mean_score)
                if distance_to_full_score <= tolerance * mean_score:
                    truncation_counter += 1
                    if truncation_counter > 5:
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

    # get baseline_value, random_score, n_points = (len train_dataset), sources etc., or load
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


    # Run baseine model on all sources
    try:
        baseline_value
    except:
        print('-' * 150, flush=True)
        print(" - Running Baseline with source sources: ", source_names, 'ALL_POS_DOMAINS: ', ALL_POS_DOMAINS, flush=True)
        print('-' * 150, flush=True)
        baseline_value, random_score, tol, mean_score = \
            get_baseline_results(source_names, target_domain=target_domain,train_data_size=train_data_size, small_performance_dict=small_performance_dict)
        print('Target: ', target_domain, "baseline_value, random_score, tol, \
        mean_score trained with sampled all source train datasets: ", baseline_value, random_score, tol, mean_score,
              flush=True)
    if sources is None:
        sources = {i: np.array([i]) for i in range(n_points)}
    elif not isinstance(sources, dict):
        sources = {i: np.where(sources == i)[0] for i in set(sources)}
    n_sources = len(sources)


    # Must be set after one run
    single_mems = {lang: small_performance_dict[target_domain][lang] for lang in ALL_POS_DOMAINS if
                   lang in small_performance_dict[target_domain]}
    loo_mems = {'_'.join([id for id in source_names if id != lang]): small_performance_dict[target_domain][
        '_'.join([id for id in source_names if id != lang])] for lang in source_names}

    random_score = np.array(list(single_mems.values()) ).mean()

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
    data_combination_name = '_'.join([source_names[id] for id in data_combination])
    small_performance_dict[target_domain][data_combination_name] = baseline_value

    tmc_number, g_number = _which_parallel(directory)
    if not g_run: g_number = None
    if not load_shapley:
        _create_results_placeholder(directory, tmc_number, mem_tmc, idxs_tmc, g_number, mem_g, idxs_g)


    # Compute LOO
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
            vals_loo = _calculate_loo_vals(source_names, sources, baseline_value, n_points, target_domain, train_data_size, small_performance_dict)

            print('LOO values are being saved!', flush=True)
            save_results(directory, vals_loo, tmc_number, mem_tmc, idxs_tmc, g_number, mem_g, idxs_g, sources,
                         source_names=source_names, overwrite=True,\
                         n_points=n_points, tol=tol, \
                         baseline_value=baseline_value, random_score=random_score, mean_score=mean_score)
        print('LOO values calculated!', flush=True)

    if not tolerance: tolerance = tol

    # Compute Shapley main iterations for all permutation
    def run_routine(tmc_run, mem_tmc, idxs_tmc, random_score, mean_score, n_points, save_every, source_names, max_iter,
                    tolerance, small_performance_dict):
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
                                                      n_points, save_every, iter_counter, max_iter, tolerance,\
                                                      source_names=source_names, sources=sources,\
                                                  small_performance_dict=small_performance_dict)

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
                                                                   mean_score, n_points, save_every, source_names, max_iter,
                                                                   tolerance, small_performance_dict)
    # merge results from different runs or using diff seeds
    mem_tmc, idxs_tmc, vals_tmc, mem_g, idxs_g, vals_g = merge_results(directory, n_points, sources)

    #Loo is also a permutation so consider it
    vals_tmc = np.mean(np.concatenate([mem_tmc, vals_loo.reshape(1, -1)]), 0)

    shapley_value_plots(directory, [vals_tmc, vals_loo], n_points, num_plot_markers=20, sources=sources, source_names=source_names,
                        name=target_domain + '_' + str(n_points))

    # Now I am not plotting the results as per adding the domains in descending order so None
    perfs = None

    single_mems = {lang: small_performance_dict[target_domain][lang] for lang in ALL_POS_DOMAINS if lang in small_performance_dict[target_domain]}
    dict = {k: v for k, v in small_performance_dict[target_domain].items()}
    best_single_comb = {k: v for k, v in single_mems.items() if v == max(single_mems.values())}


    print("=" * 50, f'\nDone Shapely Values {vals_tmc}, Loo vals {vals_loo} Perfs {perfs}', "\n", "=" * 50, flush=True)
    pdb.set_trace()

    # del the object to make it ready for next target domain in case of multiuple targets
    del baseline_value


