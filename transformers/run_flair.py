from flair.data import Corpus
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import MultiCorpus
import torch
import pdb
import os, numpy as np, random
from pathlib import Path

train_data_size = None
num_train_epochs = 150
seed = 43


data_dir = '/home/rizwan/NLPDV/SANCL/POS/gweb_sancl/parse/'
columns = {1: 'text', 3: 'pos'}
tag_type = 'pos'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ALL_POS_DOMAINS = 'wsj_emails_newsgroups_answers_reviews_weblogs'.split('_')
ALL_POS_ALL_CORPUS = {}

# target_domain = 'emails'

for domain in ALL_POS_DOMAINS:
    corpus: Corpus = ColumnCorpus(data_dir + domain, columns, \
                                  dev_file='dev.conll',
                                  train_file='train.conll',
                                  test_file='test.conll')
    ALL_POS_ALL_CORPUS.update({domain: corpus})
    print('-'*150, flush=True)
    print(" - Domain: ", domain, flush=True)
    print(" - Corpus Train Size: ", len(corpus.train), flush=True)
    print(" - Corpus Dev Size: ", len(corpus.dev), flush=True)
    print(" - Corpus Test Size: ", len(corpus.test), flush=True)
    print('-' * 150, flush=True)


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
print('-'*50, '\nTag_dictionary size: ', len(tag_dictionary), '\n','-'*50, flush=True)
# initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

for target_domain in ALL_POS_DOMAINS:

    train_output_dir = 'resources/taggers/example-pos/'+target_domain


    def get_baseline_results(sources, target_domain, train_data_size=None):
        source_corpus = ALL_POS_ALL_CORPUS[target_domain]
        source_corpus._train = []
        for domain in sources:
            temp_data = ALL_POS_ALL_CORPUS[domain]._train
            if train_data_size:
                if len(ALL_POS_ALL_CORPUS[domain].train)>train_data_size:
                    temp_data = random.choices(ALL_POS_ALL_CORPUS[domain]._train, k=train_data_size)
            source_corpus._train += temp_data

        dist = source_corpus.get_tag_distribution(corpus_name='dev')
        random_score = max(dist.values())/ sum(dist.values())

        trainer: ModelTrainer = ModelTrainer(tagger, source_corpus, use_tensorboard=True)
        result_dict = trainer.train(train_output_dir, max_epochs = num_train_epochs)
        return result_dict


    sources = ALL_POS_DOMAINS.copy()
    sources.remove(target_domain)
    result = None
    result = get_baseline_results(sources, target_domain=target_domain, train_data_size=train_data_size)

    print('Target: ', target_domain, "baseline result trained with full source train datasets: ", result, flush=True)
    print('ALL_POS_DOMAINS: ', ALL_POS_DOMAINS)


# result = trainer.final_test('resources/taggers/example-pos', eval_mini_batch_size=256)