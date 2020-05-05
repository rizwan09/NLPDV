import os, pdb

# ______________________________________NLPDV____________________________________
# _______________________________________________________________________

from flair.data import Corpus
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from flair.datasets import ColumnCorpus
# initialize sequence tagger
from flair.models import SequenceTagger
# initialize trainer
from flair.trainers import ModelTrainer
from flair.data import MultiCorpus
# _______________________________________________________________________
# ______________________________________NLPDV____________________________________
# CUDA gpus
CUDA_VISIBLE_DEVICES = [2] #,3, 4, 5, 6, 7]
ALL_POS_DOMAINS = 'emails_newsgroups_wsj_answers_reviews_weblogs'.split('_')
data_dir = '/home/rizwan/NLPDV/SANCL/POS/gweb_sancl/parse/'







target_domain = 'answers'
eval_data_size = 500
train_data_size = 500

# define columns
columns = {1: 'text', 4: 'pos'}

# this is the folder in which train, test and dev files reside
def trim_corpus_dev(corpus, eval_data_size):
    corpus.dev.sentences = corpus.dev.sentences[
                           corpus.dev.total_sentence_count // 2: corpus.dev.total_sentence_count // 2 + eval_data_size]
    corpus.dev.total_sentence_count = len(corpus.dev.sentences)

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_dir+domain, columns, \
                              train_file='train.conll', \
                              dev_file='dev.conll', \
                              test_file='test.conll')#.downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
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


trainer: ModelTrainer = ModelTrainer(tagger, corpus)









multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])

result = trainer.train('resources/taggers/example-pos',
              train_with_dev=True,
              embeddings_storage_mode='cpu',
              mini_batch_size=256,
              max_epochs = max_epochs)

