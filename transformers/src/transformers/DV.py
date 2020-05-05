import os
import sys
import time
import numpy as np
from .DShap import DShap
import matplotlib.pyplot as plt
import sklearn
from .shap_utils import *
MEM_DIR = './'
import pdb

## Classifier

problem, model = 'classification', 'logistic'
hidden_units = [] # Empty list in the case of logistic regression.
train_size = 100


## Create Synthetic Dataset

d, difficulty = 50, 1
num_classes = 2
tol = 0.03
target_accuracy = 0.8
important_dims = 5
clf = return_model(model, solver='liblinear', hidden_units=tuple(hidden_units))
_param = 1.0
for i in range(100):
    X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov = np.eye(d),
                                          size=train_size + 5000) #(5100, 50)
    _, y_raw, _, _ = label_generator(
        problem, X_raw, param = _param,  difficulty = difficulty, important=important_dims) #(5100,)
    clf.fit(X_raw[:train_size], y_raw[:train_size])
    test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])
    if test_acc > target_accuracy:
        break
    _param *= 1.1
    print(i)
print('Performance using the whole training set = {0:.2f}'.format(test_acc))


X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]
model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test,
              sources=None,
              sample_weight=None,
              model_family=model,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)
dshap.run(100, 0.1, g_run=False)

# dshap = DShap(X, y, X_test, y_test, num_test,
#               sources=None,
#               sample_weight=None,
#               model_family=model,
#               metric='accuracy',
#               overwrite=True,
#               directory=directory, seed=1)
# dshap.run(100, 0.1, g_run=False)
# #
# dshap = DShap(X, y, X_test, y_test, num_test,
#               sources=None,
#               sample_weight=None,
#               model_family=model,
#               metric='accuracy',
#               overwrite=True,
#               directory=directory, seed=2)
# dshap.run(100, 0.1, g_run=False)




#
# print("######## Merging Results ###########")
# dshap.merge_results()
#
# # pdb.set_trace()
# print("######## Converging Plots Results ###########")
#%%

# convergence_plots(dshap.marginals_tmc, type='tmc', directory=directory+'/plots')
#%%


# convergence_plots(dshap.marginals_g)



# Now let's see the effect of removing high valuen points

# print("######## Performance Plots Results ###########")
# dshap.performance_plots([dshap.vals_tmc, dshap.vals_loo], num_plot_markers=20,
#                        sources=dshap.sources, name='synthetic')
#
# dshap.performance_plots_adding([dshap.vals_tmc, dshap.vals_loo], num_plot_markers=20,
#                        sources=dshap.sources, name='synthetic')
#
# dshap.shapley_value_plots([dshap.vals_tmc, dshap.vals_loo], num_plot_markers='max',
#                        sources=dshap.sources, name='synthetic')


