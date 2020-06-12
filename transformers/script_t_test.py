# Code Apted from: https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py
######################################################################
# Compare two systems using bootstrap resampling                     #
#  * by Graham Neubig                                                #
#  * minor modifications by Mathias Müller                           #
#                                                                    #
# See, e.g. the following paper for references                       #
#                                                                    #
# Statistical Significance Tests for Machine Translation Evaluation  #
# Philipp Koehn                                                      #
# http://www.aclweb.org/anthology/W04-3250                           #
#                                                                    #
######################################################################

import numpy as np

EVAL_TYPE_ACC = "acc"
EVAL_TYPE_BLEU = "bleu"
EVAL_TYPE_BLEU_DETOK = "bleu_detok"
EVAL_TYPE_PEARSON = "pearson"

EVAL_TYPES = [EVAL_TYPE_ACC,
              EVAL_TYPE_BLEU,
              EVAL_TYPE_BLEU_DETOK,
              EVAL_TYPE_PEARSON]


def eval_preproc(data, eval_type='acc'):
    ''' Preprocess into the appropriate format for a particular evaluation type '''
    if type(data) == str:
        data = data.strip()
        if eval_type == EVAL_TYPE_BLEU:
            data = data.split()
        elif eval_type == EVAL_TYPE_PEARSON:
            data = float(data)
    return data


def eval_measure(gold, sys, eval_type='acc'):
    ''' Evaluation measure

    This takes in gold labels and system outputs and evaluates their
    accuracy. It currently supports:
    * Accuracy (acc), percentage of labels that match
    * Pearson's correlation coefficient (pearson)
    * BLEU score (bleu)
    * BLEU_detok, on detokenized references and translations, with internal tokenization
    :param gold: the correct labels
    :param sys: the system outputs
    :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
    '''
    if eval_type == EVAL_TYPE_ACC:
        return sum([1 if g == s else 0 for g, s in zip(gold, sys)]) / float(len(gold))
    elif eval_type == EVAL_TYPE_BLEU:
        import nltk
        gold_wrap = [[x] for x in gold]
        return nltk.translate.bleu_score.corpus_bleu(gold_wrap, sys)
    elif eval_type == EVAL_TYPE_PEARSON:
        return np.corrcoef([gold, sys])[0, 1]
    elif eval_type == EVAL_TYPE_BLEU_DETOK:
        import sacrebleu
        # make sure score is 0-based instead of 100-based
        return sacrebleu.corpus_bleu(sys, [gold]).score / 100.
    else:
        raise NotImplementedError('Unknown eval type in eval_measure: %s' % eval_type)


def eval_with_paired_bootstrap(gold, sys1, sys2,
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='acc', sys1_scores = [], sys2_scores = []):
    ''' Evaluate with paired boostrap
    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the accuracy of the two systems.

    :param gold: The correct labels
    :param sys1: The output of system 1
    :param sys2: The output of system 2
    :param num_samples: The number of bootstrap samples to take
    :param sample_ratio: The ratio of samples to take every time
    :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
    '''
    assert (len(gold) == len(sys1))
    assert (len(gold) == len(sys2))

    # Preprocess the data appropriately for they type of eval
    gold = [eval_preproc(x, eval_type) for x in gold]
    sys1 = [eval_preproc(x, eval_type) for x in sys1]
    sys2 = [eval_preproc(x, eval_type) for x in sys2]


    wins = [0, 0, 0]
    n = len(gold)
    ids = list(range(n))

    use_given_result = False
    if len(sys1_scores)> 1: use_given_result = True

    for i in range(num_samples):
        # Subsample the gold and system outputs
        np.random.shuffle(ids)
        reduced_ids = ids[:int(len(ids) * sample_ratio)]
        reduced_gold = [gold[i] for i in reduced_ids]
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]
        # Calculate accuracy on the reduced sample and save stats
        if not use_given_result:
            sys1_score = eval_measure(reduced_gold, reduced_sys1, eval_type=eval_type)
            sys2_score = eval_measure(reduced_gold, reduced_sys2, eval_type=eval_type)
        else:
            sys1_score = sys1_scores[i]
            sys2_score = sys2_scores[i]

        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        if not use_given_result:
            sys1_scores.append(sys1_score)
            sys2_scores.append(sys2_score)

    # Print win stats
    wins = [x / float(num_samples) for x in wins]
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(sys1 is superior with p value p=%.3f)\n' % (1 - wins[0]))
    elif wins[1] > wins[0]:
        print('(sys2 is superior with p value p=%.3f)\n' % (1 - wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    print('sys1 mean=%.4f, median=%.4f, 95%% confidence interval=[%.4f, %.4f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)],
           sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.4f, median=%.4f, 95%% confidence interval=[%.4f, %.4f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)],
           sys2_scores[int(num_samples * 0.975)]))


if __name__ == "__main__":
    '''
    # For wsj
    baseline = [0.958, 0.9551, 0.9531, 0.9521, 0.9549, 0.9534, 0.9541, 0.9546, 0.9558, 0.9574, 0.9561, 0.955, 0.9534,
                0.9546, 0.9538, 0.9551, 0.9564, 0.9565, 0.956, 0.9556]
    shapley = [0.9613, 0.9594, 0.958, 0.9575, 0.9604, 0.9597, 0.9594, 0.9597, 0.9616, 0.9625, 0.9602, 0.9591, 0.9578,
               0.9589, 0.958, 0.9595, 0.9612, 0.9618, 0.9624, 0.9617]
    print(" wsj: ")
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)

    # For emails
    shapley = [0.9522, 0.9509, 0.9491, 0.9455, 0.944, 0.9505, 0.9478, 0.9498, 0.951, 0.9488, 0.9477, 0.9469, 0.9488, 0.9532, 0.9497, 0.956, 0.9584, 0.9633, 0.9614, 0.9652]
    baseline = [0.955, 0.9511, 0.9531, 0.9568, 0.9573, 0.9531, 0.947, 0.9487, 0.9463, 0.9462, 0.9484, 0.9528, 0.9522,
                0.9526, 0.9454, 0.9521, 0.9467, 0.9482, 0.9427, 0.9415]

    print(" emails: ")
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)
    # For Newsgroup
    baseline = [0.9695, 0.9709, 0.9729, 0.9717, 0.9729, 0.9713, 0.9673, 0.9687, 0.9686, 0.9672, 0.9665, 0.9686, 0.9703,
                0.968, 0.9685, 0.9659, 0.9645, 0.9685, 0.9659, 0.9661]
    shapley = [0.9694, 0.9701, 0.9719, 0.9703, 0.9727, 0.9705, 0.9677, 0.9686, 0.9681, 0.9667, 0.9665, 0.968, 0.9698,
               0.9688, 0.9691, 0.9671, 0.9665, 0.9701, 0.968, 0.9681]

    print(" newsgroups: ")
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)
    # For Anwers
    shapley = [0.95, 0.95, 0.9506, 0.9512, 0.9517, 0.9505, 0.9498, 0.9515, 0.9523, 0.9523, 0.9559, 0.9522, 0.9522, 0.9516, 0.9515, 0.9532, 0.9524, 0.949, 0.9487, 0.9508]
    shapley = [0.9506, 0.9542, 0.9545, 0.9554, 0.9582, 0.9592, 0.9603, 0.9635, 0.9647, 0.9644, 0.9681, 0.9694, 0.97, 0.9687, 0.9704, 0.9712, 0.9694, 0.9667, 0.9652, 0.9677, 0.9695, 0.9714, 0.9716, 0.9703, 0.9679, 0.9679, 0.9668, 0.9665, 0.9626, 0.9626, 0.9646, 0.9634, 0.9627, 0.9625, 0.965, 0.9657, 0.9642, 0.9648, 0.9654, 0.9655, 0.9659, 0.9663, 0.9677, 0.9666, 0.9667, 0.964, 0.9652, 0.9649, 0.9676, 0.9669]
    baseline = [0.9508, 0.9506, 0.9476, 0.9483, 0.9473, 0.9492, 0.9478, 0.9462, 0.9431, 0.9454, 0.943, 0.9439, 0.944,
                0.9455, 0.9439, 0.9421, 0.9408, 0.9412, 0.9406, 0.9382]
    baseline = [0.9485, 0.9511, 0.9508, 0.9514, 0.954, 0.9536, 0.9551, 0.9569, 0.9566, 0.9574, 0.9597, 0.9608, 0.9612, 0.9598, 0.9595, 0.9608, 0.9614, 0.9591, 0.9579, 0.9588, 0.9619, 0.961, 0.9629, 0.9625, 0.9589, 0.9592, 0.9582, 0.9578, 0.9547, 0.9559, 0.9571, 0.9557, 0.9527, 0.9527, 0.9528, 0.9517, 0.9507, 0.9526, 0.9524, 0.9534, 0.953, 0.9534, 0.9547, 0.9567, 0.9565, 0.9552, 0.9574, 0.9568, 0.959, 0.9596]
    print(" answers: ")
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)

    print(" Review ")
    # baseline = [0.9599, 0.9606, 0.9622, 0.9628, 0.9618, 0.9597, 0.9587, 0.9572, 0.9606, 0.9602, 0.9602, 0.9598, 0.958, 0.9615, 0.9626, 0.961, 0.9641, 0.9659, 0.9671, 0.969, 0.969, 0.967, 0.9659, 0.9631, 0.9644, 0.9655, 0.9641, 0.9598, 0.9596, 0.9599, 0.9596, 0.9574, 0.9613, 0.962, 0.9625, 0.9648, 0.9665, 0.9675, 0.9679, 0.9696, 0.9687, 0.9675, 0.9686, 0.9672, 0.9671, 0.967, 0.9667, 0.9693, 0.9687, 0.9713]
    baseline = [0.9589, 0.9595, 0.9617, 0.9615, 0.9612, 0.9586, 0.9564, 0.9566, 0.9591, 0.9579, 0.9587, 0.9581, 0.9571, 0.9601, 0.961, 0.9593, 0.9619, 0.9629, 0.9651, 0.9667, 0.9679, 0.9655, 0.9657, 0.9635, 0.9649, 0.9664, 0.9646, 0.9602, 0.9599, 0.9602, 0.9597, 0.9579, 0.9613, 0.962, 0.9621, 0.9647, 0.9664, 0.9674, 0.9681, 0.9701, 0.9694, 0.9681, 0.9694, 0.9681, 0.9684, 0.9686, 0.9681, 0.9703, 0.9695, 0.972]
    shapley = [0.9588, 0.9593, 0.9625, 0.9626, 0.9608, 0.958, 0.9572, 0.9578, 0.9599, 0.9584, 0.959, 0.9585, 0.9574, 0.9605,
     0.9617, 0.9605, 0.9627, 0.9641, 0.9659, 0.9678, 0.9687, 0.967
        , 0.967, 0.9646, 0.9656, 0.9672, 0.9663, 0.9614, 0.9608, 0.9605, 0.9597, 0.9569, 0.961, 0.9618, 0.9612, 0.9626,
     0.964, 0.9639, 0.9644, 0.9665, 0.9663, 0.9652, 0.967, 0.9654, 0.9653,
     0.9655, 0.9655, 0.9667, 0.9642, 0.9677]

    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)
                               
                               
                               
    
    #Few shot:
    print(" Review")
    shapley = [0.967, 0.9677, 0.9688, 0.9683, 0.9666, 0.9649, 0.9667, 0.9656, 0.966, 0.9666, 0.9671, 0.9673, 0.9697, 0.9719, 0.9718, 0.9725, 0.9745, 0.9741, 0.9754, 0.9752, 0.973, 0.9726, 0.9692, 0.9668, 0.9668, 0.9648, 0.9666, 0.9673, 0.9662, 0.9655, 0.9624, 0.9631, 0.9659, 0.9637, 0.9648, 0.9664, 0.966, 0.9676, 0.9699, 0.9648, 0.9624, 0.9606, 0.9616, 0.9641, 0.9649, 0.9663, 0.9676, 0.966, 0.968, 0.9643]
    baseline = [0.9637, 0.9632, 0.9657, 0.965, 0.964, 0.9622, 0.9599, 0.9606, 0.9628, 0.9613, 0.9609, 0.9595, 0.959, 0.9623, 0.9637, 0.9623, 0.9651, 0.966, 0.9669, 0.9684, 0.9695, 0.9684, 0.9678, 0.9653, 0.9663, 0.9679, 0.9659, 0.9605, 0.9606, 0.9614, 0.9602, 0.9573, 0.9622, 0.9627, 0.9628, 0.9641, 0.9663, 0.9668, 0.9671, 0.97, 0.9692, 0.9681, 0.9708, 0.9689, 0.9687, 0.9684, 0.9693, 0.9712, 0.9702, 0.9733]


    print(len(shapley), shapley)
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)



    print("Answers")
    shapley = [0.9526, 0.9524, 0.9535, 0.9555, 0.9545, 0.955, 0.9551, 0.9554, 0.9539, 0.9535, 0.954, 0.956, 0.9548, 0.9521, 0.9497, 0.9515, 0.9525, 0.9549, 0.9555, 0.9553, 0.9555, 0.9553, 0.9547, 0.9548, 0.9554, 0.9552, 0.9559, 0.9554, 0.9575, 0.9562, 0.9569, 0.958, 0.9598, 0.9603, 0.9594, 0.959, 0.959, 0.9569, 0.9564, 0.9577, 0.9576, 0.9582, 0.9599, 0.9585, 0.9585, 0.9583, 0.959, 0.959, 0.9592, 0.9599]
    baseline = [0.9556, 0.956, 0.9567, 0.9563, 0.9605, 0.9584, 0.9587, 0.9555, 0.9488, 0.949, 0.9488, 0.9466, 0.9478, 0.9487, 0.9528, 0.956, 0.9572, 0.955, 0.9559, 0.9572, 0.9582, 0.9573,
 0.957, 0.9581, 0.9569, 0.9567, 0.9556, 0.9517, 0.9546, 0.9531, 0.9571, 0.956, 0.9566, 0.9564, 0.9565, 0.9559, 0.9589, 0.9591, 0.9587, 0.9568, 0.9571, 0.9578, 0.9596, 0.9592, 0.9602
, 0.9614, 0.9615, 0.9602, 0.9604, 0.9609]
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)
    
    print("news")
    shapley = [0.9702, 0.9704, 0.9694, 0.9728, 0.9646, 0.9636, 0.9637, 0.9598, 0.9611, 0.9564, 0.9559, 0.9533, 0.9531, 0.952, 0.9493, 0.9505, 0.9568, 0.952, 0.9518, 0.9528, 0.9559, 0.9583, 0.9605, 0.9646, 0.9681, 0.9739, 0.9827, 0.9842, 0.9838, 0.9837, 0.9874, 0.9887, 0.9872, 0.9862, 0.9844, 0.9872, 0.9881, 0.9912, 0.9909, 0.9921, 0.9922, 0.992, 0.9916, 0.9901, 0.9889, 0.9875, 0.9871, 0.9867, 0.9857, 0.9864]
    baseline = [0.97, 0.9679, 0.9671, 0.9698, 0.9636, 0.9636, 0.9639, 0.9602, 0.9611, 0.9578, 0.9574, 0.9556, 0.9556, 0.955, 0.9518, 0.9531, 0.9583, 0.9526, 0.9517, 0.9523, 0.9567, 0.9588, 0.9608, 0.9647, 0.9687, 0.9738, 0.9839, 0.9849, 0.9842, 0.9843, 0.9879, 0.9889, 0.9877, 0.9868, 0.9857, 0.9892, 0.9889, 0.9915, 0.9914, 0.9919, 0.9922, 0.9911, 0.9918, 0.9908, 0.9902, 0.988, 0.9869, 0.9861, 0.9851, 0.9852]
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                                                          sys1_scores=baseline, sys2_scores=shapley)
    
    print("emails")
    shapley = [0.9718, 0.9738, 0.9732, 0.9753, 0.9758, 0.9753, 0.975, 0.9781, 0.9792, 0.9793, 0.9769, 0.9773, 0.9766, 0.9755, 0.9762, 0.9779, 0.9798, 0.9793, 0.9788, 0.98, 0.9806, 0.9801, 0.9805, 0.9824, 0.9837, 0.9819, 0.9828, 0.9828, 0.9846, 0.9855, 0.9873, 0.9877, 0.9859, 0.9843, 0.9851, 0.9835, 0.9837, 0.9826, 0.9822, 0.9837, 0.9854, 0.9867, 0.9869, 0.9878, 0.9881, 0.9867, 0.9871, 0.9863, 0.9866, 0.9866]
    baseline = [0.9718, 0.9732, 0.9722, 0.9735, 0.973, 0.9705, 0.9704, 0.9738, 0.9754, 0.976, 0.9747, 0.9756, 0.9734, 0.9723, 0.9726, 0.9732, 0.9749, 0.9739, 0.9729, 0.9743, 0.9751, 0.9751, 0.9756, 0.9762, 0.9771, 0.9756, 0.9754, 0.9753, 0.9764, 0.977, 0.9786, 0.9797, 0.9773, 0.9753, 0.9746, 0.9716, 0.9726, 0.9714, 0.9709, 0.9726, 0.9724, 0.9743, 0.9742, 0.9751, 0.9741, 0.9732, 0.975, 0.9744, 0.975, 0.9746]
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                                                          sys1_scores=baseline, sys2_scores=shapley)
    
    print("wsj")
    shapley =[0.97, 0.9714, 0.9756, 0.9761, 0.9763, 0.9764, 0.9758, 0.9761, 0.9763, 0.9754, 0.9743, 0.9771, 0.9769, 0.9767, 0.9768, 0.9764, 0.9758, 0.9758, 0.9781, 0.9781, 0.978, 0.9757, 0.9747, 0.9769, 0.9774, 0.9762, 0.9779, 0.9785, 0.978, 0.9773, 0.9762, 0.9766, 0.9769, 0.9779, 0.9785, 0.9798, 0.9804, 0.9805, 0.9798, 0.9795, 0.9794, 0.981, 0.9795, 0.9795, 0.9799, 0.9806, 0.9812, 0.982, 0.9808, 0.9813]
    baseline = [0.9712, 0.9725, 0.9756, 0.9765, 0.9757, 0.9769, 0.9771, 0.9775, 0.9776, 0.9774, 0.9757, 0.9778, 0.9772, 0.9769, 0.9764, 0.9757, 0.9745, 0.9737, 0.9765, 0.9752, 0.9753, 0.9726, 0.9719, 0.9745, 0.9736, 0.973, 0.9747, 0.9749, 0.975, 0.9744, 0.9726, 0.973, 0.9738, 0.9749, 0.9748, 0.9756, 0.9765, 0.9762, 0.9753, 0.9748, 0.9743, 0.9763, 0.9742, 0.9746, 0.9757, 0.9768, 0.9773, 0.9782, 0.9772, 0.9778]
    eval_with_paired_bootstrap(baseline, baseline, shapley, eval_type=EVAL_TYPE_ACC, num_samples=len(shapley), \
                               sys1_scores=baseline, sys2_scores=shapley)

    '''

    # execute only if run as a script
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('gold', help='File of the correct answers')
    parser.add_argument('sys1', help='File of the answers for system 1')
    parser.add_argument('sys2', help='File of the answers for system 2')
    parser.add_argument('--eval_type', help='The evaluation type (acc/pearson/bleu/bleu_detok)', type=str,
                        default='acc', choices=EVAL_TYPES)
    parser.add_argument('--num_samples', help='Number of samples to use', type=int, default=10000)
    args = parser.parse_args()

    with open(args.gold, 'r') as f:
        gold = f.readlines()
    with open(args.sys1, 'r') as f:
        sys1 = f.readlines()
    with open(args.sys2, 'r') as f:
        sys2 = f.readlines()
    eval_with_paired_bootstrap(gold, sys1, sys2, eval_type=args.eval_type, num_samples=args.num_samples)

'''

 

python script_t_test.py  temp/UD_ARABIC_output_Shapley_16_3e-05/best/test_gold.txt   temp/UD_ARABIC_output_Shapley_16_3e-05/best/test_predictions.txt temp/UD_ARABIC_output_baseline_16_5e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_BASQUE_output_Shapley_32_3e-05/best/test_gold.txt   temp/UD_BASQUE_output_Shapley_32_3e-05/best/test_predictions.txt temp/UD_BASQUE_output_baseline-s_16_2e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_BULGARIAN_output_Shapley_32_5e-05/best/test_gold.txt   temp/UD_BULGARIAN_output_Shapley_32_5e-05/best/test_predictions.txt temp/UD_BULGARIAN_output_baseline-s_16_3e-05/best/test_predictions.txt 
python script_t_test.py  temp/UD_BULGARIAN_output_Shapley_32_5e-05/best/test_gold.txt   temp/UD_BULGARIAN_output_Shapley_32_5e-05/best/test_predictions.txt temp/UD_BULGARIAN_output_baseline_32_3e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_CATALAN_output_Shapley_16_5e-05/best/test_gold.txt   temp/UD_CATALAN_output_Shapley_16_5e-05/best/test_predictions.txt temp/UD_CATALAN_output_baseline-32_5e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_CROATIAN_output_Shapley_16_5e-05/best/test_gold.txt   temp/UD_CROATIAN_output_Shapley_16_5e-05/best/test_predictions.txt temp/UD_CROATIAN_output_baseline-s_16_3e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_CHINESE_output_Shapley_16_5e-05/best/test_gold.txt   temp/UD_CHINESE_output_Shapley_16_5e-05/best/test_predictions.txt temp/UD_CHINESE_output_baseline-s_16_3e-05/best/test_predictions.txt 

python script_t_test.py  temp/UD_CZECH_output_Shapley_32_3e-05/best/test_gold.txt   temp/UD_CZECH_output_Shapley_32_3e-05/best/test_predictions.txt temp/UD_CZECH_output_baseline-s_32_5e-05/best/test_predictions.txt 


python script_t_test.py  temp/UD_DANISH_output_Shapley_32_5e-05/best/test_gold.txt   temp/UD_DANISH_output_Shapley_32_5e-05/best/test_predictions.txt temp/UD_DANISH_output_baseline-s_16_2e-05/best/test_predictions.txt 


python script_t_test.py  temp/UD_DUTCH_output_Shapley_32_3e-05/best/test_gold.txt   temp/UD_DUTCH_output_Shapley_32_3e-05/best/test_predictions.txt temp/UD_DUTCH_output_baseline_16_5e-05/best/test_predictions.txt 


'''