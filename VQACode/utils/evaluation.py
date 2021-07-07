import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score

import os
import sys
from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu, test
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor 
# from poycocoevalcap.ciderD.ciderD import CiderD 


# class for evaluate the performance of model predictions
class AnswerEvaluator:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, usecols=[1, 2], names=None)
        self.df_li = self.df.values.tolist()[1:]
        self.pairs = []
        tknzr = TweetTokenizer()
        for n in range(len(self.df_li)):
            self.pairs.append([])
            self.pairs[n].append([tknzr.tokenize(str(self.df_li[n][0]))])
            self.pairs[n].append(tknzr.tokenize(str(self.df_li[n][1])))

    # New bleu
    def bleu(self):
      id = self.df.index
      gts = {}
      res= {}
      tknzr = TweetTokenizer()
      for i in id: 
        gts[i] = [' '.join(tknzr.tokenize(str(self.df['true answer'][i])))]
        res[i] = [' '.join(tknzr.tokenize(str(self.df['predicted answer'][i])))]
      scorers = [
          (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
          # (Meteor(), "METEOR"),
          (Rouge(), "ROUGE_L"),
          (Cider(), "CIDEr"),
      ]
      agg_scores = {}
      for scorer, method in scorers:
          score, scores = scorer.compute_score(gts, res)
          if type(score) == list:
              for m,s in zip(method, score):
                  agg_scores[m] = s
          else:
              agg_scores[method] = score
      return agg_scores

    # calculate accuracy
    def accuracy(self):
        match = 0
        num_pairs = 0
        for pair in self.df_li:
            num_pairs += 1
            if pair[0] == pair[1]: # when prediction and answer are exactly the same
                match += 1
        accuracy = round(match / num_pairs * 100, 2)
        return accuracy

    # calculate exact match of the words
    def exact_match(self):
        word_match = 0
        total = 0
        for pair in self.pairs:
            for word in set(pair[0][0]):
                if word in pair[1]:
                    word_match += 1
            total += len(pair[1])
        exa_match = round((word_match / total * 100), 2)
        return exa_match

    # calculate F1 score
    def f1(self):
        true_ans = []
        pre_ans = []
        for pair in self.df_li:
            true_ans.append(pair[0])
            pre_ans.append(pair[1])
        prediction_f1 = f1_score(true_ans, pre_ans, average='weighted') * 100
        prediction_f1 = round(prediction_f1, 2)
        return prediction_f1

    # # calculating BLEU scores
    # def bleu(self, b_type=1):
    #     # set weights for each BLEU type
    #     if b_type == 1:
    #         bleu_weights = [1, 0, 0, 0]
    #     elif b_type == 2:
    #         bleu_weights = (0.5, 0.5, 0, 0)
    #     elif b_type == 3:
    #         bleu_weights = (0.3, 0.3, 0.3, 0)
    #     elif b_type == 4:
    #         bleu_weights = (0.25, 0.25, 0.25, 0.25)
    #     else:
    #         raise ValueError("Please enter correct BLEU type: 1, 2, 3 or 4")

    #     n = 0
    #     score = 0
    #     # calculate BLEUs for each answer, and take average
    #     for pair in self.pairs:
    #         n += 1
    #         score += sentence_bleu(pair[0], pair[1],
    #                                weights=bleu_weights,
    #                                smoothing_function=None)
    #     avg = score / n * 100
    #     avg = round(avg, 2)
    #     return avg

    # evaluation function for each metrics
    def evaluate(self):
        accuracy = self.accuracy()
        exact_m = self.exact_match()
        F1 = self.f1()
        bleus = self.bleu()
        bleu_1 = round(bleus['Bleu_1'], 2)
        bleu_2 = round(bleus['Bleu_2'], 2)
        bleu_3 = round(bleus['Bleu_3'], 2)
        bleu_4 = round(bleus['Bleu_4'], 2)

        # save scores to dict
        dic = {'Accuracy': accuracy, 'Exact Match': exact_m, 'F1 Score': F1, 'BLEU-1': bleu_1, 'BLEU-2': bleu_2,
               'BLEU-3': bleu_3, 'BLEU-4': bleu_4}

        for key in dic.keys():
            print(key + ": " + str(dic[key]))
        return dic


# test evaluators
if __name__ == '__main__':
    file_name = '../open_ended_results/transformer/elmobert/resnet_2_open.csv'
    AnswerEvaluator(file_name).evaluate()
