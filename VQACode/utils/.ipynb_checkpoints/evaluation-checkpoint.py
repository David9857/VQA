import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score


class AnswerEvaluator:
    def __init__(self, file_path):
        df = pd.read_csv(file_path, usecols=[1, 2], names=None)
        self.df_li = df.values.tolist()[1:]
        self.pairs = []
        tknzr = TweetTokenizer()
        for n in range(len(self.df_li)):
            self.pairs.append([])
            self.pairs[n].append([tknzr.tokenize(str(self.df_li[n][0]))])
            self.pairs[n].append(tknzr.tokenize(str(self.df_li[n][1])))

    def accuracy(self):
        match = 0
        num_pairs = 0
        for pair in self.df_li:
            num_pairs += 1
            if pair[0] == pair[1]:
                match += 1
        accuracy = round(match / num_pairs * 100, 2)
        return accuracy

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

    def f1(self):
        true_ans = []
        pre_ans = []
        for pair in self.df_li:
            true_ans.append(pair[0])
            pre_ans.append(pair[1])
        prediction_f1 = f1_score(true_ans, pre_ans, average='weighted') * 100
        prediction_f1 = round(prediction_f1, 2)
        return prediction_f1

    def bleu(self, b_type=1):
        if b_type == 1:
            bleu_weights = [1, 0, 0, 0]
        elif b_type == 2:
            bleu_weights = (0.5, 0.5, 0, 0)
        elif b_type == 3:
            bleu_weights = (0.3, 0.3, 0.3, 0)
        elif b_type == 4:
            bleu_weights = (0.25, 0.25, 0.25, 0.25)
        else:
            raise ValueError("Please enter correct BLEU type: 1, 2, 3 or 4")

        n = 0
        score = 0
        for pair in self.pairs:
            n += 1
            score += sentence_bleu(pair[0], pair[1],
                                   weights=bleu_weights,
                                   smoothing_function=SmoothingFunction().method5)
        avg = score / n * 100
        avg = round(avg, 2)
        return avg

    def evaluate(self):
        accuracy = self.accuracy()
        exact_m = self.exact_match()
        F1 = self.f1()
        bleu_1 = self.bleu(1)
        bleu_2 = self.bleu(2)
        bleu_3 = self.bleu(3)
        bleu_4 = self.bleu(4)

        dic = {'Accuracy': accuracy, 'Exact Match': exact_m, 'F1 Score': F1, 'BLEU-1': bleu_1, 'BLEU-2': bleu_2,
               'BLEU-3': bleu_3, 'BLEU-4': bleu_4}

        for key in dic.keys():
            print(key + ": " + str(dic[key]))
        return dic


if __name__ == '__main__':
    file_name = 'transformer_encoder_decoder_resnet_layers_2.csv'
    AnswerEvaluator(file_name).evaluate()
