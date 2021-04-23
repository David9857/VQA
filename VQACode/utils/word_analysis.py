import json
import nltk
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


# function to tokenize a sentence to a word
def tokenize(sentence):
    tokens = tokenizer.tokenize(str(sentence).lower())
    return tokens


# count total words and vocabulary for all answers
def count_words(QA_list):
    words_list = []
    vocab_list = []
    for pair in QA_list:
        answer = pair['Answers']
        a_list = tokenize(answer.lower())
        for w in a_list:
            words_list.append(w)
            if w not in vocab_list:
                vocab_list.append(w)
    return words_list, vocab_list


if __name__ == '__main__':
    # open QA mapping jsons
    with open('../data/jsons/QA.json', 'r', encoding='utf-8') as f_train:
        train_QA = json.load(f_train)

    # get word list from answers
    word_l, _ = count_words(train_QA)
    total = len(word_l)

    # save the frequency of each word
    freq = dict()
    for word in word_l:
        freq[word] = freq.get(word, 0) + 1

    # save results to file
    df = pd.DataFrame().from_dict(freq, orient='index', columns=['frequency'])
    df.to_csv('../data/answer_word_frequency.csv')
