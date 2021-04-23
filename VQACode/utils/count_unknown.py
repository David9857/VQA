import json
from utils.data_loader import get_embedding_model, tokenize


def count_unknown(QA, model, name='the'):
    # get a list of words from QA mapping
    def count_words(QA_list):
        words_list = []
        vocab_list = []
        for pair in QA_list:
            question = pair['Questions']
            answer = pair['Answers']
            q_list = tokenize(question.lower())
            a_list = tokenize(answer.lower())
            l = q_list + a_list
            for word in l:
                words_list.append(word)
                if word not in vocab_list:
                    vocab_list.append(word)
        return words_list, vocab_list

    # count unknown words in a list of words
    def count(word_list):
        counter = 0
        for word in word_list:
            try:
                _ = model[word]
            except:
                counter += 1
        return counter

    words, vocab = count_words(QA)
    words_counter = count(words)
    vocab_counter = count(vocab)

    print("In " + name + ' dataset: ')
    print('Total words =', len(words), '| Unknown words =', words_counter)
    print('Total vocab =', len(vocab), '| Unknown vocab =', vocab_counter)
    print('')
    return {'total_words': len(words), 'unknown_words': words_counter, 'total_vocab': len(vocab), 'unknown_vocab': vocab_counter}


if __name__ == '__main__':
    w2v_model = get_embedding_model('../embedding_models/bio_embedding_extrinsic')
    with open('../data/train/QA.json', 'r', encoding='utf-8') as f_train:
        train_QA = json.load(f_train)
    with open('../data/val/QA.json', 'r', encoding='utf-8') as f_val:
        val_QA = json.load(f_val)
    with open('../data/test/QA.json', 'r', encoding='utf-8') as f_test:
        test_QA = json.load(f_test)
    count_unknown(train_QA, w2v_model, name='train')
    count_unknown(val_QA, w2v_model, name='val')
    count_unknown(test_QA, w2v_model, name='test')
