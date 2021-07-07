import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import re
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self, location, emb_folder, image_size=None):
        self.root = pathlib.Path(location)
        if not self.root.exists():  # check if path of data exist
            raise ValueError('The location of data does not exist: {}.'.format(location))
        # specify image and question feature location and size
        if image_size is None:
            image_size = [224, 224]
        self.image_size = image_size
        pic = self.root.joinpath('pic')  # get picture location
        ques = self.root.joinpath('ques_embeddings/' + emb_folder)
        kn = self.root.joinpath('knowledge_embeddings')

        # create data frame for yes_no and open-ended questions
        self.dfs = dict()
        self.dfs['QA'] = shuffle(pd.read_json(self.root.joinpath('jsons/QA').with_suffix('.json')),
                                         random_state=1)
        self.dfs['open_ended'] = shuffle(pd.read_json(self.root.joinpath('jsons/open_ended').with_suffix('.json')),
                                         random_state=1)
        self.dfs['yes_no'] = shuffle(pd.read_json(self.root.joinpath('jsons/yes_no').with_suffix('.json')),
                                     random_state=1)
        # create dataframe of paths for image and question features
        for key in self.dfs.keys():
            df = self.dfs[key]
            df['image'] = df['Images'].map(lambda file: str(pic.joinpath(str(file))) + '.jpg')
            df['question'] = df['Question_Id'].map(lambda file: str(ques.joinpath(str(file))) + '.npy')
            df['knowlwdge'] = df['Question_Id'].map(lambda file: str(kn.joinpath(str(file))) + '.npy')

    # function for loading images
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path) # read image file
        image = tf.image.decode_jpeg(image, channels=3) # decode image
        image = tf.image.resize(image, self.image_size)
        return image

    # function for loading question embeddings
    @staticmethod
    def load_question_features(path):
        return np.load(path)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:-
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    @staticmethod
    def process_answer(w):
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # function for creating dataset
    def create_dataset(self, ques_type='open_ended'):
        assert ques_type in ['QA', 'yes_no', 'open_ended'], "Invalid question type."
        print('QA:',len(self.dfs['QA']))
        print('yes_no:',len(self.dfs['yes_no']))
        print('open_ended',len(self.dfs['open_ended']))
        print('Load:',ques_type)
        # get question id for datasets, useful for result analysis
        ques_id_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['Question_Id'])

        # create image dataset
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['image'])
        image_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create question feature dataset
        ques_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['question'])
        ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ##### create knowledge feature dataset
        kn_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['knowlwdge'])
        kn_ds = kn_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create answer dataset
        answers = self.dfs[ques_type]['Answers'].map(lambda x: self.process_answer(x))
        # use tokenizer for string and one-hot mapping, counting vocab, max length
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", lower=True)
        tokenizer.fit_on_texts(answers)
        answers = tokenizer.texts_to_sequences(answers)
        # use 0 as padding
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, padding='post')
        ans_ds = tf.data.Dataset.from_tensor_slices(answers)
        # print('11111')

        return tf.data.Dataset.zip(((image_ds, ques_ds, kn_ds), ans_ds, ques_id_ds)), tokenizer

class CBDataLoader:
    def __init__(self, location, emb_folder, image_size=None):
        self.root = pathlib.Path(location)
        if not self.root.exists():  # check if path of data exist
            raise ValueError('The location of data does not exist: {}.'.format(location))

        # specify image and question feature location and size
        if image_size is None:
            image_size = [224, 224]
        self.image_size = image_size
        pic = self.root.joinpath('pic')  # get picture location
        ques = self.root.joinpath('ques_embeddings/' + emb_folder)
        # kn = self.root.joinpath('knowledge_embeddings')

        # create data frame for yes_no and open-ended questions
        self.dfs = dict()
        self.dfs['QA'] = shuffle(pd.read_json(self.root.joinpath('jsons/QA').with_suffix('.json')),
                                         random_state=1)
        self.dfs['open_ended'] = shuffle(pd.read_json(self.root.joinpath('jsons/open_ended').with_suffix('.json')),
                                         random_state=1)
        self.dfs['yes_no'] = shuffle(pd.read_json(self.root.joinpath('jsons/yes_no').with_suffix('.json')),
                                     random_state=1)
        # create dataframe of paths for image and question features
        for key in self.dfs.keys():
            df = self.dfs[key]
            df['image'] = df['Images'].map(lambda file: str(pic.joinpath(str(file))) + '.jpg')
            df['question'] = df['Question_Id'].map(lambda file: str(ques.joinpath(str(file))) + '.npy')
            # df['knowlwdge'] = df['Question_Id'].map(lambda file: str(kn.joinpath(str(file))) + '.npy')

    # function for loading images
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path) # read image file
        image = tf.image.decode_jpeg(image, channels=3) # decode image
        image = tf.image.resize(image, self.image_size)
        return image

    # function for loading question embeddings
    @staticmethod
    def load_question_features(path):
        return np.load(path)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:-
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    @staticmethod
    def process_answer(w):
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # function for creating dataset
    def create_dataset(self, ques_type='QA'):
        assert ques_type in ['QA', 'yes_no', 'open_ended'], "Invalid question type."

        print('QA:',len(self.dfs['QA']))
        print('yes_no:',len(self.dfs['yes_no']))
        print('open_ended',len(self.dfs['open_ended']))
        print('Load:',ques_type)
        # get question id for datasets, useful for result analysis
        ques_id_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['Question_Id'])

        # create image dataset
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['image'])
        image_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create question feature dataset
        ques_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['question'])
        ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ##### create knowledge feature dataset
        # kn_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['knowlwdge'])
        # kn_ds = kn_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
        #                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create answer dataset
        answers = self.dfs[ques_type]['Answers'].map(lambda x: self.process_answer(x))
        # use tokenizer for string and one-hot mapping, counting vocab, max length
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", lower=True)
        tokenizer.fit_on_texts(answers)
        answers = tokenizer.texts_to_sequences(answers)
        # use 0 as padding
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, padding='post')
        ans_ds = tf.data.Dataset.from_tensor_slices(answers)

        return tf.data.Dataset.zip(((image_ds, ques_ds), ans_ds, ques_id_ds)), tokenizer

class ClefDataLoader:
    def __init__(self, location, embedding_folder, ques_type, image_size=None):
        self.root = pathlib.Path(location)
        if not self.root.exists():  # check if path of data exist
            raise ValueError('The location of data does not exist: {}.'.format(location))

        if image_size is None:
            image_size = [224, 224]
        self.image_size = image_size
        self.pic = self.root.joinpath('pic')  # get picture location
        self.ques_type =ques_type
        self.ques = self.root.joinpath('ques_embeddings/' + embedding_folder)

        self.df = pd.read_json(self.root.joinpath('jsons/'+ques_type).with_suffix('.json'))
        self.df['image'] = self.df['Images'].map(lambda file: str(self.pic.joinpath(str(file))) + '.jpg')
        self.df['question'] = self.df['Question_Id'].map(lambda file: str(self.ques.joinpath(str(file))) + '.npy')

    # function for loading images
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)  # read image file
        image = tf.image.decode_jpeg(image, channels=3)  # decode image
        image = tf.image.resize(image, self.image_size)
        return image

    def create_dataset(self):
        if self.ques_type in ['c4', 'overall']:
            ques_id_ds = tf.data.Dataset.from_tensor_slices(self.df['Question_Id'])

            image_path_ds = tf.data.Dataset.from_tensor_slices(self.df['image'])
            image_ds = image_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # create question feature dataset
            ques_path_ds = tf.data.Dataset.from_tensor_slices(self.df['question'])
            ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(DataLoader.load_question_features, inp=[x], Tout=tf.float32),
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

            answers = self.df['Answers'].map(lambda x: DataLoader.process_answer(x))
            # use tokenizer for string and one-hot mapping, counting vocab, max length
            tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", lower=True)
            tokenizer.fit_on_texts(answers)
            answers = tokenizer.texts_to_sequences(answers)
            # use 0 as padding
            tokenizer.word_index['<pad>'] = 0
            tokenizer.index_word[0] = '<pad>'
            answers = tf.keras.preprocessing.sequence.pad_sequences(answers, padding='post')
            ans_ds = tf.data.Dataset.from_tensor_slices(answers)

            return tf.data.Dataset.zip(((image_ds, ques_ds), ans_ds, ques_id_ds)), tokenizer

        else:
            ques_id_ds = tf.data.Dataset.from_tensor_slices(self.df['Question_Id'])

            image_path_ds = tf.data.Dataset.from_tensor_slices(self.df['image'])
            image_ds = image_path_ds.map(self.load_and_preprocess_image,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # create question feature dataset
            ques_path_ds = tf.data.Dataset.from_tensor_slices(self.df['question'])
            ques_ds = ques_path_ds.map(
                lambda x: tf.numpy_function(DataLoader.load_question_features, inp=[x], Tout=tf.float32),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

            vocab = self.df['Answers'].unique()
            tokenizer = OnehotManager(vocab)
            answers = tf.data.Dataset.from_tensor_slices(self.df['Answers'])
            ans_ds = answers.map(lambda x: tf.numpy_function(tokenizer.get_index, inp=[x], Tout=tf.int32),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return tf.data.Dataset.zip(((image_ds, ques_ds), ans_ds, ques_id_ds)), tokenizer

class SlakeDataLoader:
    def __init__(self, location, emb_folder, image_size=None):
        self.root = pathlib.Path(location)
        if not self.root.exists():  # check if path of data exist
            raise ValueError('The location of data does not exist: {}.'.format(location))
        # specify image and question feature location and size
        if image_size is None:
            image_size = [224, 224]
        self.image_size = image_size
        pic = self.root.joinpath('imgs')  # get picture location
        ques = self.root.joinpath('slake_question_embeddings/questions_embedding_train')
        kn = self.root.joinpath('slake_knowledge_embeddings/knowledge_embeddings_train')

        # create data frame for yes_no and open-ended questions
        self.dfs = dict()
        self.dfs['QA'] = shuffle(pd.read_json(self.root.joinpath('train').with_suffix('.json')),
                                         random_state=1)
        # print(type(self.dfs['QA']))
        # print(self.dfs['QA'])
        tmp = self.dfs['QA']
        tmp = tmp.drop(tmp[tmp.q_lang == 'zh'].index)
        self.dfs['QA'] = tmp
        # print(self.dfs['QA']['q_lang'])
        
        # self.dfs['open_ended'] = shuffle(pd.read_json(self.root.joinpath('jsons/open_ended').with_suffix('.json')),
        #                                  random_state=1)
        # self.dfs['yes_no'] = shuffle(pd.read_json(self.root.joinpath('jsons/yes_no').with_suffix('.json')),
                                    #  random_state=1)
        # create dataframe of paths for image and question features
        for key in self.dfs.keys():
            df = self.dfs[key]
            df['image'] = df['img_name'].map(lambda file: str(pic.joinpath(str(file))))
            df['question'] = df['qid'].map(lambda file: str(ques.joinpath(str(file))) + '.npy')
            df['knowlwdge'] = df['qid'].map(lambda file: str(kn.joinpath(str(file))) + '.npy')

    # function for loading images
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path) # read image file
        image = tf.image.decode_jpeg(image, channels=3) # decode image
        image = tf.image.resize(image, self.image_size)
        return image

    # function for loading question embeddings
    @staticmethod
    def load_question_features(path):
        return np.load(path)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:-
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    @staticmethod
    def process_answer(w):
        w = str(w).lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z1-9?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # function for creating dataset
    def create_dataset(self, ques_type='open_ended'):
        assert ques_type in ['QA', 'yes_no', 'open_ended'], "Invalid question type."
        print('QA:',len(self.dfs['QA']))
        # print('yes_no:',len(self.dfs['yes_no']))
        # print('open_ended',len(self.dfs['open_ended']))
        print('Load:',ques_type)
        # get question id for datasets, useful for result analysis
        ques_id_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['qid'])

        # create image dataset
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['image'])
        image_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create question feature dataset
        ques_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['question'])
        ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ##### create knowledge feature dataset
        kn_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['knowlwdge'])
        kn_ds = kn_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create answer dataset
        answers = self.dfs[ques_type]['answer'].map(lambda x: self.process_answer(x))
        # use tokenizer for string and one-hot mapping, counting vocab, max length
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", lower=True)
        tokenizer.fit_on_texts(answers)
        answers = tokenizer.texts_to_sequences(answers)
        # use 0 as padding
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, padding='post')
        ans_ds = tf.data.Dataset.from_tensor_slices(answers)

        return tf.data.Dataset.zip(((image_ds, ques_ds, kn_ds), ans_ds, ques_id_ds)), tokenizer
        
class RadDataLoader:
    def __init__(self, location, emb_folder, image_size=None):
        self.root = pathlib.Path(location)
        if not self.root.exists():  # check if path of data exist
            raise ValueError('The location of data does not exist: {}.'.format(location))
        # specify image and question feature location and size
        if image_size is None:
            image_size = [224, 224]
        self.image_size = image_size
        pic = self.root.joinpath('pic')  # get picture location
        ques = self.root.joinpath('ques_embeddings/' + emb_folder)
        kn = self.root.joinpath('knowledge_embeddings')

        # create data frame for yes_no and open-ended questions
        self.dfs = dict()
        self.dfs['QA'] = shuffle(pd.read_json(self.root.joinpath('jsons/QA').with_suffix('.json')),
                                         random_state=1)
        self.dfs['open_ended'] = shuffle(pd.read_json(self.root.joinpath('jsons/open_ended').with_suffix('.json')),
                                         random_state=1)
        self.dfs['yes_no'] = shuffle(pd.read_json(self.root.joinpath('jsons/yes_no').with_suffix('.json')),
                                     random_state=1)
        # create dataframe of paths for image and question features
        for key in self.dfs.keys():
            df = self.dfs[key]
            df['image'] = df['Images'].map(lambda file: str(pic.joinpath(str(file))) + '.jpg')
            df['question'] = df['Question_Id'].map(lambda file: str(ques.joinpath(str(file))) + '.npy')
            df['knowlwdge'] = df['Question_Id'].map(lambda file: str(kn.joinpath(str(file))) + '.npy')

    # function for loading images
    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path) # read image file
        image = tf.image.decode_jpeg(image, channels=3) # decode image
        image = tf.image.resize(image, self.image_size)
        return image

    # function for loading question embeddings
    @staticmethod
    def load_question_features(path):
        return np.load(path)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:-
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    @staticmethod
    def process_answer(w):
        w = str(w).lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z1-9?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # function for creating dataset
    def create_dataset(self, ques_type='open_ended'):
        assert ques_type in ['QA', 'yes_no', 'open_ended'], "Invalid question type."
        print('QA:',len(self.dfs['QA']))
        print('yes_no:',len(self.dfs['yes_no']))
        print('open_ended',len(self.dfs['open_ended']))
        print('Load:',ques_type)
        # get question id for datasets, useful for result analysis
        ques_id_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['Question_Id'])

        # create image dataset
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['image'])
        image_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create question feature dataset
        ques_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['question'])
        ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ##### create knowledge feature dataset
        kn_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['knowlwdge'])
        kn_ds = kn_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create answer dataset
        answers = self.dfs[ques_type]['Answers'].map(lambda x: self.process_answer(x))
        # use tokenizer for string and one-hot mapping, counting vocab, max length
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", lower=True)
        tokenizer.fit_on_texts(answers)
        answers = tokenizer.texts_to_sequences(answers)
        # use 0 as padding
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, padding='post')
        ans_ds = tf.data.Dataset.from_tensor_slices(answers)

        return tf.data.Dataset.zip(((image_ds, ques_ds, kn_ds), ans_ds, ques_id_ds)), tokenizer
        
class OnehotManager:
    def __init__(self, vocab):
        vocab = vocab.tolist()
        vocab.insert(0, '<start>')
        self.int2vocab = vocab
        self.vocab2int = dict()
        for idx, token in enumerate(self.int2vocab):
            self.vocab2int[token] = idx
        self.depth = len(self.vocab2int)

    def get_index(self, answer):
        return self.vocab2int[answer.decode('utf-8')]

    def get_answer(self, idx):
        return self.int2vocab[idx]

    def get_onehot(self, answer):
        return np.eye(self.depth, dtype=np.float32)[self.get_index(answer.decode('utf-8'))]

# testing on data loader
if __name__ == '__main__':
    data_loader = ClefDataLoader('../data_clef/training', 'bioelmo', 'overall')
    ds, tokenizer = data_loader.create_dataset()
    vocab_size = len(tokenizer.index_word) + 1
    print(vocab_size)
    for (image, ques), ans, ques_id in ds.take(1):
        print(image.shape)
        print(ques.shape)
        print(ans)
        print(ques_id)

    path_loader = DataLoader('../data', 'bioelmo')
    dataset, toke = path_loader.create_dataset('QA')
    for (image, ques), ans, ques_id in dataset.take(1):
        print(image.shape)
        print(ques.shape)
        print(ans)
        print(ques_id)