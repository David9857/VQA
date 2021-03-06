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

        # create data frame for yes_no and open-ended questions
        self.dfs = dict()
        self.dfs['open_ended'] = shuffle(pd.read_json(self.root.joinpath('jsons/open_ended').with_suffix('.json')),
                                         random_state=1)
        self.dfs['yes_no'] = shuffle(pd.read_json(self.root.joinpath('jsons/yes_no').with_suffix('.json')),
                                     random_state=1)

        # create dataframe of paths for image and question features
        for key in self.dfs.keys():
            df = self.dfs[key]
            df['image'] = df['Images'].map(lambda file: str(pic.joinpath(str(file))) + '.jpg')
            df['question'] = df['Question_Id'].map(lambda file: str(ques.joinpath(str(file))) + '.npy')

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
        w = re.sub(r"([?.!,??])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,??]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # function for creating dataset
    def create_dataset(self, ques_type='yes_no'):
        assert ques_type in ['yes_no', 'open_ended'], "Invalid question type."

        # get question id for datasets, useful for result analysis
        ques_id_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['Question_Id'])

        # create image dataset
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['image'])
        image_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create question feature dataset
        ques_path_ds = tf.data.Dataset.from_tensor_slices(self.dfs[ques_type]['question'])
        ques_ds = ques_path_ds.map(lambda x: tf.numpy_function(self.load_question_features, inp=[x], Tout=tf.float32),
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

        return tf.data.Dataset.zip(((image_ds, ques_ds), ans_ds, ques_id_ds)), tokenizer


# testing on data loader
if __name__ == '__main__':
    data_loader = DataLoader('../data', emb_folder='biobert', image_size=[224, 224])
    full_dataset, tokenizer = data_loader.create_dataset(ques_type='open_ended')
    for (img, ques), ans in full_dataset.take(1):
        print(img.shape, ques.shape, ans.shape)