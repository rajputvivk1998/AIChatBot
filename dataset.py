import os
import re
import json
import numpy as np
import pandas as pd
from ast import literal_eval
import yaml

from utils import clean_text

class CornellMovieDialogDataset:    # I just copied this class , rest of the classes are codeed by me.ok
    def __init__(self, path):
        """ Constructor:
            Arguments:
                path(string): Path of the dataset
        """
        self.DELIM = ' +++$+++ '
        self.path  = path
        self.movie_lines_filepath = os.path.join(self.path, 'movie_lines.txt')
        self.movie_conversations = os.path.join(self.path, 'movie_conversations.txt')

    def get_id2line(self):
        """
        1. Read from 'movie-lines.txt'
        2. Create a dictionary with ( key = line_id, value = text )
        :return: (dict) {line-id: text, ...}
        """
        id2line = {}
        id_index = 0
        text_index = 4
        with open(self.movie_lines_filepath, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(self.DELIM)
                if len(items) == 5:
                    line_id = items[id_index]
                    dialog_text = items[text_index].strip()
                    dialog_text = clean_text(dialog_text)
                    id2line[line_id] = dialog_text
        return id2line

    def get_conversations(self):
        """
        1. Read from 'movie_conversations.txt'
        2. Create a list of [list of line_id's]
        :return: [list of line_id's]
        """
        conversation_ids_index = -1
        conversations = []
        with open(self.movie_conversations, 'r', encoding='iso-8859-1') as f:
            for line in f:
                items = line.split(self.DELIM)
                conversation_ids_field = items[conversation_ids_index]
                conversation_ids = literal_eval(conversation_ids_field)  # evaluate as a python list
                conversations.append(conversation_ids)
        return conversations

    def get_question_answer_set(self, id2line, conversations):
        """
        Want to collect questions and answers
        (this current method is iffy... not sure how this accurately defines questions/answers...)
        :param conversations: (list) Collection line ids consisting of a single conversation
        :param id2line: (dict) mapping of line-ids to actual line text
        :return: (list) questions, (list) answers
        """
        questions = []
        answers = []

        # This uses a simple method in an attempt to gather question/answers
        for conversation in conversations:
            if len(conversation) % 2 != 0:
                conversation = conversation[:-1]  # remove last item

            for idx, line_id in enumerate(conversation):
                if idx % 2 == 0:
                    questions.append(id2line[line_id])
                else:
                    answers.append(id2line[line_id])

        return questions, answers

    def get_QA(self):
        id2line = self.get_id2line()
        conv = self.get_conversations()
        return self.get_question_answer_set(id2line, conv)


    def prepare_seq2seq_files(self, questions, answers, output_directory, test_set_size=30000):
        """
        Preparing training/test data for:
        https://github.com/llSourcell/tensorflow_chatbot
        :param questions: (list)
        :param answers: (list)
        :param output_directory: (str) Directory to write files
        :param test_set_size: (int) number of samples to use for test data set
        :return: train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath
        """

        # open files
        train_enc_filepath = os.path.join(output_directory, 'train.enc')
        train_dec_filepath = os.path.join(output_directory, 'train.dec')
        test_enc_filepath = os.path.join(output_directory, 'test.enc')
        test_dec_filepath = os.path.join(output_directory,'test.dec')

        train_enc = open(train_enc_filepath, 'w', encoding='utf8')
        train_dec = open(train_dec_filepath, 'w', encoding='utf8')
        test_enc = open(test_enc_filepath, 'w', encoding='utf8')
        test_dec = open(test_dec_filepath, 'w', encoding='utf8')

        # choose test_set_size number of items to put into testset
        test_ids = random.sample(range(len(questions)), test_set_size)

        for i in range(len(questions)):
            if i in test_ids:
                test_enc.write(questions[i])
                test_dec.write(answers[i])
            else:
                train_enc.write(questions[i])
                train_dec.write(answers[i])

        # close files
        train_enc.close()
        train_dec.close()
        test_enc.close()
        test_dec.close()
        return train_enc_filepath, train_dec_filepath, test_enc_filepath, test_dec_filepath

class GuntercoxDataset:
    def __init__(self, path):
        self.path = path

    def parse_dataset(self):
        input_texts = []
        target_texts = []

        for file in os.listdir(self.path):
            filepath = os.path.join(self.path, file)
            if os.path.isfile(filepath):
                my_dict = yaml.load(open(filepath))
                conversations = my_dict['conversations']

                for qa in conversations:
                    input_texts.append(clean_text(qa[0]))
                    target_texts.append(clean_text(qa[1]))

        return input_texts, target_texts

## Fliker-30K Dataset
class Flickr30KDataset:
    def __init__(self, path):
        self.path = path
        caption_file = open(os.path.join(self.path, "results.csv"), "r").read()
        lines = caption_file.split("\n")

        self.file_names = []
        self.captions = []

        for line in lines:
            line = line.strip()
            cols = line.split("|")

            try:
                self.captions.append(cols[2])
                self.file_names.append(cols[0])
            except Exception as e:
                pass

    def get_data(self):
        return self.file_names[1:], self.captions[1:]

## Rdant conversations
class Rdany:
    def __init__(self, path):
        self.path = path
        self.file = pd.read_csv(self.path)
        self.file['text'] = self.file['text'].apply(lambda x:clean_text(x))
        self.file = self.file[['source','text']]
        self.prepare_dataset()

    def prepare_dataset(self):
        x = []
        y = []

        messages = self.file["text"]
        labels = self.file["source"]

        for i in range(len(messages)-1):
            if messages[i] == "start":
                pass
            else:
                x.append(messages[i])
                y.append(messages[i+1])
        return x, y

if __name__ == "__main__":
    ds = GuntercoxDataset("data/gunthercox/")
    q, a = ds.parse_dataset()

    for i in range(len(q)):
        print(q[i], "->", a[i])
