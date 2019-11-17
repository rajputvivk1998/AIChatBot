import os, json
import numpy as np
from collections import Counter
import tensorflow as tf

from utils import clean_text

class Vocabulary:
    def __init__(self, corpus=None, path="model/chatbot/", max_words=1000, maxlen=10):
        self.corpus = corpus
        self.path = path
        self.max_words = max_words
        self.maxlen = maxlen

        self.word_index_path = os.path.join(self.path, "word_index.json")
        self.index_word_path = os.path.join(self.path, "index_word.json")
        self.index = 0

        if self.corpus == None:
            with open(self.word_index_path, 'r') as f:
                self.word_index = dict(json.load(f))

            with open(self.index_word_path, 'r') as f:
                self.index_word = dict(json.load(f))

        else:
            self.prepare()

        self.vocab_size = len(self.word_index.keys())

    def prepare(self):
        corpus = self.corpus
        corpus = (corpus.encode('ascii', 'ignore')).decode("utf-8")
        corpus = clean_text(corpus)
        words = corpus.split(" ")
        counts = Counter(words)
        most_common = counts.most_common(self.max_words)
        print(most_common)

        self.word_index = {}
        self.index_word = {}

        for word in ["<pad>", "<start>", "<end>", "<unk>"]:
            self.word_index[word] = self.index
            self.index_word[self.index] = word
            self.index += 1

        for (word, freq) in most_common:
            self.word_index[word] = self.index
            self.index_word[self.index] = word
            self.index += 1

        with open(self.word_index_path, 'w') as f:
            json.dump(self.word_index, f)

        with open(self.index_word_path, 'w') as f:
            json.dump(self.index_word, f)

    def text_to_sequence(self, text, end=False):
        text = text.strip()
        text = clean_text(text)
        text = text.split(" ")
        x2 = np.full((self.maxlen), self.word_index["<pad>"])        ## Decoder Inputs
        y1 = np.full((self.maxlen), self.word_index["<pad>"])        ## Decoder Outputs
        mask = np.zeros((self.maxlen))                               ## Masks

        ## Adding <start>
        x2[0] = self.word_index["<start>"]

        for i, word in enumerate(text):
            try:
                idx = self.word_index[word]
            except KeyError as e:
                idx = self.word_index["<unk>"]
            if i+1 < self.maxlen:
                x2[i+1] = idx
            if i < self.maxlen:
                y1[i] = idx
                mask[i] = 1

        if end == True:
            if len(text) < self.maxlen:
                y1[len(text)] = self.word_index["<end>"]
            else:
                y1[-1] = self.word_index["<end>"]

            if i+1 < self.maxlen:
                mask[i+1] = 1
            return x2, y1, mask

        return y1

    def seqs_to_text(self, seqs):
        text = ""
        for idx in seqs:
            w = self.index_word[str(idx)]
            if w == "<end>":
                break
            text += w + " "
        return text

    def dataset(self, x, y):
        assert len(x) == len(y)
        len_x = len(x)

        enc_inputs = np.zeros((len_x, self.maxlen), dtype=np.int32)        ## Encoder Inputs    x1
        dec_inputs = np.zeros((len_x, self.maxlen), dtype=np.int32)        ## Decoder Inputs    x2
        dec_output = np.zeros((len_x, self.maxlen), dtype=np.int32)        ## Decoder Outputs   y1
        target_mask = np.zeros((len_x, self.maxlen), dtype=np.int32)       ## Mask              w

        i = 0
        for dx, dy in zip(x, y):
            enc_inputs[i] = self.text_to_sequence(dx)
            x2, y1, mask = self.text_to_sequence(dy, end=True)
            dec_inputs[i] = x2
            dec_output[i] = y1
            target_mask[i] = mask

            i+=1

        return enc_inputs, dec_inputs, dec_output, target_mask

if __name__ == "__main__":
    from dataset import *
    ds1 = CornellMovieDialogDataset("data/cornell-dialogs/")
    ds2 = GuntercoxDataset("data/gunthercox/")
    ds3 = Flickr30KDataset("../../../../ML Dataset/flickr30k/")
    ds4 = Rdany("data/rdany_conversations_2016-03-01.csv")

    q1, a1 = ds1.get_QA()
    q2, a2 = ds2.parse_dataset()
    _, text = ds3.get_data()
    q4, a4 = ds4.prepare_dataset()

    Q = q1 + q2 + q4
    A = a1 + a2 + a4

    corpus = Q + A + text

    corpus = " ".join([q for q in Q]) + " ".join([a for a in A])
    vocab = Vocabulary(corpus=corpus, max_words=50000)
