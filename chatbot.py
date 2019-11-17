import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import gc
from tqdm import tqdm # tqdm is? A fast, extensible progress bar for Python and CLI. It's just to see the progress. We use it during training.
from sklearn.utils import shuffle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2

from vocabulary import Vocabulary
from dataset import *

class ChatBot:
    def __init__(self, layers=5, maxlen=10, embedding_size=128, batch_size=32, is_train=True, lr=0.0001):
        self.layers = layers
        self.maxlen = maxlen
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = lr
        self.model_path = "model/chatbot/model.npz" #what is npz? It is the extension , it is the file in which we save the weight of our seq2seq model.

        ## Vocabulary
        self.vocab = Vocabulary(corpus=None, maxlen=maxlen)
        self.vocab_size = self.vocab.vocab_size

        ## Init Session
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf.reset_default_graph()
        self.sess = tf.Session(config=sess_config)

        ## Placeholders
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.decoder_outputs = tf.placeholder(tf.int32, shape=[None, None])
        self.mask = tf.placeholder(tf.int32, shape=[None, None])

        ## Model
        self.net_out, _ = self.create_model(
            self.encoder_inputs,
            self.decoder_inputs,
            self.vocab_size,
            self.embedding_size,
            reuse=False)
        self.net_out.print_params(False)

        self.loss = tl.cost.cross_entropy_seq_with_mask(
            logits=self.net_out.outputs,
            target_seqs=self.decoder_outputs,
            input_mask=self.mask,
            return_details=False,
            name='cost')

        ## Optimizer
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, X, Y, num_epochs=1):
        ## Init Vars
        self.sess.run(tf.global_variables_initializer())

        ## Load Model
        tl.files.load_and_assign_npz(sess=self.sess, name=self.model_path, network=self.net_out)

        n_step = len(X)//self.batch_size

        for epoch in range(num_epochs):
            X, Y = shuffle(X, Y, random_state=0)
            total_loss, n_iter = 0, 0
            for x, y in tqdm(tl.iterate.minibatches(
                inputs=X,
                targets=Y,
                batch_size=self.batch_size,
                shuffle=False),
                total=n_step,
                desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs),
                leave=False):

                x1, x2, y1, W = self.vocab.dataset(x, y)
                feed_data = {}
                feed_data[self.encoder_inputs] = x1
                feed_data[self.decoder_inputs] = x2
                feed_data[self.decoder_outputs] = y1
                feed_data[self.mask] = W


                _, loss_iter = self.sess.run([self.train_op, self.loss], feed_dict=feed_data)
                total_loss += loss_iter
                n_iter += 1

            ## printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))

            ## saving the model
            tl.files.save_npz(self.net_out.all_params, name=self.model_path, sess=self.sess)

        ## session cleanup
        self.sess.close()


    """
    Creates the LSTM Model
    """
    def create_model(self, encoder_inputs, decoder_inputs, vocab_size, emb_dim, is_train=True, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            # for chatbot, you can use the same embedding layer,
            # for translation, you may want to use 2 seperated embedding layers # embedding layers?
            with tf.variable_scope("embedding") as vs:
                net_encode = EmbeddingInputlayer(
                    inputs = encoder_inputs,
                    vocabulary_size = vocab_size,
                    embedding_size = emb_dim,
                    name = 'seq_embedding')
                vs.reuse_variables()
                net_decode = EmbeddingInputlayer(
                    inputs = decoder_inputs,
                    vocabulary_size = vocab_size,
                    embedding_size = emb_dim,
                    name = 'seq_embedding')

            net_rnn = Seq2Seq(net_encode, net_decode,
                    cell_fn = tf.nn.rnn_cell.LSTMCell,
                    n_hidden = emb_dim,
                    initializer = tf.random_uniform_initializer(-0.1, 0.1),
                    encode_sequence_length = retrieve_seq_length_op2(encoder_inputs),
                    decode_sequence_length = retrieve_seq_length_op2(decoder_inputs),
                    initial_state_encode = None,
                    dropout = (0.5 if is_train else None),
                    n_layer = self.layers,
                    return_seq_2d = True,
                    name = 'seq2seq')

            net_out = DenseLayer(net_rnn, n_units=vocab_size, act=tf.identity, name='output')
        return net_out, net_rnn


    def infer(self, query):
        unk_id = self.vocab.word_index["<unk>"]
        pad_id = self.vocab.word_index["<pad>"]

        start_id = self.vocab.word_index["<start>"]
        end_id = self.vocab.word_index["<end>"]

        ## Init Session
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf.reset_default_graph()
        sess = tf.Session(config=sess_config)

        ## Inference Data Placeholders
        encode_inputs = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_inputs")
        decode_inputs = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_inputs")

        net, net_rnn = self.create_model(
            encode_inputs,
            decode_inputs,
            self.vocab_size,
            self.embedding_size,
            is_train=False,
            reuse=False)
        y = tf.nn.softmax(net.outputs)

        ## Init Vars
        sess.run(tf.global_variables_initializer())

        ## Load Model
        tl.files.load_and_assign_npz(sess=sess, name=self.model_path, network=net)

        """
        Inference using pre-trained model
        """
        def inference(seed):
            seed_id = self.vocab.text_to_sequence(seed)

            ## Encode and get state
            state = sess.run(net_rnn.final_state_encode, {encode_inputs: [seed_id]})

            ## Decode, feed start_id and get first word [https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py]
            o, state = sess.run([y, net_rnn.final_state_decode], {
                net_rnn.initial_state_decode: state,
                decode_inputs: [[start_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=3)
            #w = self.vocab.index_word[w_id]

            ## Decode and feed state iteratively
            sentence = [w_id]
            for _ in range(self.maxlen): # max sentence length
                o, state = sess.run([y, net_rnn.final_state_decode],{
                    net_rnn.initial_state_decode: state,
                    decode_inputs: [[w_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=2)
                #w = self.vocab.index_word[w_id]
                if w_id == end_id:
                    break
                sentence = sentence + [w_id]
            return sentence

        ## infer
        sentence = inference(query)
        response = self.vocab.seqs_to_text(sentence)
        response = " ".join(response.split(" "))
        return response

if __name__ == "__main__":
    ds1 = CornellMovieDialogDataset("data/cornell-dialogs/")
    ds2 = GuntercoxDataset("data/gunthercox/")
    ds3 = Rdany("data/rdany_conversations_2016-03-01.csv")

    q1, a1 = ds1.get_QA()
    q2, a2 = ds2.parse_dataset()
    q3, a3 = ds3.prepare_dataset()

    Q = q1 + q2 + q3
    A = a1 + a2 + a3

    chatbot = ChatBot(layers=4, maxlen=20, embedding_size=256, batch_size=128, is_train=True, lr=0.001)
    chatbot.train(Q, A, num_epochs=50)
    while True:
        resp = input("YOU=> ")
        msg = chatbot.infer(resp)
        print("BOT=> ", msg)
