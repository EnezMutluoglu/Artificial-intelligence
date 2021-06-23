import tensorflow as tf
import numpy as np


class LSTM_Model():
    def __init__(self, num_layers, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate

        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        self.lstm_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell for _ in range(self.num_layers)])
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('lstm_vars'):
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))

            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.rnn_size],
                                            tf.float32, tf.random_normal_initializer())
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]




        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed, self.initial_state, self.lstm_cell)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, rnn_size])
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)

        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        s2s_loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                            [tf.ones([self.batch_size * self.training_seq_len])],
                            self.vocab_size)
        self.loss = tf.reduce_sum(s2s_loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))



    def sample(self, sess, words, vocab, num=20, prime_text='hakan'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        char_list = list(prime_text)
        for char in char_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        char = char_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            char = words[sample]
            out_sentence = out_sentence + char

        return out_sentence
