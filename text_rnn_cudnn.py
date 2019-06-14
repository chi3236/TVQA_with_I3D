import sys
import numpy as np
import tensorflow as tf

'''
network for two-stream video feature
'''
class tRNN:
    def __init__(self, rnn_type, num_hidden, text_type):
        self.rnn_type = rnn_type
        self.num_hidden = num_hidden
        self.text_type = text_type

    def build(self, inputs, dropout_keep_prob, seq_len=None):
        '''
        :param input: (batch_size, sequcnce_len, 300)
        :param is_training: whether to use training or inference mode
        :param dropout_keep_prob: Probability for the tf.nn.dropout layer (float in [0, 1)).
        :return:
        '''
        if self.rnn_type == "GRU":
            def make_cell():
                #cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, name=self.text_type)
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob[0][0])
                return cell
        elif self.rnn_type == "LSTM":
            def make_cell():
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob[0][0])
                return cell
        else:
            ValueError("invalid rnn type")
            exit()

        #net = tf.reshape(inputs, [-1, tf.shape(inputs)[1], 300])
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(1)], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(1)], state_is_tuple=True)
        if seq_len is not None:
            seq_len = tf.squeeze(seq_len, axis=[1])
            net, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length=seq_len, dtype=tf.float32)
        else:
            net, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)

        #net = tf.layers.dense(net, 500, tf.nn.relu)
        #fw_net = tf.reshape(net[0], [-1, 300])
        #bw_net = tf.reshape(net[1], [-1, 300])
        net = tf.concat([net[0], net[1]], 2)
        return net

class tRNN_answer:
    def __init__(self, rnn_type, num_hidden, text_type):
        self.rnn_type = rnn_type
        self.num_hidden = num_hidden
        self.text_type = text_type

    def build(self, inputs, is_training, dropout_keep_prob=1.0, seq_len=None):
        '''
        :param input: (batch_size, 5, 300)
        :param is_training: whether to use training or inference mode
        :param dropout_keep_prob: Probability for the tf.nn.dropout layer (float in [0, 1)).
        :return:
        '''
        if is_training:
            dropout_keep_prob = 0.5
        if self.rnn_type == "GRU":
            def make_cell():
                cell = tf.nn.rnn_cell.GRUCell(self.num_hidden * 5, name=self.text_type)
                #cell = tf.contrib.cudnn_rnn_CudnnCompatibleGRUCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                return cell
        elif self.rnn_type == "LSTM":
            def make_cell():
                #cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden * 5, name=self.text_type)
                cell = tf.contrib.cudnn_rnn_CudnnCompatibleLSTMCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                return cell
        else:
            ValueError("invalid rnn type")
            exit()

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(2)], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(2)], state_is_tuple=True)
        #net = tf.reshape(inputs, [-1, 1, 1500])
        if seq_len is not None:
            net, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length=seq_len, dtype=tf.float64)
        else:
            net, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float64)

        #fw_net = tf.reshape(net[0], [-1, 5, 300])
        #bw_net = tf.reshape(net[1], [-1, 5, 300])
        net = tf.concat([net[0], net[1]], 2)
        return net
