import sys
import numpy as np
import tensorflow as tf

'''
network for two-stream video feature
'''
class vRNN:
    def __init__(self, rnn_type, num_hidden, video_type, FLAGS):
        self.rnn_type = rnn_type
        self.num_hidden = num_hidden
        self.video_type = video_type
        self.FLAGS = FLAGS

    def build(self, inputs, is_training, dropout_keep_prob=1.0, seq_len=None):
        '''
        :param input: (batch_size, 400)
        :param is_training: whether to use training or inference mode
        :param dropout_keep_prob: Probability for the tf.nn.dropout layer (float in [0, 1)).
        :return:
        '''
        if is_training:
            dropout_keep_prob = 0.5
        if self.rnn_type == "GRU":
            def make_cell():
                cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, name=self.video_type)
                #cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                return cell
        elif self.rnn_type == "LSTM":
            def make_cell():
                cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, name=self.video_type)
                #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                return cell
        else:
            ValueError("invalid rnn type")
            exit()

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(2)], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(2)], state_is_tuple=True)
        #net = tf.reshape(inputs, [-1, 1, 400])
        seq_len = tf.squeeze(seq_len, axis=[1])
        shape = inputs.get_shape().as_list()
        net = tf.reshape(inputs, [self.FLAGS.batch_size, -1, 50176])
        net, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, net, dtype=tf.float32, sequence_length=seq_len)
        net = tf.concat([net[0], net[1]], 2)
        #net = tf.layers.dense(net, 500, tf.nn.relu)
        return net
