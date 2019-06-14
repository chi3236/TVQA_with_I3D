import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
sys.path.append("./kinetics-i3d")
import i3d2
import video_rnn
import text_rnn_cudnn

TOWER_NAME = 'tower'
#def build_graph(FLAGS, rgb_input, flow_input, sub, q, ac, a, rgb_seq_len, flow_seq_len,sub_seq_len, q_seq_len):
def build_graph(FLAGS, vocab_embedding, rgb_input, flow_input, sub, q, a0, a1, a2, a3, a4, a, qid, sub_seq_len, q_seq_len, a0_seq_len, a1_seq_len, a2_seq_len, a3_seq_len, a4_seq_len, prob, text_prob, is_training):

    #with tf.device("/GPU:0"):
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.wd)
    vocab_embedding_tensor = tf.convert_to_tensor(vocab_embedding, dtype=tf.float32)
    sub_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, sub)
    q_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, q)
    a0_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, a0)
    a1_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, a1)
    a2_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, a2)
    a3_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, a3)
    a4_tensor = tf.nn.embedding_lookup(vocab_embedding_tensor, a4)
    sub_tensor = tf.cast(sub_tensor, tf.float32)
    q_tensor = tf.cast(q_tensor, tf.float32)
    a0_tensor = tf.cast(a0_tensor, tf.float32)
    a1_tensor = tf.cast(a1_tensor, tf.float32)
    a2_tensor = tf.cast(a2_tensor, tf.float32)
    a3_tensor = tf.cast(a3_tensor, tf.float32)
    a4_tensor = tf.cast(a4_tensor, tf.float32)
    with tf.variable_scope("RGB"):
        rgb_model = i3d2.InceptionI3d(FLAGS.num_classes, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_input, is_training=is_training[0][0], dropout_keep_prob=prob[0][0])
        #rgb_logits = tf.layers.dense(rgb_logits, 300, activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
        #                             name="rgb_fc")
        #rgb_logits = tf.nn.dropout(rgb_logits, prob[0][0])
        #rgb_logits = tf.expand_dims(rgb_logits, 1)

    #with tf.device("/GPU:1"):
    with tf.variable_scope("Flow"):
        flow_model = i3d2.InceptionI3d(FLAGS.num_classes, spatial_squeeze=True, final_endpoint='Logits')
        flow_logits, _ = flow_model(flow_input, is_training=is_training[0][0], dropout_keep_prob=prob[0][0])
        #flow_logits = tf.layers.dense(flow_logits, 300, activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
        #                             name="flow_fc")
        #flow_logits = tf.nn.dropout(flow_logits, prob[0][0])
        #flow_logits = tf.expand_dims(flow_logits, 1)

    #with tf.device("/GPU:2"):
    #with tf.variable_scope("Video_RNN"):
    #rgb_rnn_model = video_rnn.vRNN("GRU", FLAGS.num_hidden, 'rgb', FLAGS)
    #flow_rnn_model = video_rnn.vRNN("GRU", FLAGS.num_hidden, 'flow', FLAGS)
    #rgb_rnn_logits = rgb_rnn_model.build(rgb_logits, is_training=FLAGS.is_training, seq_len=rgb_seq_len)
    #flow_rnn_logits = flow_rnn_model.build(flow_logits, is_training=FLAGS.is_training, seq_len=flow_seq_len)
    #rgb_rnn_logits = tf.layers.batch_normalization(rgb_rnn_logits)
    #flow_rnn_logits = tf.layers.batch_normalization(flow_rnn_logits)
    #rgb_rnn_logits = tf.reduce_mean(rgb_rnn_logits, axis=1, keepdims=True)
    #flow_rnn_logits = tf.reduce_mean(flow_rnn_logits, axis=1, keepdims=True)
    #rgb_rnn_logits = tf.nn.l2_normalize(rgb_rnn_logits, axis=2)
    #flow_rnn_logits = tf.nn.l2_normalize(flow_rnn_logits, axis=2)
    with tf.variable_scope("Text_LSTM", reuse=tf.AUTO_REUSE):
        '''
        def apply_softmax(elem):
            pass

        def apply_attention(elem):
            previous = 0
            attention_sum_logit = tf.zeros(tf.shape(elem[0]))

            attention_sum_logit = tf.map_fn(lambda x: attention_sum_logit + tf.multiply())
            for i in elem[1]:
                attention_logit = elem[0][previous:i]
                attention_logit = tf.multiply(attention_logit, elem[2])
                attention_logit = tf.nn.softmax(attention_logit)
                attention_logit = tf.reduce_sum(attention_logit, axis=0, keepdims=True)
                if not attention_sum_logit:
                    attention_sum_logit = attention_logit
                else:
                    attention_sum_logit = tf.stack([attention_sum_logit, attention_logit], axis=0)

            return attention_sum_logit
        '''
        text_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'question')
        #question_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'question')
        #answer_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'answer')
        #answer_rnn_model = text_rnn_cudnn.tRNN_answer("GRU", 200, 'answer')
        #subtitle_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'subtitle')
        question_rnn_logits = text_rnn_model.build(q_tensor, dropout_keep_prob=text_prob, seq_len=q_seq_len)
        subtitle_rnn_logits = text_rnn_model.build(sub_tensor, dropout_keep_prob=text_prob, seq_len=sub_seq_len)
        answer0_rnn_logits = text_rnn_model.build(a0_tensor, dropout_keep_prob=text_prob, seq_len=a0_seq_len)
        answer1_rnn_logits = text_rnn_model.build(a1_tensor, dropout_keep_prob=text_prob, seq_len=a1_seq_len)
        answer2_rnn_logits = text_rnn_model.build(a2_tensor, dropout_keep_prob=text_prob, seq_len=a2_seq_len)
        answer3_rnn_logits = text_rnn_model.build(a3_tensor, dropout_keep_prob=text_prob, seq_len=a3_seq_len)
        answer4_rnn_logits = text_rnn_model.build(a4_tensor, dropout_keep_prob=text_prob, seq_len=a4_seq_len)

    with tf.variable_scope("Text_BiDAF_subtitle_question", reuse=tf.AUTO_REUSE):
        #subtitle_attention_ex = tf.expand_dims(subtitle_rnn_logits, 2)
        #subtitle_attention = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
        #                                                             tf.shape(subtitle_rnn_logits)[1],
        #                                                             tf.shape(question_rnn_logits)[1],
        #                                                             FLAGS.num_hidden * 2])
        #question_attention = tf.expand_dims(question_rnn_logits, 1)
        #question_attention = tf.broadcast_to(question_attention, [tf.shape(question_rnn_logits)[0],
        #                                                          tf.shape(subtitle_rnn_logits)[1],
        #                                                          tf.shape(question_rnn_logits)[1],
        #                                                          FLAGS.num_hidden * 2])
        #subtitle_question_mul = tf.multiply(subtitle_attention, question_attention)
        #subtitle_question_concat = tf.concat([subtitle_attention, question_attention, subtitle_question_mul],
        #                                     axis=3)
        #subtitle_question_similarity = tf.layers.dense(subtitle_question_concat, 1, use_bias=False,
        #                                               name='subtitle_question_similarity',
        #                                               kernel_regularizer=regularizer)
        #subtitle_question_similarity = tf.squeeze(subtitle_question_similarity, axis=[3])
        subtitle_question_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(question_rnn_logits, perm=[0, 2, 1]))
        subtitle_question_c2q = tf.nn.softmax(subtitle_question_similarity)
        subtitle_question_c2q = tf.matmul(subtitle_question_c2q, question_rnn_logits)
        #subtitle_question_b = tf.nn.softmax(tf.reduce_max(subtitle_question_similarity, axis=2))
        #subtitle_question_b = tf.expand_dims(subtitle_question_b, 1)
        #subtitle_question_q2c = tf.matmul(subtitle_question_b, subtitle_rnn_logits)
        #subtitle_question_g = tf.concat([subtitle_rnn_logits, subtitle_question_c2q,
        #                                 tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
        #                                 tf.multiply(subtitle_rnn_logits, subtitle_question_q2c)], axis=2)

    with tf.variable_scope("Text_BiDAF_subtitle_answer", reuse=tf.AUTO_REUSE):
        '''
        answer0_attention = tf.expand_dims(answer0_rnn_logits, 1)
        answer1_attention = tf.expand_dims(answer1_rnn_logits, 1)
        answer2_attention = tf.expand_dims(answer2_rnn_logits, 1)
        answer3_attention = tf.expand_dims(answer3_rnn_logits, 1)
        answer4_attention = tf.expand_dims(answer4_rnn_logits, 1)
        answer0_attention = tf.broadcast_to(answer0_attention, [tf.shape(answer0_rnn_logits)[0],
                                                                tf.shape(subtitle_rnn_logits)[1],
                                                                tf.shape(answer0_rnn_logits)[1],
                                                                FLAGS.num_hidden * 2])
        answer1_attention = tf.broadcast_to(answer1_attention, [tf.shape(answer1_rnn_logits)[0],
                                                                tf.shape(subtitle_rnn_logits)[1],
                                                                tf.shape(answer1_rnn_logits)[1],
                                                                FLAGS.num_hidden * 2])
        answer2_attention = tf.broadcast_to(answer2_attention, [tf.shape(answer2_rnn_logits)[0],
                                                                tf.shape(subtitle_rnn_logits)[1],
                                                                tf.shape(answer2_rnn_logits)[1],
                                                                FLAGS.num_hidden * 2])
        answer3_attention = tf.broadcast_to(answer3_attention, [tf.shape(answer3_rnn_logits)[0],
                                                                tf.shape(subtitle_rnn_logits)[1],
                                                                tf.shape(answer3_rnn_logits)[1],
                                                                FLAGS.num_hidden * 2])
        answer4_attention = tf.broadcast_to(answer4_attention, [tf.shape(answer4_rnn_logits)[0],
                                                                tf.shape(subtitle_rnn_logits)[1],
                                                                tf.shape(answer4_rnn_logits)[1],
                                                                FLAGS.num_hidden * 2])
        subtitle_attention0 = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
                                                                      tf.shape(subtitle_rnn_logits)[1],
                                                                      tf.shape(answer0_rnn_logits)[1],
                                                                      FLAGS.num_hidden * 2])
        subtitle_attention1 = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
                                                                      tf.shape(subtitle_rnn_logits)[1],
                                                                      tf.shape(answer1_rnn_logits)[1],
                                                                      FLAGS.num_hidden * 2])
        subtitle_attention2 = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
                                                                      tf.shape(subtitle_rnn_logits)[1],
                                                                      tf.shape(answer2_rnn_logits)[1],
                                                                      FLAGS.num_hidden * 2])
        subtitle_attention3 = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
                                                                      tf.shape(subtitle_rnn_logits)[1],
                                                                      tf.shape(answer3_rnn_logits)[1],
                                                                      FLAGS.num_hidden * 2])
        subtitle_attention4 = tf.broadcast_to(subtitle_attention_ex, [tf.shape(subtitle_rnn_logits)[0],
                                                                      tf.shape(subtitle_rnn_logits)[1],
                                                                      tf.shape(answer4_rnn_logits)[1],
                                                                      FLAGS.num_hidden * 2])
        subtitle_answer0_mul = tf.multiply(subtitle_attention0, answer0_attention)
        subtitle_answer1_mul = tf.multiply(subtitle_attention1, answer1_attention)
        subtitle_answer2_mul = tf.multiply(subtitle_attention2, answer2_attention)
        subtitle_answer3_mul = tf.multiply(subtitle_attention3, answer3_attention)
        subtitle_answer4_mul = tf.multiply(subtitle_attention4, answer4_attention)
        subtitle_answer0_concat = tf.concat([subtitle_attention0, answer0_attention, subtitle_answer0_mul], axis=3)
        subtitle_answer1_concat = tf.concat([subtitle_attention1, answer1_attention, subtitle_answer1_mul], axis=3)
        subtitle_answer2_concat = tf.concat([subtitle_attention2, answer2_attention, subtitle_answer2_mul], axis=3)
        subtitle_answer3_concat = tf.concat([subtitle_attention3, answer3_attention, subtitle_answer3_mul], axis=3)
        subtitle_answer4_concat = tf.concat([subtitle_attention4, answer4_attention, subtitle_answer4_mul], axis=3)
        subtitle_answer0_similarity = tf.layers.dense(subtitle_answer0_concat, 1, use_bias=False,
                                                      name='subtitle_answer_similarity',
                                                      kernel_regularizer=regularizer)
        subtitle_answer0_similarity = tf.squeeze(subtitle_answer0_similarity, axis=[3])
        subtitle_answer1_similarity = tf.layers.dense(subtitle_answer1_concat, 1, use_bias=False,
                                                      name='subtitle_answer_similarity', reuse=True,
                                                      kernel_regularizer=regularizer)
        subtitle_answer1_similarity = tf.squeeze(subtitle_answer1_similarity, axis=[3])
        subtitle_answer2_similarity = tf.layers.dense(subtitle_answer2_concat, 1, use_bias=False,
                                                      name='subtitle_answer_similarity', reuse=True,
                                                      kernel_regularizer=regularizer)
        subtitle_answer2_similarity = tf.squeeze(subtitle_answer2_similarity, axis=[3])
        subtitle_answer3_similarity = tf.layers.dense(subtitle_answer3_concat, 1, use_bias=False,
                                                      name='subtitle_answer_similarity', reuse=True,
                                                      kernel_regularizer=regularizer)
        subtitle_answer3_similarity = tf.squeeze(subtitle_answer3_similarity, axis=[3])
        subtitle_answer4_similarity = tf.layers.dense(subtitle_answer4_concat, 1, use_bias=False,
                                                      name='subtitle_answer_similarity', reuse=True,
                                                      kernel_regularizer=regularizer)
        subtitle_answer4_similarity = tf.squeeze(subtitle_answer4_similarity, axis=[3])
        '''
        subtitle_answer0_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(answer0_rnn_logits, perm=[0, 2, 1]))
        subtitle_answer1_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(answer1_rnn_logits, perm=[0, 2, 1]))
        subtitle_answer2_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(answer2_rnn_logits, perm=[0, 2, 1]))
        subtitle_answer3_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(answer3_rnn_logits, perm=[0, 2, 1]))
        subtitle_answer4_similarity = tf.matmul(subtitle_rnn_logits, tf.transpose(answer4_rnn_logits, perm=[0, 2, 1]))
        subtitle_answer0_c2q = tf.nn.softmax(subtitle_answer0_similarity)
        subtitle_answer0_c2q = tf.matmul(subtitle_answer0_c2q, answer0_rnn_logits)
        subtitle_answer1_c2q = tf.nn.softmax(subtitle_answer1_similarity)
        subtitle_answer1_c2q = tf.matmul(subtitle_answer1_c2q, answer1_rnn_logits)
        subtitle_answer2_c2q = tf.nn.softmax(subtitle_answer2_similarity)
        subtitle_answer2_c2q = tf.matmul(subtitle_answer2_c2q, answer2_rnn_logits)
        subtitle_answer3_c2q = tf.nn.softmax(subtitle_answer3_similarity)
        subtitle_answer3_c2q = tf.matmul(subtitle_answer3_c2q, answer3_rnn_logits)
        subtitle_answer4_c2q = tf.nn.softmax(subtitle_answer4_similarity)
        subtitle_answer4_c2q = tf.matmul(subtitle_answer4_c2q, answer4_rnn_logits)
        #subtitle_answer0_b = tf.nn.softmax(tf.reduce_max(subtitle_answer0_similarity, axis=2))
        #subtitle_answer0_b = tf.expand_dims(subtitle_answer0_b, 1)
        #subtitle_answer1_b = tf.nn.softmax(tf.reduce_max(subtitle_answer1_similarity, axis=2))
        #subtitle_answer1_b = tf.expand_dims(subtitle_answer1_b, 1)
        #subtitle_answer2_b = tf.nn.softmax(tf.reduce_max(subtitle_answer2_similarity, axis=2))
        #subtitle_answer2_b = tf.expand_dims(subtitle_answer2_b, 1)
        #subtitle_answer3_b = tf.nn.softmax(tf.reduce_max(subtitle_answer3_similarity, axis=2))
        #subtitle_answer3_b = tf.expand_dims(subtitle_answer3_b, 1)
        #subtitle_answer4_b = tf.nn.softmax(tf.reduce_max(subtitle_answer4_similarity, axis=2))
        #subtitle_answer4_b = tf.expand_dims(subtitle_answer4_b, 1)
        #subtitle_answer0_q2c = tf.matmul(subtitle_answer0_b, subtitle_rnn_logits)
        #subtitle_answer1_q2c = tf.matmul(subtitle_answer1_b, subtitle_rnn_logits)
        #subtitle_answer2_q2c = tf.matmul(subtitle_answer2_b, subtitle_rnn_logits)
        #subtitle_answer3_q2c = tf.matmul(subtitle_answer3_b, subtitle_rnn_logits)
        #subtitle_answer4_q2c = tf.matmul(subtitle_answer4_b, subtitle_rnn_logits)
        concat_subtitle_query0 = tf.concat([subtitle_rnn_logits, subtitle_question_c2q, subtitle_answer0_c2q,
                                            tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
                                            tf.multiply(subtitle_rnn_logits, subtitle_answer0_c2q)], axis=2)
        concat_subtitle_query1 = tf.concat([subtitle_rnn_logits, subtitle_question_c2q, subtitle_answer1_c2q,
                                            tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
                                            tf.multiply(subtitle_rnn_logits, subtitle_answer1_c2q)], axis=2)
        concat_subtitle_query2 = tf.concat([subtitle_rnn_logits, subtitle_question_c2q, subtitle_answer2_c2q,
                                            tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
                                            tf.multiply(subtitle_rnn_logits, subtitle_answer2_c2q)], axis=2)
        concat_subtitle_query3 = tf.concat([subtitle_rnn_logits, subtitle_question_c2q, subtitle_answer3_c2q,
                                            tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
                                            tf.multiply(subtitle_rnn_logits, subtitle_answer3_c2q)], axis=2)
        concat_subtitle_query4 = tf.concat([subtitle_rnn_logits, subtitle_question_c2q, subtitle_answer4_c2q,
                                            tf.multiply(subtitle_rnn_logits, subtitle_question_c2q),
                                            tf.multiply(subtitle_rnn_logits, subtitle_answer4_c2q)], axis=2)

        #subtitle_answer0_g = tf.concat([subtitle_rnn_logits, subtitle_answer0_c2q,
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer0_c2q),
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer0_q2c)], axis=2)
        #subtitle_answer1_g = tf.concat([subtitle_rnn_logits, subtitle_answer1_c2q,
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer1_c2q),
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer1_q2c)], axis=2)
        #subtitle_answer2_g = tf.concat([subtitle_rnn_logits, subtitle_answer2_c2q,
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer2_c2q),
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer2_q2c)], axis=2)
        #subtitle_answer3_g = tf.concat([subtitle_rnn_logits, subtitle_answer3_c2q,
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer3_c2q),
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer3_q2c)], axis=2)
        #subtitle_answer4_g = tf.concat([subtitle_rnn_logits, subtitle_answer4_c2q,
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer4_c2q),
        #                                tf.multiply(subtitle_rnn_logits, subtitle_answer4_q2c)], axis=2)
    #with tf.device("/GPU:1"):
    with tf.variable_scope("RGB_question_match", reuse=tf.AUTO_REUSE):
        rgb_logits = tf.layers.dense(rgb_logits, 300, name="RGB_question_fc", kernel_regularizer=regularizer,
                                     activation=tf.nn.leaky_relu)
        # question_d = tf.reduce_max(question_rnn_logits, axis=1, keepdims=True)
        # question_rgb = tf.layers.dense(question_d, 300, name="RGB_question_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu)
        # question_rgb = tf.nn.dropout(question_rgb, prob[0][0])
        rgb_question_mul = tf.matmul(rgb_logits, tf.transpose(question_rnn_logits, perm=[0, 2, 1]))
        rgb_question_similarity = tf.nn.softmax(rgb_question_mul)
        rgb_question_masked = tf.matmul(rgb_question_similarity, question_rnn_logits)
        # rgb_question_masked = tf.matmul(rgb_question_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))
        # rgb_question_masked = tf.squeeze(rgb_question_masked, axis=2)

    with tf.variable_scope("RGB_answer_match", reuse=tf.AUTO_REUSE):
        # answer0_d = tf.reduce_max(answer0_rnn_logits, axis=1, keepdims=True)
        # answer1_d = tf.reduce_max(answer1_rnn_logits, axis=1, keepdims=True)
        # answer2_d = tf.reduce_max(answer2_rnn_logits, axis=1, keepdims=True)
        # answer3_d = tf.reduce_max(answer3_rnn_logits, axis=1, keepdims=True)
        # answer4_d = tf.reduce_max(answer4_rnn_logits, axis=1, keepdims=True)
        # answer0_rgb = tf.layers.dense(answer0_d, 300, name="RGB_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu)
        # answer1_rgb = tf.layers.dense(answer1_d, 300, name="RGB_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu, reuse=True)
        # answer2_rgb = tf.layers.dense(answer2_d, 300, name="RGB_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu, reuse=True)
        # answer3_rgb = tf.layers.dense(answer3_d, 300, name="RGB_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu, reuse=True)
        # answer4_rgb = tf.layers.dense(answer4_d, 300, name="RGB_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu, reuse=True)

        # answer0_rgb = tf.nn.dropout(answer0_rgb, prob[0][0])
        # answer1_rgb = tf.nn.dropout(answer1_rgb, prob[0][0])
        # answer2_rgb = tf.nn.dropout(answer2_rgb, prob[0][0])
        # answer3_rgb = tf.nn.dropout(answer3_rgb, prob[0][0])
        # answer4_rgb = tf.nn.dropout(answer4_rgb, prob[0][0])

        rgb_answer0_mul = tf.matmul(rgb_logits, tf.transpose(answer0_rnn_logits, perm=[0, 2, 1]))
        rgb_answer1_mul = tf.matmul(rgb_logits, tf.transpose(answer1_rnn_logits, perm=[0, 2, 1]))
        rgb_answer2_mul = tf.matmul(rgb_logits, tf.transpose(answer2_rnn_logits, perm=[0, 2, 1]))
        rgb_answer3_mul = tf.matmul(rgb_logits, tf.transpose(answer3_rnn_logits, perm=[0, 2, 1]))
        rgb_answer4_mul = tf.matmul(rgb_logits, tf.transpose(answer4_rnn_logits, perm=[0, 2, 1]))
        # rgb_answer1_mul = tf.matmul(tf.transpose(answer1_d, perm=[0, 2, 1]), rgb_logits)
        # rgb_answer2_mul = tf.matmul(tf.transpose(answer2_d, perm=[0, 2, 1]), rgb_logits)
        # rgb_answer3_mul = tf.matmul(tf.transpose(answer3_d, perm=[0, 2, 1]), rgb_logits)
        # rgb_answer4_mul = tf.matmul(tf.transpose(answer4_d, perm=[0, 2, 1]), rgb_logits)

        rgb_answer0_similarity = tf.nn.softmax(rgb_answer0_mul)
        rgb_answer1_similarity = tf.nn.softmax(rgb_answer1_mul)
        rgb_answer2_similarity = tf.nn.softmax(rgb_answer2_mul)
        rgb_answer3_similarity = tf.nn.softmax(rgb_answer3_mul)
        rgb_answer4_similarity = tf.nn.softmax(rgb_answer4_mul)

        rgb_answer0_masked = tf.matmul(rgb_answer0_similarity, answer0_rnn_logits)
        rgb_answer1_masked = tf.matmul(rgb_answer1_similarity, answer1_rnn_logits)
        rgb_answer2_masked = tf.matmul(rgb_answer2_similarity, answer2_rnn_logits)
        rgb_answer3_masked = tf.matmul(rgb_answer3_similarity, answer3_rnn_logits)
        rgb_answer4_masked = tf.matmul(rgb_answer4_similarity, answer4_rnn_logits)
        # rgb_answer0_masked = tf.matmul(rgb_answer0_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))
        # rgb_answer1_masked = tf.matmul(rgb_answer1_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))
        # rgb_answer2_masked = tf.matmul(rgb_answer2_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))
        # rgb_answer3_masked = tf.matmul(rgb_answer3_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))
        # rgb_answer4_masked = tf.matmul(rgb_answer4_similarity, tf.transpose(rgb_logits, perm=[0, 2, 1]))

        # rgb_answer0_masked = tf.squeeze(rgb_answer0_masked, axis=2)
        # rgb_answer1_masked = tf.squeeze(rgb_answer1_masked, axis=2)
        # rgb_answer2_masked = tf.squeeze(rgb_answer2_masked, axis=2)
        # rgb_answer3_masked = tf.squeeze(rgb_answer3_masked, axis=2)
        # rgb_answer4_masked = tf.squeeze(rgb_answer4_masked, axis=2)

    with tf.variable_scope("RGB_question_answer_att_layer", reuse=tf.AUTO_REUSE):
        rgb_question_answer0_att = tf.concat([rgb_logits, rgb_question_masked, rgb_answer0_masked,
                                              tf.multiply(rgb_logits, rgb_question_masked),
                                              tf.multiply(rgb_logits, rgb_answer0_masked)], axis=1)
        rgb_question_answer1_att = tf.concat([rgb_logits, rgb_question_masked, rgb_answer1_masked,
                                              tf.multiply(rgb_logits, rgb_question_masked),
                                              tf.multiply(rgb_logits, rgb_answer1_masked)], axis=1)
        rgb_question_answer2_att = tf.concat([rgb_logits, rgb_question_masked, rgb_answer2_masked,
                                              tf.multiply(rgb_logits, rgb_question_masked),
                                              tf.multiply(rgb_logits, rgb_answer2_masked)], axis=1)
        rgb_question_answer3_att = tf.concat([rgb_logits, rgb_question_masked, rgb_answer3_masked,
                                              tf.multiply(rgb_logits, rgb_question_masked),
                                              tf.multiply(rgb_logits, rgb_answer3_masked)], axis=1)
        rgb_question_answer4_att = tf.concat([rgb_logits, rgb_question_masked, rgb_answer4_masked,
                                              tf.multiply(rgb_logits, rgb_question_masked),
                                              tf.multiply(rgb_logits, rgb_answer4_masked)], axis=1)
        # rgb_question_answer1_att = tf.concat([rgb_question_masked, rgb_answer1_masked, tf.squeeze(question_d, 1), tf.squeeze(answer1_d, 1)], axis=1)
        # rgb_question_answer2_att = tf.concat([rgb_question_masked, rgb_answer2_masked, tf.squeeze(question_d, 1), tf.squeeze(answer2_d, 1)], axis=1)
        # rgb_question_answer3_att = tf.concat([rgb_question_masked, rgb_answer3_masked, tf.squeeze(question_d, 1), tf.squeeze(answer3_d, 1)], axis=1)
        # rgb_question_answer4_att = tf.concat([rgb_question_masked, rgb_answer4_masked, tf.squeeze(question_d, 1), tf.squeeze(answer4_d, 1)], axis=1)
        # rgb_question_answer0_att = tf.matmul(rgb_question_masked, tf.transpose(rgb_answer0_masked, perm=[0, 2, 1]))
        # rgb_question_answer1_att = tf.matmul(rgb_question_masked, tf.transpose(rgb_answer1_masked, perm=[0, 2, 1]))
        # rgb_question_answer2_att = tf.matmul(rgb_question_masked, tf.transpose(rgb_answer2_masked, perm=[0, 2, 1]))
        # rgb_question_answer3_att = tf.matmul(rgb_question_masked, tf.transpose(rgb_answer3_masked, perm=[0, 2, 1]))
        # rgb_question_answer4_att = tf.matmul(rgb_question_masked, tf.transpose(rgb_answer4_masked, perm=[0, 2, 1]))

        '''
        rgb_question_answer0_cnn = tf.layers.conv1d(rgb_question_answer0_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    kernel_initializer=tf.initializers.random_normal, name="rgb_conv")
        rgb_question_answer1_cnn = tf.layers.conv1d(rgb_question_answer1_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    name="rgb_conv", reuse=True)
        rgb_question_answer2_cnn = tf.layers.conv1d(rgb_question_answer2_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    name="rgb_conv", reuse=True)
        rgb_question_answer3_cnn = tf.layers.conv1d(rgb_question_answer3_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    name="rgb_conv", reuse=True)
        rgb_question_answer4_cnn = tf.layers.conv1d(rgb_question_answer4_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    name="rgb_conv", reuse=True)

        rgb_question_answer0_cnn = tf.nn.dropout(rgb_question_answer0_cnn, prob[0][0])
        rgb_question_answer1_cnn = tf.nn.dropout(rgb_question_answer1_cnn, prob[0][0])
        rgb_question_answer2_cnn = tf.nn.dropout(rgb_question_answer2_cnn, prob[0][0])
        rgb_question_answer3_cnn = tf.nn.dropout(rgb_question_answer3_cnn, prob[0][0])
        rgb_question_answer4_cnn = tf.nn.dropout(rgb_question_answer4_cnn, prob[0][0])

        rgb_question_answer0_cnn = tf.squeeze(rgb_question_answer0_cnn, 2)
        rgb_question_answer1_cnn = tf.squeeze(rgb_question_answer1_cnn, 2)
        rgb_question_answer2_cnn = tf.squeeze(rgb_question_answer2_cnn, 2)
        rgb_question_answer3_cnn = tf.squeeze(rgb_question_answer3_cnn, 2)
        rgb_question_answer4_cnn = tf.squeeze(rgb_question_answer4_cnn, 2)
        '''

        rgb_question_answer0_fc = tf.layers.dense(rgb_question_answer0_att, 1500, activation=tf.nn.leaky_relu,
                                                  kernel_regularizer=regularizer, name='rgb_mask_fc')
        rgb_question_answer1_fc = tf.layers.dense(rgb_question_answer1_att, 1500, activation=tf.nn.leaky_relu,
                                                  kernel_regularizer=regularizer, name='rgb_mask_fc', reuse=True)
        rgb_question_answer2_fc = tf.layers.dense(rgb_question_answer2_att, 1500, activation=tf.nn.leaky_relu,
                                                  kernel_regularizer=regularizer, name='rgb_mask_fc', reuse=True)
        rgb_question_answer3_fc = tf.layers.dense(rgb_question_answer3_att, 1500, activation=tf.nn.leaky_relu,
                                                  kernel_regularizer=regularizer, name='rgb_mask_fc', reuse=True)
        rgb_question_answer4_fc = tf.layers.dense(rgb_question_answer4_att, 1500, activation=tf.nn.leaky_relu,
                                                  kernel_regularizer=regularizer, name='rgb_mask_fc', reuse=True)

        rgb_question_answer0_fc = tf.nn.dropout(rgb_question_answer0_fc, prob[0][0])
        rgb_question_answer1_fc = tf.nn.dropout(rgb_question_answer1_fc, prob[0][0])
        rgb_question_answer2_fc = tf.nn.dropout(rgb_question_answer2_fc, prob[0][0])
        rgb_question_answer3_fc = tf.nn.dropout(rgb_question_answer3_fc, prob[0][0])
        rgb_question_answer4_fc = tf.nn.dropout(rgb_question_answer4_fc, prob[0][0])

        rgb_question_answer0_fc = tf.reduce_max(rgb_question_answer0_fc, 1)
        rgb_question_answer1_fc = tf.reduce_max(rgb_question_answer1_fc, 1)
        rgb_question_answer2_fc = tf.reduce_max(rgb_question_answer2_fc, 1)
        rgb_question_answer3_fc = tf.reduce_max(rgb_question_answer3_fc, 1)
        rgb_question_answer4_fc = tf.reduce_max(rgb_question_answer4_fc, 1)

    with tf.variable_scope("Flow_question_match", reuse=tf.AUTO_REUSE):
        flow_logits = tf.layers.dense(flow_logits, 300, name="Flow_question_fc", kernel_regularizer=regularizer,
                                      activation=tf.nn.leaky_relu)
        # question_d = tf.reduce_max(question_rnn_logits, axis=1, keepdims=True)
        # question_flow = tf.layers.dense(question_d, 300, name="flow_question_dense", kernel_regularizer=regularizer,
        #                               activation=tf.nn.leaky_relu)
        # question_flow = tf.nn.dropout(question_flow, prob[0][0])
        flow_question_mul = tf.matmul(flow_logits, tf.transpose(question_rnn_logits, perm=[0, 2, 1]))
        flow_question_similarity = tf.nn.softmax(flow_question_mul)
        flow_question_masked = tf.matmul(flow_question_similarity, question_rnn_logits)
        # flow_question_masked = tf.matmul(flow_question_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))
        # flow_question_masked = tf.squeeze(flow_question_masked, axis=2)

    with tf.variable_scope("Flow_answer_match", reuse=tf.AUTO_REUSE):
        # answer0_d = tf.reduce_max(answer0_rnn_logits, axis=1, keepdims=True)
        # answer1_d = tf.reduce_max(answer1_rnn_logits, axis=1, keepdims=True)
        # answer2_d = tf.reduce_max(answer2_rnn_logits, axis=1, keepdims=True)
        # answer3_d = tf.reduce_max(answer3_rnn_logits, axis=1, keepdims=True)
        # answer4_d = tf.reduce_max(answer4_rnn_logits, axis=1, keepdims=True)
        # answer0_flow = tf.layers.dense(answer0_d, 300, name="flow_answer_dense", kernel_regularizer=regularizer,
        #                             activation=tf.nn.leaky_relu)
        # answer1_flow = tf.layers.dense(answer1_d, 300, name="flow_answer_dense", kernel_regularizer=regularizer,
        #                               activation=tf.nn.leaky_relu)
        # answer2_flow = tf.layers.dense(answer2_d, 300, name="flow_answer_dense", kernel_regularizer=regularizer,
        #                               activation=tf.nn.leaky_relu)
        # answer3_flow = tf.layers.dense(answer3_d, 300, name="flow_answer_dense", kernel_regularizer=regularizer,
        #                               activation=tf.nn.leaky_relu)
        # answer4_flow = tf.layers.dense(answer4_d, 300, name="flow_answer_dense", kernel_regularizer=regularizer,
        #                               activation=tf.nn.leaky_relu)

        # answer0_flow = tf.nn.dropout(answer0_flow, prob[0][0])
        # answer1_flow = tf.nn.dropout(answer1_flow, prob[0][0])
        # answer2_flow = tf.nn.dropout(answer2_flow, prob[0][0])
        # answer3_flow = tf.nn.dropout(answer3_flow, prob[0][0])
        # answer4_flow = tf.nn.dropout(answer4_flow, prob[0][0])

        flow_answer0_mul = tf.matmul(flow_logits, tf.transpose(answer0_rnn_logits, perm=[0, 2, 1]))
        flow_answer1_mul = tf.matmul(flow_logits, tf.transpose(answer1_rnn_logits, perm=[0, 2, 1]))
        flow_answer2_mul = tf.matmul(flow_logits, tf.transpose(answer2_rnn_logits, perm=[0, 2, 1]))
        flow_answer3_mul = tf.matmul(flow_logits, tf.transpose(answer3_rnn_logits, perm=[0, 2, 1]))
        flow_answer4_mul = tf.matmul(flow_logits, tf.transpose(answer4_rnn_logits, perm=[0, 2, 1]))
        # flow_answer0_mul = tf.matmul(tf.transpose(answer0_d, [0, 2, 1]), flow_logits)
        # flow_answer1_mul = tf.matmul(tf.transpose(answer1_d, [0, 2, 1]), flow_logits)
        # flow_answer2_mul = tf.matmul(tf.transpose(answer2_d, [0, 2, 1]), flow_logits)
        # flow_answer3_mul = tf.matmul(tf.transpose(answer3_d, [0, 2, 1]), flow_logits)
        # flow_answer4_mul = tf.matmul(tf.transpose(answer4_d, [0, 2, 1]), flow_logits)

        flow_answer0_similarity = tf.nn.softmax(flow_answer0_mul)
        flow_answer1_similarity = tf.nn.softmax(flow_answer1_mul)
        flow_answer2_similarity = tf.nn.softmax(flow_answer2_mul)
        flow_answer3_similarity = tf.nn.softmax(flow_answer3_mul)
        flow_answer4_similarity = tf.nn.softmax(flow_answer4_mul)

        flow_answer0_masked = tf.matmul(flow_answer0_similarity, answer0_rnn_logits)
        flow_answer1_masked = tf.matmul(flow_answer1_similarity, answer1_rnn_logits)
        flow_answer2_masked = tf.matmul(flow_answer2_similarity, answer2_rnn_logits)
        flow_answer3_masked = tf.matmul(flow_answer3_similarity, answer3_rnn_logits)
        flow_answer4_masked = tf.matmul(flow_answer4_similarity, answer4_rnn_logits)

        # flow_answer0_mask = tf.matmul(flow_answer0_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))
        # flow_answer1_mask = tf.matmul(flow_answer1_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))
        # flow_answer2_mask = tf.matmul(flow_answer2_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))
        # flow_answer3_mask = tf.matmul(flow_answer3_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))
        # flow_answer4_mask = tf.matmul(flow_answer4_similarity, tf.transpose(flow_logits, perm=[0, 2, 1]))

        # flow_answer0_mask = tf.squeeze(flow_answer0_mask, axis=2)
        # flow_answer1_mask = tf.squeeze(flow_answer1_mask, axis=2)
        # flow_answer2_mask = tf.squeeze(flow_answer2_mask, axis=2)
        # flow_answer3_mask = tf.squeeze(flow_answer3_mask, axis=2)
        # flow_answer4_mask = tf.squeeze(flow_answer4_mask, axis=2)

    with tf.variable_scope("Flow_question_answer_att_layer", reuse=tf.AUTO_REUSE):
        flow_question_answer0_att = tf.concat([flow_logits, flow_question_masked, flow_answer0_masked,
                                               tf.multiply(flow_logits, flow_question_masked),
                                               tf.multiply(flow_logits, flow_answer0_masked)], axis=1)
        flow_question_answer1_att = tf.concat([flow_logits, flow_question_masked, flow_answer1_masked,
                                               tf.multiply(flow_logits, flow_question_masked),
                                               tf.multiply(flow_logits, flow_answer1_masked)], axis=1)
        flow_question_answer2_att = tf.concat([flow_logits, flow_question_masked, flow_answer2_masked,
                                               tf.multiply(flow_logits, flow_question_masked),
                                               tf.multiply(flow_logits, flow_answer2_masked)], axis=1)
        flow_question_answer3_att = tf.concat([flow_logits, flow_question_masked, flow_answer3_masked,
                                               tf.multiply(flow_logits, flow_question_masked),
                                               tf.multiply(flow_logits, flow_answer3_masked)], axis=1)
        flow_question_answer4_att = tf.concat([flow_logits, flow_question_masked, flow_answer4_masked,
                                               tf.multiply(flow_logits, flow_question_masked),
                                               tf.multiply(flow_logits, flow_answer4_masked)], axis=1)
        # flow_question_answer0_att = tf.concat([flow_question_masked, flow_answer0_mask], axis=1)
        # flow_question_answer1_att = tf.concat([flow_question_masked, flow_answer1_mask], axis=1)
        # flow_question_answer2_att = tf.concat([flow_question_masked, flow_answer2_mask], axis=1)
        # flow_question_answer3_att = tf.concat([flow_question_masked, flow_answer3_mask], axis=1)
        # flow_question_answer4_att = tf.concat([flow_question_masked, flow_answer4_mask], axis=1)
        # flow_question_answer0_att = tf.matmul(flow_question_masked, tf.transpose(flow_answer0_mask, perm=[0, 2, 1]))
        # flow_question_answer1_att = tf.matmul(flow_question_masked, tf.transpose(flow_answer1_mask, perm=[0, 2, 1]))
        # flow_question_answer2_att = tf.matmul(flow_question_masked, tf.transpose(flow_answer2_mask, perm=[0, 2, 1]))
        # flow_question_answer3_att = tf.matmul(flow_question_masked, tf.transpose(flow_answer3_mask, perm=[0, 2, 1]))
        # flow_question_answer4_att = tf.matmul(flow_question_masked, tf.transpose(flow_answer4_mask, perm=[0, 2, 1]))

        '''
        flow_question_answer0_cnn = tf.layers.conv1d(flow_question_answer0_att, 1, 1, padding='same',
                                                    activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                    kernel_initializer=tf.initializers.random_normal, name="flow_conv")
        flow_question_answer1_cnn = tf.layers.conv1d(flow_question_answer1_att, 1, 1, padding='same',
                                                     activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                     name="flow_conv", reuse=True)
        flow_question_answer2_cnn = tf.layers.conv1d(flow_question_answer2_att, 1, 1, padding='same',
                                                     activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                     name="flow_conv", reuse=True)
        flow_question_answer3_cnn = tf.layers.conv1d(flow_question_answer3_att, 1, 1, padding='same',
                                                     activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                     name="flow_conv", reuse=True)
        flow_question_answer4_cnn = tf.layers.conv1d(flow_question_answer4_att, 1, 1, padding='same',
                                                     activation=tf.nn.leaky_relu, kernel_regularizer=regularizer,
                                                     name="flow_conv", reuse=True)

        flow_question_answer0_cnn = tf.nn.dropout(flow_question_answer0_cnn, prob[0][0])
        flow_question_answer1_cnn = tf.nn.dropout(flow_question_answer1_cnn, prob[0][0])
        flow_question_answer2_cnn = tf.nn.dropout(flow_question_answer2_cnn, prob[0][0])
        flow_question_answer3_cnn = tf.nn.dropout(flow_question_answer3_cnn, prob[0][0])
        flow_question_answer4_cnn = tf.nn.dropout(flow_question_answer4_cnn, prob[0][0])

        flow_question_answer0_cnn = tf.squeeze(flow_question_answer0_cnn, 2)
        flow_question_answer1_cnn = tf.squeeze(flow_question_answer1_cnn, 2)
        flow_question_answer2_cnn = tf.squeeze(flow_question_answer2_cnn, 2)
        flow_question_answer3_cnn = tf.squeeze(flow_question_answer3_cnn, 2)
        flow_question_answer4_cnn = tf.squeeze(flow_question_answer4_cnn, 2)
        '''
        flow_question_answer0_fc = tf.layers.dense(flow_question_answer0_att, 1500, activation=tf.nn.leaky_relu,
                                                   kernel_regularizer=regularizer, name="Flow_mask_fc")
        flow_question_answer1_fc = tf.layers.dense(flow_question_answer1_att, 1500, activation=tf.nn.leaky_relu,
                                                   kernel_regularizer=regularizer, name="Flow_mask_fc", reuse=True)
        flow_question_answer2_fc = tf.layers.dense(flow_question_answer2_att, 1500, activation=tf.nn.leaky_relu,
                                                   kernel_regularizer=regularizer, name="Flow_mask_fc", reuse=True)
        flow_question_answer3_fc = tf.layers.dense(flow_question_answer3_att, 1500, activation=tf.nn.leaky_relu,
                                                   kernel_regularizer=regularizer, name="Flow_mask_fc", reuse=True)
        flow_question_answer4_fc = tf.layers.dense(flow_question_answer4_att, 1500, activation=tf.nn.leaky_relu,
                                                   kernel_regularizer=regularizer, name="Flow_mask_fc", reuse=True)

        flow_question_answer0_fc = tf.nn.dropout(flow_question_answer0_fc, keep_prob=prob[0][0])
        flow_question_answer1_fc = tf.nn.dropout(flow_question_answer1_fc, keep_prob=prob[0][0])
        flow_question_answer2_fc = tf.nn.dropout(flow_question_answer2_fc, keep_prob=prob[0][0])
        flow_question_answer3_fc = tf.nn.dropout(flow_question_answer3_fc, keep_prob=prob[0][0])
        flow_question_answer4_fc = tf.nn.dropout(flow_question_answer4_fc, keep_prob=prob[0][0])

        flow_question_answer0_fc = tf.reduce_max(flow_question_answer0_fc, 1)
        flow_question_answer1_fc = tf.reduce_max(flow_question_answer1_fc, 1)
        flow_question_answer2_fc = tf.reduce_max(flow_question_answer2_fc, 1)
        flow_question_answer3_fc = tf.reduce_max(flow_question_answer3_fc, 1)
        flow_question_answer4_fc = tf.reduce_max(flow_question_answer4_fc, 1)

        # flow_question_answer0_fc = tf.squeeze(flow_question_answer0_fc, 2)
        # flow_question_answer1_fc = tf.squeeze(flow_question_answer1_fc, 2)
        # flow_question_answer2_fc = tf.squeeze(flow_question_answer2_fc, 2)
        # flow_question_answer3_fc = tf.squeeze(flow_question_answer3_fc, 2)
        # flow_question_answer4_fc = tf.squeeze(flow_question_answer4_fc, 2)

    with tf.variable_scope("video_fc", reuse=tf.AUTO_REUSE):
        rgb_answer0_g_logits = tf.layers.dense(rgb_question_answer0_fc, 1, name="RGB_fc",
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        rgb_answer1_g_logits = tf.layers.dense(rgb_question_answer1_fc, 1, name="RGB_fc",
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        rgb_answer2_g_logits = tf.layers.dense(rgb_question_answer2_fc, 1, name="RGB_fc",
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        rgb_answer3_g_logits = tf.layers.dense(rgb_question_answer3_fc, 1, name="RGB_fc",
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        rgb_answer4_g_logits = tf.layers.dense(rgb_question_answer4_fc, 1, name="RGB_fc",
                                               kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)

        rgb_answer0_g_logits = tf.nn.dropout(rgb_answer0_g_logits, prob[0][0])
        rgb_answer1_g_logits = tf.nn.dropout(rgb_answer1_g_logits, prob[0][0])
        rgb_answer2_g_logits = tf.nn.dropout(rgb_answer2_g_logits, prob[0][0])
        rgb_answer3_g_logits = tf.nn.dropout(rgb_answer3_g_logits, prob[0][0])
        rgb_answer4_g_logits = tf.nn.dropout(rgb_answer4_g_logits, prob[0][0])

        flow_answer0_g_logits = tf.layers.dense(flow_question_answer0_fc, 1, name="Flow_fc",
                                                kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        flow_answer1_g_logits = tf.layers.dense(flow_question_answer1_fc, 1, name="Flow_fc",
                                                kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        flow_answer2_g_logits = tf.layers.dense(flow_question_answer2_fc, 1, name="Flow_fc",
                                                kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        flow_answer3_g_logits = tf.layers.dense(flow_question_answer3_fc, 1, name="Flow_fc",
                                                kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)
        flow_answer4_g_logits = tf.layers.dense(flow_question_answer4_fc, 1, name="Flow_fc",
                                                kernel_regularizer=regularizer, activation=tf.nn.leaky_relu, reuse=True)

        flow_answer0_g_logits = tf.nn.dropout(flow_answer0_g_logits, prob[0][0])
        flow_answer1_g_logits = tf.nn.dropout(flow_answer1_g_logits, prob[0][0])
        flow_answer2_g_logits = tf.nn.dropout(flow_answer2_g_logits, prob[0][0])
        flow_answer3_g_logits = tf.nn.dropout(flow_answer3_g_logits, prob[0][0])
        flow_answer4_g_logits = tf.nn.dropout(flow_answer4_g_logits, prob[0][0])

    with tf.variable_scope("post_rnn", reuse=tf.AUTO_REUSE):
        #subtitle_question_g_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, "subtitle_question_g")
        subtitle_answer_g_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden * 5, "subtitle_answer_g")
        #subtitle_question_g_rnn_logits = subtitle_question_g_rnn_model.build(subtitle_question_g,
        #                                                                     is_training=FLAGS.is_training,
        #                                                                     seq_len=sub_seq_len)
        subtitle_answer0_g_rnn_logits = subtitle_answer_g_rnn_model.build(concat_subtitle_query0,
                                                                          dropout_keep_prob=text_prob,
                                                                          seq_len=sub_seq_len)
        subtitle_answer1_g_rnn_logits = subtitle_answer_g_rnn_model.build(concat_subtitle_query1,
                                                                          dropout_keep_prob=text_prob,
                                                                          seq_len=sub_seq_len)
        subtitle_answer2_g_rnn_logits = subtitle_answer_g_rnn_model.build(concat_subtitle_query2,
                                                                          dropout_keep_prob=text_prob,
                                                                          seq_len=sub_seq_len)
        subtitle_answer3_g_rnn_logits = subtitle_answer_g_rnn_model.build(concat_subtitle_query3,
                                                                          dropout_keep_prob=text_prob,
                                                                          seq_len=sub_seq_len)
        subtitle_answer4_g_rnn_logits = subtitle_answer_g_rnn_model.build(concat_subtitle_query4,
                                                                          dropout_keep_prob=text_prob,
                                                                          seq_len=sub_seq_len)
        #subtitle_question_g_rnn_logits = tf.reduce_max(subtitle_question_g_rnn_logits, axis=1)
        #subtitle_question_g_logits = tf.layers.dense(subtitle_question_g_rnn_logits, 1,
        #                                             name="subtitle_question_g_logits")
        subtitle_answer0_g_rnn_logits = tf.reduce_max(subtitle_answer0_g_rnn_logits, axis=1)
        subtitle_answer1_g_rnn_logits = tf.reduce_max(subtitle_answer1_g_rnn_logits, axis=1)
        subtitle_answer2_g_rnn_logits = tf.reduce_max(subtitle_answer2_g_rnn_logits, axis=1)
        subtitle_answer3_g_rnn_logits = tf.reduce_max(subtitle_answer3_g_rnn_logits, axis=1)
        subtitle_answer4_g_rnn_logits = tf.reduce_max(subtitle_answer4_g_rnn_logits, axis=1)

        subtitle_answer0_g_logits = tf.layers.dense(subtitle_answer0_g_rnn_logits, 1,
                                                    name="subtitle_answer_g_logits",
                                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        subtitle_answer1_g_logits = tf.layers.dense(subtitle_answer1_g_rnn_logits, 1,
                                                    name="subtitle_answer_g_logits", reuse=True,
                                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        subtitle_answer2_g_logits = tf.layers.dense(subtitle_answer2_g_rnn_logits, 1,
                                                    name="subtitle_answer_g_logits", reuse=True,
                                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        subtitle_answer3_g_logits = tf.layers.dense(subtitle_answer3_g_rnn_logits, 1,
                                                    name="subtitle_answer_g_logits", reuse=True,
                                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        subtitle_answer4_g_logits = tf.layers.dense(subtitle_answer4_g_rnn_logits, 1,
                                                    name="subtitle_answer_g_logits", reuse=True,
                                                    kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)

        #subtitle_answer_g_concat = tf.concat([subtitle_answer0_g_rnn_logits, subtitle_answer1_g_rnn_logits,
        #                                      subtitle_answer2_g_rnn_logits, subtitle_answer3_g_rnn_logits,
        #                                      subtitle_answer4_g_rnn_logits], axis=1)

        #subtitle_answer_embed = tf.layers.dense(subtitle_answer_g_concat, 5, kernel_regularizer=regularizer,
        #                                        activation=tf.nn.leaky_relu, name="text_classifier")

        '''
        subtitle_answer0_g_logits = tf.layers.dropout(subtitle_answer0_g_logits, training=is_training)
        subtitle_answer1_g_logits = tf.layers.dropout(subtitle_answer1_g_logits, training=is_training)
        subtitle_answer2_g_logits = tf.layers.dropout(subtitle_answer2_g_logits, training=is_training)
        subtitle_answer3_g_logits = tf.layers.dropout(subtitle_answer3_g_logits, training=is_training)
        subtitle_answer4_g_logits = tf.layers.dropout(subtitle_answer4_g_logits, training=is_training)
        '''
        subtitle_answer0_g_logits = tf.nn.dropout(subtitle_answer0_g_logits, text_prob[0][0])
        subtitle_answer1_g_logits = tf.nn.dropout(subtitle_answer1_g_logits, text_prob[0][0])
        subtitle_answer2_g_logits = tf.nn.dropout(subtitle_answer2_g_logits, text_prob[0][0])
        subtitle_answer3_g_logits = tf.nn.dropout(subtitle_answer3_g_logits, text_prob[0][0])
        subtitle_answer4_g_logits = tf.nn.dropout(subtitle_answer4_g_logits, text_prob[0][0])

    with tf.variable_scope("Embed"):
        #subtitle_question_embed = tf.concat([subtitle_question_g_logits, subtitle_question_g_logits,
        #                                            subtitle_question_g_logits, subtitle_question_g_logits,
        #                                            subtitle_question_g_logits], axis=1)
        subtitle_answer_embed = tf.concat([subtitle_answer0_g_logits, subtitle_answer1_g_logits,
                                           subtitle_answer2_g_logits, subtitle_answer3_g_logits,
                                           subtitle_answer4_g_logits], axis=1)
        subtitle_text_embed = subtitle_answer_embed

        rgb_answer_embed = tf.concat([rgb_answer0_g_logits, rgb_answer1_g_logits, rgb_answer2_g_logits,
                                      rgb_answer3_g_logits, rgb_answer4_g_logits], axis=1)

        flow_answer_embed = tf.concat([flow_answer0_g_logits, flow_answer1_g_logits, flow_answer2_g_logits,
                                       flow_answer3_g_logits, flow_answer4_g_logits], axis=1)
    '''
    with tf.variable_scope("RGB_text_embed"):
        rgb_rnn_logits = tf.reshape(tf.cast(rgb_rnn_logits, tf.float64), [-1, FLAGS.num_hidden*2, 1])
        rgb_question_embed_logits = tf.matmul(question_rnn_logits, rgb_rnn_logits)
        rgb_question_embed_logits = tf.concat([rgb_question_embed_logits, rgb_question_embed_logits,
                                              rgb_question_embed_logits, rgb_question_embed_logits,
                                              rgb_question_embed_logits], axis=1)
        rgb_answer_embed_logits = tf.matmul(answer_rnn_logits, rgb_rnn_logits)
        rgb_text_embed = rgb_question_embed_logits + rgb_answer_embed_logits

    with tf.variable_scope("flow_text_embed"):
        flow_rnn_logits = tf.reshape(tf.cast(flow_rnn_logits, tf.float64), [-1, FLAGS.num_hidden*2, 1])
        flow_question_embed_logits = tf.matmul(question_rnn_logits, flow_rnn_logits)
        flow_question_embed_logits = tf.concat([flow_question_embed_logits, flow_question_embed_logits,
                                                 flow_question_embed_logits, flow_question_embed_logits,
                                                 flow_question_embed_logits], axis=1)
        flow_answer_embed_logits = tf.matmul(answer_rnn_logits, flow_rnn_logits)
        flow_text_embed = flow_question_embed_logits + flow_answer_embed_logits
    '''
    #with tf.device("/GPU:1"):
    with tf.variable_scope("prediction"):

        def loss_calc(elem):
            correct = elem[0][tf.argmax(elem[1])]

            #loss1 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][0])
            #loss2 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][1])
            #loss3 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][2])
            #loss4 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][3])
            #loss5 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][4])
            #not_loss = FLAGS.margin
            loss1 = tf.exp(elem[0][0] - correct)
            loss2 = tf.exp(elem[0][1] - correct)
            loss3 = tf.exp(elem[0][2] - correct)
            loss4 = tf.exp(elem[0][3] - correct)
            loss5 = tf.exp(elem[0][4] - correct)
            #return loss1 + loss2 + loss3 + loss4 + loss5 - not_loss
            return tf.log(loss1 + loss2 + loss3 + loss4 + loss5)

        #video_logits = tf.scalar_mul(tf.constant(0.3), rgb_answer_embed + flow_answer_embed)
        total_logits = tf.nn.softmax(subtitle_text_embed, axis=1) + tf.scalar_mul(0.2, tf.nn.softmax(rgb_answer_embed, axis=1)) + tf.scalar_mul(0.8, tf.nn.softmax(flow_answer_embed, axis=1))
        #train_var_list = []
        #reg_var_list = []
        #for v in tf.trainable_variables():
        #    if (v.name.startswith(u"RGB") or v.name.startswith(u"Flow") or v.name.startswith(u"video")):
        #        train_var_list.append(v)
            #if not ('bias' in v.name) and ('bidirectional_rnn' in v.name):
            #    reg_var_list.append(v)
        #zero_tensor = tf.zeros([FLAGS.batch_size, 5], tf.int32)
        #bool_mask = tf.equal(zero_tensor, a)
        #margin_tensor = tf.constant(0.2, shape=[FLAGS.batch_size])
        #ranking_pos = tf.gather(total_logits, tf.argmax(a, axis=1), axis=1)
        #ranking_pos = tf.reduce_sum(ranking_pos, axis=1)
        #ranking_neg = tf.boolean_mask(total_logits, bool_mask)
        #ranking_neg = tf.reshape(ranking_neg, [-1, 4])
        #ranking_neg = tf.reduce_sum(ranking_neg, axis=1)
        #zero_tensor2 = tf.zeros([FLAGS.batch_size], tf.float32)
        #loss = tf.map_fn(loss_calc, (total_logits, a), dtype=tf.float32)

        #video_loss = tf.map_fn(loss_calc, (video_logits, a), dtype=tf.float32)
        #loss = tf.maximum(zero_tensor2, margin_tensor - ranking_pos + ranking_neg)
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #cost = tf.reduce_mean(loss)
        #cost = tf.reduce_mean(video_loss) + tf.cast(tf.contrib.layers.apply_regularization(regularizer, reg_losses),
        #                                      tf.float32)

        #tf.add_to_collection('losses', cost)
        #optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        #op = optimizer.minimize(cost, var_list=train_var_list)
        comparison = tf.equal(tf.argmax(total_logits, axis=1), tf.argmax(a, axis=1))
        accuracy = tf.reduce_mean(tf.cast(comparison, tf.float32), name="accuracy")
        #loss_summary = tf.summary.scalar("loss", cost)
        #accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        #summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        #train_var_list = []
        #for v in tf.trainable_variables():
        #    if v.name.startswith(u"RGB") or v.name.startswith(u"Flow"):
        #        train_var_list.append(v)
        '''
        prediction = tf.nn.softmax(total_logits, axis=1)
        train_var_list = []
        for v in tf.trainable_variables():
            if not (v.name.startswith(u"RGB") or v.name.startswith(u"Flow")):
                train_var_list.append(v)
        #a = 3
        #a = tf.squeeze(a, axis=[1])
        #cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(a, tf.float64) * tf.log(prediction), 1))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=a, logits=total_logits))
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
        op = optimizer.minimize(cost, var_list=train_var_list)

        comparison = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(a, axis=1))
        accuracy = tf.reduce_mean(tf.cast(comparison, tf.float64), name="accuracy")

        loss_summary = tf.summary.scalar("loss", cost)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        '''
    return accuracy, qid, comparison
