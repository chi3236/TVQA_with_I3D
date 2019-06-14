import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
sys.path.append("./kinetics-i3d")
import i3d
import video_rnn
import text_rnn_cudnn


#def build_graph(FLAGS, rgb_input, flow_input, sub, q, ac, a, rgb_seq_len, flow_seq_len,sub_seq_len, q_seq_len):
def build_graph(FLAGS, sub, q, a0, a1, a2, a3, a4, a, sub_seq_len, q_seq_len, a0_seq_len, a1_seq_len, a2_seq_len, a3_seq_len, a4_seq_len):

    with tf.device("/GPU:1"):
        #with tf.variable_scope("RGB"):
            #rgb_model = i3d.InceptionI3d(FLAGS.num_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
            #rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    #with tf.device("/GPU:1"):
        #with tf.variable_scope("Flow"):
            #flow_model = i3d.InceptionI3d(FLAGS.num_classes, spatial_squeeze=True, final_endpoint='Mixed_5c')
            #flow_logits, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)

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

            question_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'question')
            answer_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'answer')
            #answer_rnn_model = text_rnn.tRNN_answer("GRU", 200, 'answer')
            subtitle_rnn_model = text_rnn_cudnn.tRNN("LSTM", FLAGS.num_hidden, 'subtitle')
            question_rnn_logits = question_rnn_model.build(q, is_training=FLAGS.is_training, seq_len=q_seq_len)
            subtitle_rnn_logits = subtitle_rnn_model.build(sub, is_training=FLAGS.is_training, seq_len=sub_seq_len)
            subtitle_context_vector = tf.get_variable(name='subtitle_context_vector', shape=[FLAGS.num_hidden * 2], trainable=True,
                                             dtype=tf.float32)
            question_context_vector = tf.get_variable(name='question_context_vector', shape=[FLAGS.num_hidden * 2], trainable=True,
                                             dtype=tf.float32)
            answer_context_vector = tf.get_variable(name='answer_context_vector', shape=[FLAGS.num_hidden * 2], trainable=True,
                                             dtype=tf.float32)
            subtitle_attention = tf.layers.dense(subtitle_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                  name="subtitle_attention_layer")
            subtitle_attention = tf.multiply(subtitle_attention, subtitle_context_vector)
            subtitle_attention = tf.nn.l2_normalize(subtitle_attention, axis=1)
            subtitle_rnn_logits = tf.multiply(subtitle_rnn_logits, subtitle_attention)
            question_attention = tf.layers.dense(question_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                  name="question_attention_layer")
            question_attention = tf.multiply(question_attention, question_context_vector)
            question_attention = tf.nn.l2_normalize(question_attention, axis=1)
            question_rnn_logits = tf.multiply(question_rnn_logits, question_attention)
            #subtitle_word_attention_logits = tf.map_fn(apply_attention, (subtitle_rnn_logits, sub_eos, word_attention_context_vector), dtype=tf.float64)
            question_rnn_logits = tf.reduce_sum(question_rnn_logits, axis=1, keepdims=True)
            subtitle_rnn_logits = tf.reduce_sum(subtitle_rnn_logits, axis=1, keepdims=True)
            #answer_rnn_logits = answer_rnn_model.build(ac, is_training=FLAGS.is_training)
            answer0_rnn_logits = answer_rnn_model.build(a0, is_training=FLAGS.is_training, seq_len=a0_seq_len)
            answer1_rnn_logits = answer_rnn_model.build(a1, is_training=FLAGS.is_training, seq_len=a1_seq_len)
            answer2_rnn_logits = answer_rnn_model.build(a2, is_training=FLAGS.is_training, seq_len=a2_seq_len)
            answer3_rnn_logits = answer_rnn_model.build(a3, is_training=FLAGS.is_training, seq_len=a3_seq_len)
            answer4_rnn_logits = answer_rnn_model.build(a4, is_training=FLAGS.is_training, seq_len=a4_seq_len)
            answer0_attention = tf.layers.dense(answer0_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                    name="answer_attention_layer")
            answer1_attention = tf.layers.dense(answer1_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                    name="answer_attention_layer", reuse=True)
            answer2_attention = tf.layers.dense(answer2_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                    name="answer_attention_layer", reuse=True)
            answer3_attention = tf.layers.dense(answer3_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                    name="answer_attention_layer", reuse=True)
            answer4_attention = tf.layers.dense(answer4_rnn_logits, FLAGS.num_hidden * 2, activation=tf.nn.relu,
                                                    name="answer_attention_layer", reuse=True)
            answer0_attention = tf.multiply(answer0_attention, answer_context_vector)
            answer1_attention = tf.multiply(answer1_attention, answer_context_vector)
            answer2_attention = tf.multiply(answer2_attention, answer_context_vector)
            answer3_attention = tf.multiply(answer3_attention, answer_context_vector)
            answer4_attention = tf.multiply(answer4_attention, answer_context_vector)
            answer0_attention = tf.nn.l2_normalize(answer0_attention, axis=1)
            answer1_attention = tf.nn.l2_normalize(answer1_attention, axis=1)
            answer2_attention = tf.nn.l2_normalize(answer2_attention, axis=1)
            answer3_attention = tf.nn.l2_normalize(answer3_attention, axis=1)
            answer4_attention = tf.nn.l2_normalize(answer4_attention, axis=1)
            answer0_rnn_logits = tf.multiply(answer0_rnn_logits, answer0_attention)
            answer1_rnn_logits = tf.multiply(answer1_rnn_logits, answer1_attention)
            answer2_rnn_logits = tf.multiply(answer2_rnn_logits, answer2_attention)
            answer3_rnn_logits = tf.multiply(answer3_rnn_logits, answer3_attention)
            answer4_rnn_logits = tf.multiply(answer4_rnn_logits, answer4_attention)
            answer0_rnn_logits = tf.reduce_sum(answer0_rnn_logits, axis=1, keepdims=True)
            answer1_rnn_logits = tf.reduce_sum(answer1_rnn_logits, axis=1, keepdims=True)
            answer2_rnn_logits = tf.reduce_sum(answer2_rnn_logits, axis=1, keepdims=True)
            answer3_rnn_logits = tf.reduce_sum(answer3_rnn_logits, axis=1, keepdims=True)
            answer4_rnn_logits = tf.reduce_sum(answer4_rnn_logits, axis=1, keepdims=True)

            #subtitle_rnn_logits = tf.nn.l2_normalize(subtitle_rnn_logits, axis=2)
            #question_rnn_logits = tf.nn.l2_normalize(question_rnn_logits, axis=2)
            #answer_rnn_logits = tf.nn.l2_normalize(answer_rnn_logits, axis=2)
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
        with tf.variable_scope("subtitle_embed"):
            subtitle_rnn_logits = tf.reshape(subtitle_rnn_logits, [-1, FLAGS.num_hidden*2, 1])
            subtitle_question_embed_logits = tf.matmul(question_rnn_logits, subtitle_rnn_logits)
            subtitle_question_embed_logits = tf.concat([subtitle_question_embed_logits, subtitle_question_embed_logits,
                                                        subtitle_question_embed_logits, subtitle_question_embed_logits,
                                                        subtitle_question_embed_logits], axis=1)
            #subtitle_answer_embed_logits = tf.matmul(answer_rnn_logits, subtitle_rnn_logits)
            subtitle_answer0_embed_logits = tf.matmul(answer0_rnn_logits, subtitle_rnn_logits)
            subtitle_answer1_embed_logits = tf.matmul(answer1_rnn_logits, subtitle_rnn_logits)
            subtitle_answer2_embed_logits = tf.matmul(answer2_rnn_logits, subtitle_rnn_logits)
            subtitle_answer3_embed_logits = tf.matmul(answer3_rnn_logits, subtitle_rnn_logits)
            subtitle_answer4_embed_logits = tf.matmul(answer4_rnn_logits, subtitle_rnn_logits)
            subtitle_answer_embed_logits = tf.concat([subtitle_answer0_embed_logits, subtitle_answer1_embed_logits,
                                                      subtitle_answer2_embed_logits, subtitle_answer3_embed_logits,
                                                      subtitle_answer4_embed_logits], axis=1)
            #subtitle_text_embed = subtitle_question_embed_logits + subtitle_answer_embed_logits
            subtitle_text_embed = (subtitle_question_embed_logits * tf.cast(0.7, tf.float32)) + \
                                  (subtitle_answer_embed_logits * tf.cast(0.3, tf.float32))
            #subtitle_text_embed = tf.reshape(subtitle_text_embed, [-1, 5, 1])

        with tf.variable_scope("prediction"):

            def loss_calc(elem):
                correct = elem[0][tf.argmax(elem[1])]

                loss1 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][0])
                loss2 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][1])
                loss3 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][2])
                loss4 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][3])
                loss5 = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin - correct + elem[0][4])
                not_loss = tf.maximum(tf.cast(0.0, tf.float32), FLAGS.margin)

                return loss1 + loss2 + loss3 + loss4 + loss5 - not_loss

            total_logits = subtitle_text_embed
            total_logits = tf.squeeze(total_logits, axis=[2])

            loss = tf.map_fn(loss_calc, (total_logits, a), dtype=tf.float32)
            cost = tf.reduce_mean(loss)
            train_var_list = []
            for v in tf.trainable_variables():
                if not (v.name.startswith(u"RGB") or v.name.startswith(u"Flow")):
                    train_var_list.append(v)

            optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
            op = optimizer.minimize(cost, var_list=train_var_list)
            comparison = tf.equal(tf.argmax(total_logits, axis=1), tf.argmax(a, axis=1))
            accuracy = tf.reduce_mean(tf.cast(comparison, tf.float32), name="accuracy")
            loss_summary = tf.summary.scalar("loss", cost)
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge([loss_summary, accuracy_summary])
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
    return cost, accuracy, op, summary_op
