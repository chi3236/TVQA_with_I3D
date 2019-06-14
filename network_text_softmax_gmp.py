import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
sys.path.append("./kinetics-i3d")
import i3d
import video_rnn
import text_rnn


#def build_graph(FLAGS, rgb_input, flow_input, sub, q, ac, a, rgb_seq_len, flow_seq_len,sub_seq_len, q_seq_len):
def build_graph(FLAGS, sub, q, a0, a1, a2, a3, a4, a, sub_seq_len, q_seq_len, a0_seq_len, a1_seq_len, a2_seq_len, a3_seq_len, a4_seq_len):

    with tf.device("/GPU:0"):
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
        with tf.variable_scope("Text_GRU", reuse=tf.AUTO_REUSE):
            question_rnn_model = text_rnn.tRNN("GRU", FLAGS.num_hidden, 'question')
            answer_rnn_model = text_rnn.tRNN("GRU", FLAGS.num_hidden, 'answer')
            #answer_rnn_model = text_rnn.tRNN_answer("GRU", 200, 'answer')
            subtitle_rnn_model = text_rnn.tRNN("GRU", FLAGS.num_hidden, 'subtitle')
            question_rnn_logits = question_rnn_model.build(q, is_training=FLAGS.is_training, seq_len=q_seq_len)
            subtitle_rnn_logits = subtitle_rnn_model.build(sub, is_training=FLAGS.is_training, seq_len=sub_seq_len)
            question_rnn_logits = tf.layers.batch_normalization(question_rnn_logits)
            subtitle_rnn_logits = tf.layers.batch_normalization(subtitle_rnn_logits)
            question_rnn_logits = tf.nn.softmax(question_rnn_logits)
            subtitle_rnn_logits = tf.nn.softmax(subtitle_rnn_logits)
            question_rnn_logits = tf.reduce_max(question_rnn_logits, axis=1, keepdims=True)
            subtitle_rnn_logits = tf.reduce_max(subtitle_rnn_logits, axis=1, keepdims=True)
            #answer_rnn_logits = answer_rnn_model.build(ac, is_training=FLAGS.is_training)
            answer0_rnn_logits = answer_rnn_model.build(a0, is_training=FLAGS.is_training, seq_len=a0_seq_len)
            answer1_rnn_logits = answer_rnn_model.build(a1, is_training=FLAGS.is_training, seq_len=a1_seq_len)
            answer2_rnn_logits = answer_rnn_model.build(a2, is_training=FLAGS.is_training, seq_len=a2_seq_len)
            answer3_rnn_logits = answer_rnn_model.build(a3, is_training=FLAGS.is_training, seq_len=a3_seq_len)
            answer4_rnn_logits = answer_rnn_model.build(a4, is_training=FLAGS.is_training, seq_len=a4_seq_len)
            answer0_rnn_logits = tf.layers.batch_normalization(answer0_rnn_logits)
            answer1_rnn_logits = tf.layers.batch_normalization(answer1_rnn_logits)
            answer2_rnn_logits = tf.layers.batch_normalization(answer2_rnn_logits)
            answer3_rnn_logits = tf.layers.batch_normalization(answer3_rnn_logits)
            answer4_rnn_logits = tf.layers.batch_normalization(answer4_rnn_logits)
            answer0_rnn_logits = tf.nn.softmax(answer0_rnn_logits)
            answer1_rnn_logits = tf.nn.softmax(answer1_rnn_logits)
            answer2_rnn_logits = tf.nn.softmax(answer2_rnn_logits)
            answer3_rnn_logits = tf.nn.softmax(answer3_rnn_logits)
            answer4_rnn_logits = tf.nn.softmax(answer4_rnn_logits)
            answer0_rnn_logits = tf.reduce_max(answer0_rnn_logits, axis=1, keepdims=True)
            answer1_rnn_logits = tf.reduce_max(answer1_rnn_logits, axis=1, keepdims=True)
            answer2_rnn_logits = tf.reduce_max(answer2_rnn_logits, axis=1, keepdims=True)
            answer3_rnn_logits = tf.reduce_max(answer3_rnn_logits, axis=1, keepdims=True)
            answer4_rnn_logits = tf.reduce_max(answer4_rnn_logits, axis=1, keepdims=True)

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
            subtitle_text_embed = (subtitle_question_embed_logits * tf.cast(0.7, tf.float64)) + \
                                  (subtitle_answer_embed_logits * tf.cast(0.3, tf.float64))
            #subtitle_text_embed = tf.reshape(subtitle_text_embed, [-1, 5, 1])

        with tf.variable_scope("prediction"):

            def loss_calc(elem):
                correct = elem[0][tf.argmax(elem[1])]

                loss1 = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin - correct + elem[0][0])
                loss2 = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin - correct + elem[0][1])
                loss3 = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin - correct + elem[0][2])
                loss4 = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin - correct + elem[0][3])
                loss5 = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin - correct + elem[0][4])
                not_loss = tf.maximum(tf.cast(0.0, tf.float64), FLAGS.margin)

                return loss1 + loss2 + loss3 + loss4 + loss5 - not_loss

            total_logits = subtitle_text_embed
            total_logits = tf.squeeze(total_logits, axis=[2])

            loss = tf.map_fn(loss_calc, (total_logits, a), dtype=tf.float64)
            cost = tf.reduce_mean(loss)
            train_var_list = []
            for v in tf.trainable_variables():
                if not (v.name.startswith(u"RGB") or v.name.startswith(u"Flow")):
                    train_var_list.append(v)

            optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
            op = optimizer.minimize(cost, var_list=train_var_list)
            comparison = tf.equal(tf.argmax(total_logits, axis=1), tf.argmax(a, axis=1))
            accuracy = tf.reduce_mean(tf.cast(comparison, tf.float64), name="accuracy")
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
