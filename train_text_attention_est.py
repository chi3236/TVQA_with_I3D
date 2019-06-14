import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import re
import numpy as np
from collections import defaultdict
import math
from utils import read_json_lines, load_json, save_json, load_pickle
#sys.path.append("./kinetics-i3d")
#import i3d
import video_rnn
import text_rnn
import network_text_attention_bidaf_tower2

qa_srt_file_path = "."
_CHECKPOINT_PATHS = {
    'rgb': './kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': './kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': './kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_classes', 400, 'Number of action classes')
tf.flags.DEFINE_integer('batch_size', 8, "Batch size")
tf.flags.DEFINE_integer('num_hidden', 150, "Number of hidden layer")
tf.flags.DEFINE_string("gpus", "0, 1, 2, 3", "GPUs to use")
tf.flags.DEFINE_integer("num_gpus", 2, "number of gpus")
tf.flags.DEFINE_integer("num_epoch", 100, "number of epoch")
#tf.flags.DEFINE_float('lr', 3e-4, 'learning_rate')
tf.flags.DEFINE_float('lr', 3e-4, 'learning_rate')
tf.flags.DEFINE_float("wd", 1e-5, 'weight decay')
#tf.flags.DEFINE_float("wd", 0.0, 'weight decay')
tf.flags.DEFINE_float('margin', 1.0, 'margin for ranking loss')
#tf.flags.DEFINE_bool("is_training", True, 'Is training')


#video_file_path = "./grey.h5"
glove_path = "./glove.840B.300d.txt"

checkpoint_path = './log/checkpoint_text_lstm_bidaf'
checkpoint_prefix = os.path.join(checkpoint_path, "model")
#glove_model = {}
'''
with open(glove_path, 'r') as glove:
    print "Loading Glove Model"
    for line in glove:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove_model[word] = embedding
print "Done.", len(glove_model), " words loaded!"
'''

'''
print "Loading Glove Model"
with open(glove_path, 'r') as glove_file:
    for (i, line) in enumerate(glove_file):
        split = line.split(' ')
        word = split[0]
        representation = split[1:]
        representation = np.array([float(val) for val in representation])
        word_to_index_dict[word] = i
        index_to_embedding.append(representation)

_WORD_NOT_FOUND = np.random.randn(300) * 0.4
_LAST_INDEX = len(index_to_embedding)
word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
index_to_embedding = np.array(index_to_embedding + [_WORD_NOT_FOUND])
print "Done.", len(word_to_index_dict), " words loaded!"
'''
class EarlyStopping():
    def __init__(self, patience=0, verbose=False):
        self._step = 0
        self._acc = 0.0
        self.patience = patience
        self.verbose = verbose

    def validate(self, acc):
        if self._acc > acc:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._acc = acc

        return False

class generator:
    def __init__(self, qa_file, prob, word2idx):
        #self.video_file = video_file_path
        self.qa_file = qa_file
        self.prob = prob
        self.word2idx = word2idx
    def __call__(self):
        #with hf.File(self.video_file, "r") as data:
        '''
        def glove_embedding(word):
            try:
                word_embedding = self.model[word]
                #word_embedding = np.float32(word_embedding)
            except:
                word_embedding = np.random.randn(300) * 0.4
                #word_embedding = np.float32(word_embedding)
            return word_embedding
        '''
        for qa in self.qa_file:
            subtitles = []
            question = []
            answer0 = []
            answer1 = []
            answer2 = []
            answer3 = []
            answer4 = []

            for word in qa['located_sub_text'].split():
                try:
                    subtitles.append(self.word2idx[word.lower()])
                except KeyError:
                    subtitles.append(self.word2idx["<unk>"])

            for word in qa['q'].split():
                try:
                    question.append(self.word2idx[word.lower()])
                except KeyError:
                    question.append(self.word2idx["<unk>"])

            for word in qa['a0'].split():
                try:
                    answer0.append(self.word2idx[word.lower()])
                except KeyError:
                    answer0.append(self.word2idx["<unk>"])

            for word in qa['a1'].split():
                try:
                    answer1.append(self.word2idx[word.lower()])
                except KeyError:
                    answer1.append(self.word2idx["<unk>"])

            for word in qa['a2'].split():
                try:
                    answer2.append(self.word2idx[word.lower()])
                except KeyError:
                    answer2.append(self.word2idx["<unk>"])

            for word in qa['a3'].split():
                try:
                    answer3.append(self.word2idx[word.lower()])
                except KeyError:
                    answer3.append(self.word2idx["<unk>"])

            for word in qa['a4'].split():
                try:
                    answer4.append(self.word2idx[word.lower()])
                except KeyError:
                    answer4.append(self.word2idx["<unk>"])

            answer = [0] * 5
            answer[qa['answer_idx']] = 1
            return_value = {'sub': subtitles, 'q': question,
                            'a0': answer0, 'a1': answer1, 'a2': answer2, 'a3': answer3, 'a4': answer4, 'a': answer,
                            'qid': [qa['qid']], 'sub_seq_len': [len(subtitles)], 'q_seq_len': [len(question)],
                            'a0_seq_len': [len(answer0)], 'a1_seq_len': [len(answer1)],
                            'a2_seq_len': [len(answer2)], 'a3_seq_len': [len(answer3)],
                            'a4_seq_len': [len(answer4)], 'prob': [self.prob]}

            #return_value = {'rgb': rgb_images, 'of': of_images, 'sub': subtitles, 'q': question,  'ac': answer_candidates}

            yield return_value
            #yield rgb_images, of_images

def preprocessing():
    pass

def tower_loss(scope, vocab_embedding, next_element):
    _, accuracy, qid, comparison = network_text_attention_bidaf_tower2.build_graph(FLAGS, vocab_embedding,
                                                                                              next_element['sub'],
                                                                                              next_element['q'],
                                                                                              next_element['a0'],
                                                                                              next_element['a1'],
                                                                                              next_element['a2'],
                                                                                              next_element['a3'],
                                                                                              next_element['a4'],
                                                                                              next_element['a'],
                                                                                              next_element['qid'],
                                                                                              next_element[
                                                                                                  'sub_seq_len'],
                                                                                              next_element['q_seq_len'],
                                                                                              next_element[
                                                                                                  'a0_seq_len'],
                                                                                              next_element[
                                                                                                  'a1_seq_len'],
                                                                                              next_element[
                                                                                                  'a2_seq_len'],
                                                                                              next_element[
                                                                                                  'a3_seq_len'],
                                                                                              next_element[
                                                                                                  'a4_seq_len'],
                                                                                              next_element[
                                                                                                  'prob'])

    #losses = tf.get_collection('losses', scope)
    #total_loss = tf.add_n(losses, name='total_loss')
    #for l in losses:
    loss_name = re.sub("%s_[0-9]*/" % network_text_attention_bidaf_tower2.TOWER_NAME, '', _.op.name)
    tf.summary.scalar(loss_name, _)
    return _, accuracy, qid

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(_):
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    qa_srt_files = glob.glob(os.path.join(qa_srt_file_path, "*json"))
    qa_file_list = []
    early_stopping = EarlyStopping(patience=3, verbose=True)
    print("Loading json")
    for i, qa_file in sorted(enumerate(qa_srt_files), key=lambda x: x[1]):
        qa_file_list.append(load_json(qa_file))
    print("Done")
    print("Loading pickle")
    word2idx = load_pickle("word2idx.pkl")
    vocab_embedding = load_pickle("vocab_embedding_matrix.pkl")
    print("Done")
    train_generator = generator(qa_file_list[1], 0.5, word2idx)
    val_generator = generator(qa_file_list[2], 1.0, word2idx)
    #train_data = train_generator()
    #val_data = val_generator()
    #for x in train_data:
    #    pass

    #for y in val_data:
    #    pass

    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'sub': tf.int32,
                                                                                  'q': tf.int32,
                                                                                  'a0': tf.int32, 'a1': tf.int32,
                                                                                  'a2': tf.int32, 'a3': tf.int32,
                                                                                  'a4': tf.int32, 'a': tf.int32,
                                                                                  'qid': tf.int32,
                                                                                  'sub_seq_len': tf.int32,
                                                                                  'q_seq_len': tf.int32,
                                                                                  'a0_seq_len': tf.int32,
                                                                                  'a1_seq_len': tf.int32,
                                                                                  'a2_seq_len': tf.int32,
                                                                                  'a3_seq_len': tf.int32,
                                                                                  'a4_seq_len': tf.int32,
                                                                                  'prob': tf.float32})

    #train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
    #                                                                              'sub': tf.float64, 'q': tf.float64,
    #                                                                              'ac': tf.float64})
    train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
                                               padded_shapes={'sub': [None], 'q': [None],
                                                              'a0': [None], 'a1': [None], 'a2': [None],
                                                              'a3': [None], 'a4': [None], 'a': [5], 'qid': [1],
                                                              'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                              'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                              'a4_seq_len': [1], 'prob': [1]})

    #train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
    #                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
    #                                                          'sub': [None, 300], 'q': [None, 300], 'ac': [None, 300]})

    val_dataset = tf.data.Dataset.from_generator(val_generator, output_types={'sub': tf.int32,
                                                                              'q': tf.int32,
                                                                              'a0': tf.int32, 'a1': tf.int32,
                                                                              'a2': tf.int32, 'a3': tf.int32,
                                                                              'a4': tf.int32, 'a': tf.int32,
                                                                              'qid': tf.int32,
                                                                              'sub_seq_len': tf.int32,
                                                                              'q_seq_len': tf.int32,
                                                                              'a0_seq_len': tf.int32,
                                                                              'a1_seq_len': tf.int32,
                                                                              'a2_seq_len': tf.int32,
                                                                              'a3_seq_len': tf.int32,
                                                                              'a4_seq_len': tf.int32,
                                                                              'prob': tf.float32})

    val_dataset = val_dataset.padded_batch(FLAGS.batch_size,
                                           padded_shapes={'sub': [None], 'q': [None],
                                                          'a0': [None], 'a1': [None], 'a2': [None],
                                                          'a3': [None], 'a4': [None], 'a': [5], 'qid': [1],
                                                          'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                          'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                          'a4_seq_len': [1], 'prob': [1]})

    #iterator = train_dataset.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    #training_init_op = iterator.make_initializer(train_dataset)
    #validation_init_op = iterator.make_initializer(val_dataset)

    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = (122039 / FLAGS.batch_size / FLAGS.num_gpus)
        decay_steps = int(num_batches_per_epoch * 5)
        lr = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps, 0.9, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        tower_grads = []
        accuracy_list = []
        tower_qid = []
        summaries = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for j in range(FLAGS.num_gpus):
                next_element = iterator.get_next()
                with tf.device('/gpu:%d' % j):
                    with tf.name_scope('%s_%d' % (network_text_attention_bidaf_tower2.TOWER_NAME, j)) as scope:
                        loss, accuracy, qid = tower_loss(scope, vocab_embedding, next_element)
                        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        tower_qid.append(qid)
                        accuracy_list.append(accuracy)

        grads = average_gradients(tower_grads)
        accuracy_avg = sum(accuracy_list) / len(accuracy_list)
        summaries.append(tf.summary.scalar("accuracy", accuracy_avg))
        summaries.append(tf.summary.scalar('learning_rate', lr))

        '''
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        '''
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        '''
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        '''
        #variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        #variable_averages_op = variable_averages.apply(tf.trainable_variables())

        #train_op = tf.group(apply_gradient_op, variable_averages_op)
        train_op = apply_gradient_op

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)

        network_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        summary_op = tf.summary.merge(summaries)
        #global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
        _global_step = 0
        val_step = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #network_saver.restore(sess, "./log/checkpoint_text_gru_bidaf/model-22872")
            train_summary_writer = tf.summary.FileWriter('./log/train_text_bidaf_lstm', sess.graph)
            val_summary_writer = tf.summary.FileWriter('./log/val_text_bidaf_lstm', sess.graph)

            for i in range(FLAGS.num_epoch):
                training_init_op = iterator.make_initializer(train_dataset)
                sess.run(training_init_op)

                while True:
                    try:
                        #next_element = iterator.get_next()
                        output = sess.run({'_': train_op, 'summary_op': summary_op})
                        train_summary_writer.add_summary(output['summary_op'], global_step=_global_step)
                        #print output['qid']
                        _global_step += 1
                    except tf.errors.OutOfRangeError:
                        break

                with open("val_log.txt", 'a') as f:
                    #f.write("epoch: " + str(i+1) + "\n")
                    validation_init_op = iterator.make_initializer(val_dataset)
                    sess.run(validation_init_op)
                    cum_acc = 0.0
                    cum_loss = 0.0
                    k = 0
                    while True:
                        try:
                            #next_element = iterator.get_next()
                            output = sess.run({'accuracy': accuracy_avg, 'summary_op': summary_op})
                            #f.write(str(outputs['qid']) + ': ' + str(outputs[comparison]) + '\n')
                            val_summary_writer.add_summary(output['summary_op'], global_step=val_step)
                            cum_acc += output['accuracy']
                            val_step += 1
                            k += 1
                        except tf.errors.OutOfRangeError:
                            break
                    print "epoch: " + str(i+1) + " val_accuracy: " + str(cum_acc / k)
                    #if early_stopping.validate(cum_acc / k):
                    #    break
                path = network_saver.save(sess, checkpoint_prefix, global_step=global_step)

'''
for x in train_data:
    print x
'''


if __name__ == '__main__':
    tf.app.run(main)