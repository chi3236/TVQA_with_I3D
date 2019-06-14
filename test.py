import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
import math
import re
import time
from utils import read_json_lines, load_json, save_json, load_pickle
sys.path.append("./kinetics-i3d")
import network_tower2

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
tf.flags.DEFINE_integer('batch_size', 4, "Batch size")
tf.flags.DEFINE_integer('num_hidden', 300, "Number of hidden layer")
tf.flags.DEFINE_string("gpus", "4, 6", "GPUs to use")
tf.flags.DEFINE_integer("num_epoch", 100, "number of epoch")
tf.flags.DEFINE_float('lr', 0.001, 'learning_rate')

video_file_path = "./grey.h5"
glove_path = "./glove.840B.300d.txt"

checkpoint_path = './log/checkpoint'
checkpoint_prefix = os.path.join(checkpoint_path, "model")
glove_model = {}

with open(glove_path, 'r') as glove:
    print "Loading Glove Model"
    for line in glove:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove_model[word] = embedding
print "Done.", len(glove_model), " words loaded!"



class generator:
    def __init__(self, qa_file, prob, word2idx, is_training):
        self.video_file = video_file_path
        self.qa_file = qa_file
        self.prob = prob
        self.word2idx = word2idx
        self.is_training = is_training
        self.show_list = {'Grey\'s Anatomy': 'grey_frames', 'How I Met You Mother': 'met_frames',
                          'Friends': 'friends_frames', 'The Big Bang Theory': 'bbt_frames',
                          'House M.D.': 'house_frames', 'Castle': 'castle_frames'}
    def __call__(self):
        with hf.File(self.video_file, "r") as data:
            for qa in self.qa_file:
                if qa['vid_name'].startswith('bbt'):
                    rgb_image_path = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                       qa['vid_name'].split('bbt_')[1]))
                    #rgb_image_path_s = data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                    #                                   qa['vid_name'].split('bbt_')[1])
                    #of_image_path_s = data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                    #                                   qa['vid_name'].split('bbt_')[1])
                    #of_image_path = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                    #                                  qa['vid_name'].split('bbt_')[1]))
                else:
                    rgb_image_path = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                       qa['vid_name']))
                    #rgb_image_path_s = data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                    #                                   qa['vid_name'])
                    #of_image_path_s = data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                    #                                   qa['vid_name'])
                    #of_image_path = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                    #                                  qa['vid_name']))
                rgb_images = []
                of_images = []
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

                if qa['located_frame'][1] - qa['located_frame'][0] > 69:
                    interval = float(qa['located_frame'][1] - qa['located_frame'][0]) / 69.0
                    img_list = rgb_image_path[qa['located_frame'][0]:qa['located_frame'][1] + 1].tolist()
                    i = 0.0
                    while qa['located_frame'][0] + int(math.ceil(i)) < qa['located_frame'][1] + 1:
                        try:

                            if qa['vid_name'].startswith('bbt'):
                                #rgb_images.append(np.array(rgb_image_path_s[img_list[math.ceil(i)]]))
                                #of_image = np.array(of_image_path_s[img_list[math.ceil(i)]])
                                #of_images.append(of_image[:, :, 1:])
                                rgb_images.append(np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                                    qa['vid_name'].split('bbt_')[1] + '/' + img_list[int(math.ceil(i))])))
                                of_image = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                                                             qa['vid_name'].split('bbt_')[1] + '/' + img_list[int(math.ceil(i))]))
                                of_images.append(of_image[:, :, 1:])
                            else:
                                #rgb_images.append(np.array())
                                rgb_images.append(
                                    np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                      qa['vid_name'] + '/' + img_list[int(math.ceil(i))])))
                                of_image = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                                                             qa['vid_name'] + '/' + img_list[int(math.ceil(i))]))
                                of_images.append(of_image[:, :, 1:])

                            #rgb_images.append(np.array(rgb_image_path_s[img_list[math.ceil(i)]]))
                            #of_image = np.array(of_image_path_s[img_list[math.ceil(i)]])
                            #of_images.append(of_image[:, :, 1:])
                            i += interval
                        except:
                            break

                else:
                    for image in rgb_image_path[qa['located_frame'][0]:qa['located_frame'][1] + 1]:
                        try:
                            if qa['vid_name'].startswith('bbt'):
                                rgb_images.append(np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                                    qa['vid_name'].split('bbt_')[1] + '/' + image)))
                                of_image = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                                                             qa['vid_name'].split('bbt_')[1] + '/' + image))
                                of_images.append(of_image[:, :, 1:])
                            else:
                                rgb_images.append(
                                    np.array(data.get('video/' + self.show_list[qa['show_name']] + '/rgb_' +
                                                      qa['vid_name'] + '/' + image)))
                                of_image = np.array(data.get('video/' + self.show_list[qa['show_name']] + '/of_' +
                                                             qa['vid_name'] + '/' + image))
                                of_images.append(of_image[:, :, 1:])

                            #rgb_images.append(np.array(rgb_image_path_s[image]))
                            #of_image = np.array(of_image_path_s[image])
                            #of_images.append(of_image[:, :, 1:])
                        except:
                            break
                if len(rgb_images) == 0:
                    zero_arr = np.zeros((224, 224, 3), dtype=np.float32)
                    rgb_images.append(zero_arr)
                if len(of_images) == 0:
                    zero_arr = np.full((224, 224, 2), 127.0, dtype=np.float32)
                    of_images.append(zero_arr)
                answer = [0] * 5
                answer[qa['answer_idx']] = 1

                return_value = {'rgb': rgb_images, 'of': of_images, 'sub': subtitles, 'q': question,
                                'a0': answer0, 'a1': answer1, 'a2': answer2, 'a3': answer3, 'a4': answer4,
                                'a': answer, 'qid': [qa['qid']], 'sub_seq_len': [len(subtitles)],
                                'q_seq_len': [len(question)], 'a0_seq_len': [len(answer0)],
                                'a1_seq_len': [len(answer1)], 'a2_seq_len': [len(answer2)],
                                'a3_seq_len': [len(answer3)], 'a4_seq_len': [len(answer4)], 'prob': [self.prob],
                                'text_prob': [1.0], 'is_training': [self.is_training]}
                yield return_value
                # yield rgb_images, of_images


def preprocessing():
    pass

def tower_loss(scope, vocab_embedding, next_element):
    _, accuracy, video_accuracy, qid, comparison, train_var_list = network_tower2.build_graph(FLAGS, vocab_embedding,
                                                                             next_element['rgb'], next_element['of'],
                                                                             next_element['sub'], next_element['q'],
                                                                             next_element['a0'], next_element['a1'],
                                                                             next_element['a2'], next_element['a3'],
                                                                             next_element['a4'], next_element['a'],
                                                                             next_element['qid'],
                                                                             next_element['sub_seq_len'],
                                                                             next_element['q_seq_len'],
                                                                             next_element['a0_seq_len'],
                                                                             next_element['a1_seq_len'],
                                                                             next_element['a2_seq_len'],
                                                                             next_element['a3_seq_len'],
                                                                             next_element['a4_seq_len'],
                                                                             next_element['prob'],
                                                                             next_element['text_prob'],
                                                                             next_element['is_training'])

    #losses = tf.get_collection('losses', scope)
    #total_loss = tf.add_n(losses, name='total_loss')
    #for l in losses:
    loss_name = re.sub("%s_[0-9]*/" % network_tower2.TOWER_NAME, '', _.op.name)
    tf.summary.scalar(loss_name, _)
    return _, accuracy, video_accuracy ,qid, train_var_list

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
    #early_stopping = EarlyStopping(patience=3, verbose=True)
    print("Loading json")
    for i, qa_file in sorted(enumerate(qa_srt_files), key=lambda x: x[1]):
        qa_file_list.append(load_json(qa_file))
    print("Done")
    print("Loading pickle")
    word2idx = load_pickle("word2idx.pkl")
    vocab_embedding = load_pickle("vocab_embedding_matrix.pkl")
    print("Done")
    train_generator = generator(qa_file_list[1], 0.5, word2idx, True)
    val_generator = generator(qa_file_list[2], 1.0, word2idx, False)
    #train_data = train_generator()
    #val_data = val_generator()
    #for x in train_data:
    #    pass

    #for y in val_data:
    #    pass

    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
                                                                                  'sub': tf.int32, 'q': tf.int32,
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
                                                                                  'prob': tf.float32,
                                                                                  'text_prob': tf.float32,
                                                                                  'is_training': tf.bool})

    #train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
    #                                                                              'sub': tf.float64, 'q': tf.float64,
    #                                                                              'ac': tf.float64})
    train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
                                               padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
                                                              'sub': [None], 'q': [None], 'a0': [None], 'a1': [None],
                                                              'a2': [None], 'a3': [None], 'a4': [None], 'a': [5],
                                                              'qid': [1], 'sub_seq_len': [1], 'q_seq_len': [1],
                                                              'a0_seq_len': [1], 'a1_seq_len': [1], 'a2_seq_len': [1],
                                                              'a3_seq_len': [1], 'a4_seq_len': [1], 'prob': [1],
                                                              'text_prob': [1], 'is_training': [1]})

    #train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
    #                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
    #                                                          'sub': [None, 300], 'q': [None, 300], 'ac': [None, 300]})

    val_dataset = tf.data.Dataset.from_generator(val_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
                                                                              'sub': tf.int32, 'q': tf.int32,
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
                                                                              'prob': tf.float32,
                                                                              'text_prob': tf.float32,
                                                                              'is_training': tf.bool})

    val_dataset = val_dataset.padded_batch(FLAGS.batch_size,
                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
                                                          'sub': [None], 'q': [None], 'a0': [None], 'a1': [None],
                                                          'a2': [None], 'a3': [None], 'a4': [None], 'a': [5],
                                                          'qid': [1], 'sub_seq_len': [1], 'q_seq_len': [1],
                                                          'a0_seq_len': [1], 'a1_seq_len': [1], 'a2_seq_len': [1],
                                                          'a3_seq_len': [1], 'a4_seq_len': [1], 'prob': [1],
                                                          'text_prob': [1], 'is_training': [1]})

    #iterator = train_dataset.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    #training_init_op = iterator.make_initializer(train_dataset)
    #validation_init_op = iterator.make_initializer(val_dataset)

    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        #num_batches_per_epoch = (122039 / FLAGS.batch_size / FLAGS.num_gpus)
        #num_batches_per_epoch = (12204 / FLAGS.batch_size / FLAGS.num_gpus)
        #decay_steps = int(num_batches_per_epoch * 3)
        #lr = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps, 0.1, staircase=True)
        #opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        #opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)
        #tower_grads = []
        accuracy_list = []
        video_accuracy_list = []
        tower_qid = []
        #summaries = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for j in range(FLAGS.num_gpus):
                next_element = iterator.get_next()
                with tf.device('/gpu:%d' % j):
                    with tf.name_scope('%s_%d' % (network_tower2.TOWER_NAME, j)) as scope:
                        loss, accuracy, video_accuracy,qid, train_var_list = tower_loss(scope, vocab_embedding, next_element)
                        #summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
                        #grads = opt.compute_gradients(loss, var_list=train_var_list)
                        #tower_grads.append(grads)
                        tower_qid.append(qid)
                        accuracy_list.append(accuracy)
                        video_accuracy_list.append(video_accuracy)

        #grads = average_gradients(tower_grads)
        accuracy_avg = sum(accuracy_list) / len(accuracy_list)
        video_accuracy_avg = sum(video_accuracy_list) / len(video_accuracy_list)
        #summaries.append(tf.summary.scalar("accuracy", accuracy_avg))
        #summaries.append(tf.summary.scalar("video_accuracy", video_accuracy_avg))
        #summaries.append(tf.summary.scalar('learning_rate', lr))

        '''
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        '''
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        '''
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        '''
        #update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        #variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        #variable_averages_op = variable_averages.apply(tf.trainable_variables())
        #train_op = tf.group(apply_gradient_op, variable_averages_op)
        #train_op = apply_gradient_op
        #train_op = tf.group(apply_gradient_op, update_ops)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
        '''
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and not "Momentum" in variable.name and not "rgb_fc" in variable.name:
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow' and not "Momentum" in variable.name and not "flow_fc" in variable.name:
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        text_var_list = []
        for v in tf.global_variables():
            if v.name.startswith(u"Text") or v.name.startswith(u"post_rnn") or v.name.startswith(u"beta"):
                text_var_list.append(v)
        network_saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        text_saver = tf.train.Saver(text_var_list)
        summary_op = tf.summary.merge(summaries)
        #global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
        '''
        saver = tf.train.Saver()
        _global_step = 0
        val_step = 0
    with tf.Session(config=config) as sess:
        saver.restore(sess, "./log/best_result/video1/flow_best/model-108972")
        val_init_op = iterator.make_initializer(val_dataset)
        sess.run(val_init_op)
        cummulative_acc = 0.0
        while True:
            try:
                # next_element = iterator.get_next()
                outputs = sess.run({'accuracy': accuracy})
                cummulative_acc += outputs["accuracy"]
                global_step += 1
            except tf.errors.OutOfRangeError:
                break

    loss = cummulative_acc / global_step
    print str(loss)


'''
for x in train_data:
    print x
'''

if __name__ == '__main__':
    tf.app.run(main)