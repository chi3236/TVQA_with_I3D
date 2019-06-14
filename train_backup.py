import glob
import sys
import os
import tensorflow as tf
import h5py as hf
import numpy as np
import math
from utils import read_json_lines, load_json, save_json, load_pickle
sys.path.append("./kinetics-i3d")
import i3d
import video_rnn
import text_rnn
import network

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
tf.flags.DEFINE_integer('batch_size', 16, "Batch size")
tf.flags.DEFINE_integer('num_hidden', 150, "Number of hidden layer")
tf.flags.DEFINE_string("gpus", "2, 3, 4", "GPUs to use")
tf.flags.DEFINE_integer("num_epoch", 150, "number of epoch")
tf.flags.DEFINE_float('lr', 0.001, 'learning_rate')
tf.flags.DEFINE_float('margin', 0.2, 'margin for ranking loss')
tf.flags.DEFINE_bool("is_training", True, 'Is training')

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
    def __init__(self, qa_file):
        self.video_file = video_file_path
        self.qa_file = qa_file
        self.model = glove_model
    def __call__(self):
        with hf.File(self.video_file, "r") as data:
                def glove_embedding(word):
                    try:
                        word_embedding = self.model[word]
                        word_embedding = np.float32(word_embedding)
                    except:
                        word_embedding = np.random.randn(300) * 0.4
                        word_embedding = np.float32(word_embedding)
                    return word_embedding
                for qa in self.qa_file:
                    #rgb_image_path = np.array(data.get('video/rgb_' + qa['vid_name']))
                    #of_image_path = np.array(data.get('video/of_' + qa['vid_name']))
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
                        if word != '<eos>':
                            subtitles.append(glove_embedding(word))

                    for word in qa['q'].split():
                        question.append(glove_embedding(word))

                    for word in qa['a0'].split():
                        answer0.append(glove_embedding(word))

                    for word in qa['a1'].split():
                        answer1.append(glove_embedding(word))

                    for word in qa['a2'].split():
                        answer2.append(glove_embedding(word))

                    for word in qa['a3'].split():
                        answer3.append(glove_embedding(word))

                    for word in qa['a4'].split():
                        answer4.append(glove_embedding(word))
                    '''
                    if qa['located_frame'][1] - qa['located_frame'][0] > 75:
                        interval = float(qa['located_frame'][1] - qa['located_frame'][0]) / 75.0
                        img_list = rgb_image_path[qa['located_frame'][0]:qa['located_frame'][1] + 1].tolist()
                        i = 0.0
                        while i < qa['located_frame'][1]:
                            try:
                                rgb_images.append(np.array(data.get('video/rgb_' + qa['vid_name'] + '/' + img_list[int(math.ceil(i))])))
                                of_image = np.array(
                                    data.get('video/of_' + qa['vid_name'] + '/' + img_list[int(math.ceil(i))]))
                                of_images.append(of_image[:, :, 1:])
                                i += interval
                            except IndexError:
                                break

                    else:
                        for image in rgb_image_path[qa['located_frame'][0]:qa['located_frame'][1] + 1]:
                            try:
                                rgb_images.append(np.array(data.get('video/rgb_' + qa['vid_name'] + '/' + image)))
                                of_image = np.array(data.get('video/of_' + qa['vid_name'] + '/' + image))
                                of_images.append(of_image[:, :, 1:])
                            except IndexError:
                                break
                    '''
                    answer = [0] * 5
                    answer[qa['answer_idx']] = 1
                    '''
                    return_value = {'rgb': rgb_images, 'of': of_images, 'sub': subtitles, 'q': question,
                                    'a0': answer0, 'a1': answer1, 'a2': answer2, 'a3': answer3, 'a4': answer4, 'a': answer, 'rgb_seq_len': [len(rgb_images)],
                                    'flow_seq_len': [len(of_images)], 'sub_seq_len': [len(subtitles)],
                                    'q_seq_len': [len(question)], 'a0_seq_len': [len(answer0)], 'a1_seq_len': [len(answer1)],
                                    'a2_seq_len': [len(answer2)], 'a3_seq_len': [len(answer3)],
                                    'a4_seq_len': [len(answer4)]}
                    '''
                    return_value = {'sub': subtitles, 'q': question,
                                    'a0': answer0, 'a1': answer1, 'a2': answer2, 'a3': answer3, 'a4': answer4,
                                    'a': answer,
                                    'sub_seq_len': [len(subtitles)], 'q_seq_len': [len(question)],
                                    'a0_seq_len': [len(answer0)], 'a1_seq_len': [len(answer1)],
                                    'a2_seq_len': [len(answer2)], 'a3_seq_len': [len(answer3)],
                                    'a4_seq_len': [len(answer4)]}

                    #return_value = {'rgb': rgb_images, 'of': of_images, 'sub': subtitles, 'q': question,  'ac': answer_candidates}

                    yield return_value
                    #yield rgb_images, of_images

def preprocessing():
    pass

def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    qa_srt_files = glob.glob(os.path.join(qa_srt_file_path, "*json"))
    qa_file_list = []
    for i, qa_file in enumerate(qa_srt_files):
        qa_file_list.append(load_json(qa_file))

    train_generator = generator(qa_file_list[1])
    val_generator = generator(qa_file_list[0])
    #train_data = train_generator()

    #for x in train_data:
    #    pass
    '''
    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
                                                                                  'sub': tf.float64, 'q': tf.float64,
                                                                                  'a0': tf.float64, 'a1': tf.float64,
                                                                                  'a2': tf.float64, 'a3': tf.float64,
                                                                                  'a4': tf.float64, 'a': tf.int32,
                                                                                  'rgb_seq_len': tf.int32,
                                                                                  'flow_seq_len': tf.int32,
                                                                                  'sub_seq_len': tf.int32,
                                                                                  'q_seq_len': tf.int32, 'a0_seq_len': tf.int32,
                                                                                  'a1_seq_len': tf.int32,
                                                                                  'a2_seq_len': tf.int32,
                                                                                  'a3_seq_len': tf.int32,
                                                                                  'a4_seq_len': tf.int32})

    #train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
    #                                                                              'sub': tf.float64, 'q': tf.float64,
    #                                                                              'ac': tf.float64})
    train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
                                               padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
                                                              'sub': [None, 300], 'q': [None, 300], 'a0': [None, 300], 'a1': [None, 300], 'a2': [None, 300],
                                                              'a3': [None, 300], 'a4': [None, 300],
                                                              'rgb_seq_len': [1], 'flow_seq_len': [1],
                                                              'a': [5], 'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                              'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                              'a4_seq_len': [1]})
    '''

    #train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
    #                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
    #                                                          'sub': [None, 300], 'q': [None, 300], 'ac': [None, 300]})
    '''
    val_dataset = tf.data.Dataset.from_generator(val_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
                                                                                  'sub': tf.float64, 'q': tf.float64,
                                                                                  'a0': tf.float64, 'a1': tf.float64,
                                                                                  'a2': tf.float64, 'a3': tf.float64,
                                                                                  'a4': tf.float64, 'a': tf.int32,
                                                                                  'rgb_seq_len': tf.int32,
                                                                                  'flow_seq_len': tf.int32,
                                                                                  'sub_seq_len': tf.int32,
                                                                                  'q_seq_len': tf.int32, 'a0_seq_len': tf.int32,
                                                                                  'a1_seq_len': tf.int32,
                                                                                  'a2_seq_len': tf.int32,
                                                                                  'a3_seq_len': tf.int32,
                                                                                  'a4_seq_len': tf.int32})

    val_dataset = val_dataset.padded_batch(FLAGS.batch_size,
                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
                                                              'sub': [None, 300], 'q': [None, 300], 'a0': [None, 300], 'a1': [None, 300], 'a2': [None, 300],
                                                              'a3': [None, 300], 'a4': [None, 300],
                                                              'rgb_seq_len': [1], 'flow_seq_len': [1],
                                                              'a': [5], 'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                              'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                              'a4_seq_len': [1]})
    '''
    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'sub': tf.float64,
                                                                                  'q': tf.float64,
                                                                                  'a0': tf.float64, 'a1': tf.float64,
                                                                                  'a2': tf.float64, 'a3': tf.float64,
                                                                                  'a4': tf.float64, 'a': tf.int32,
                                                                                  'sub_seq_len': tf.int32,
                                                                                  'q_seq_len': tf.int32,
                                                                                  'a0_seq_len': tf.int32,
                                                                                  'a1_seq_len': tf.int32,
                                                                                  'a2_seq_len': tf.int32,
                                                                                  'a3_seq_len': tf.int32,
                                                                                  'a4_seq_len': tf.int32})

    # train_dataset = tf.data.Dataset.from_generator(train_generator, output_types={'rgb': tf.float32, 'of': tf.float32,
    #                                                                              'sub': tf.float64, 'q': tf.float64,
    #                                                                              'ac': tf.float64})
    train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
                                               padded_shapes={'sub': [None, 300], 'q': [None, 300],
                                                              'a0': [None, 300], 'a1': [None, 300], 'a2': [None, 300],
                                                              'a3': [None, 300], 'a4': [None, 300], 'a': [5],
                                                              'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                              'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                              'a4_seq_len': [1]})

    # train_dataset = train_dataset.padded_batch(FLAGS.batch_size,
    #                                           padded_shapes={'rgb': [None, 224, 224, 3], 'of': [None, 224, 224, 2],
    #                                                          'sub': [None, 300], 'q': [None, 300], 'ac': [None, 300]})

    val_dataset = tf.data.Dataset.from_generator(val_generator, output_types={'sub': tf.float64,
                                                                              'q': tf.float64,
                                                                              'a0': tf.float64, 'a1': tf.float64,
                                                                              'a2': tf.float64, 'a3': tf.float64,
                                                                              'a4': tf.float64, 'a': tf.int32,
                                                                              'sub_seq_len': tf.int32,
                                                                              'q_seq_len': tf.int32,
                                                                              'a0_seq_len': tf.int32,
                                                                              'a1_seq_len': tf.int32,
                                                                              'a2_seq_len': tf.int32,
                                                                              'a3_seq_len': tf.int32,
                                                                              'a4_seq_len': tf.int32})

    val_dataset = val_dataset.padded_batch(FLAGS.batch_size,
                                           padded_shapes={'sub': [None, 300], 'q': [None, 300],
                                                          'a0': [None, 300], 'a1': [None, 300], 'a2': [None, 300],
                                                          'a3': [None, 300], 'a4': [None, 300], 'a': [5],
                                                          'sub_seq_len': [1], 'q_seq_len': [1], 'a0_seq_len': [1],
                                                          'a1_seq_len': [1], 'a2_seq_len': [1], 'a3_seq_len': [1],
                                                          'a4_seq_len': [1]})
    #iterator = train_dataset.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    #training_init_op = iterator.make_initializer(train_dataset)
    #validation_init_op = iterator.make_initializer(val_dataset)
    '''
    cost, accuracy, op, summary_op = network.build_graph(FLAGS, next_element['rgb'],
                                                         next_element['of'], next_element['sub'],
                                                         next_element['q'], next_element['a0'], next_element['a1'],
                                                         next_element['a2'], next_element['a3'], next_element['a4'],
                                                         next_element['a'],
                                                         next_element['rgb_seq_len'],
                                                         next_element['flow_seq_len'],
                                                         next_element['sub_seq_len'],
                                                         next_element['q_seq_len'], next_element['a0_seq_len'],
                                                         next_element['a1_seq_len'], next_element['a2_seq_len'],
                                                         next_element['a3_seq_len'], next_element['a4_seq_len'])
    '''
    cost, accuracy, op, summary_op = network.build_graph(FLAGS, next_element['sub'],
                                                                           next_element['q'], next_element['a0'],
                                                                           next_element['a1'],
                                                                           next_element['a2'], next_element['a3'],
                                                                           next_element['a4'],
                                                                           next_element['a'],
                                                                           next_element['sub_seq_len'],
                                                                           next_element['q_seq_len'],
                                                                           next_element['a0_seq_len'],
                                                                           next_element['a1_seq_len'],
                                                                           next_element['a2_seq_len'],
                                                                           next_element['a3_seq_len'],
                                                                           next_element['a4_seq_len'])
    #cost, accuracy, op = network.build_graph(FLAGS, next_element['rgb'], next_element['of'], next_element['sub'],
    #                                         next_element['q'], next_element['ac'])


    '''
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    flow_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
            flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    '''
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)

    network_saver = tf.train.Saver(tf.global_variables())
    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    global_step = 0
    val_step = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        #flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
        train_summary_writer = tf.summary.FileWriter('./log/train', sess.graph)
        val_summary_writer = tf.summary.FileWriter('./log/val', sess.graph)
        for i in range(FLAGS.num_epoch):
            training_init_op = iterator.make_initializer(train_dataset)
            sess.run(training_init_op)
            while True:
                try:
                    #next_element = iterator.get_next()
                    outputs = sess.run({'cost': cost, 'accuracy': accuracy, 'op': op, 'summary': summary_op})
                    train_summary_writer.add_summary(outputs['summary'], global_step=global_step)
                    global_step += 1
                except tf.errors.OutOfRangeError:
                    break

            if i+1 % 10 == 0:
                validation_init_op = iterator.make_initializer(val_dataset)
                sess.run(validation_init_op)
                while True:
                    try:
                        #next_element = iterator.get_next()
                        outputs = sess.run({'cost': cost, 'accuracy': accuracy, 'op': op, 'summary': summary_op})
                        val_summary_writer.add_summary(outputs['summary'], global_step=val_step)
                        val_step += 1
                    except tf.errors.OutOfRangeError:
                        break
            path = network_saver.save(sess, checkpoint_prefix, global_step=global_step)

'''
for x in train_data:
    print x
'''


if __name__ == '__main__':
    tf.app.run(main)