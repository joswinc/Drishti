#!/usr/bin/python
import tensorflow as tf

#from config import Config
from img_cap.model import CaptionGenerator
from img_cap.dataset import prepare_train_data, prepare_eval_data, prepare_test_data

class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 2
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = './img_cap/models/'
        self.summary_dir = './img_cap/summary/'

        # about the vocabulary
        self.vocabulary_file = './img_cap/vocabulary.csv'
        self.vocabulary_size = 5000

        # about the training
        self.train_image_dir = './train/images/'
        self.train_caption_file = './train/captions_train2014.json'
        self.temp_annotation_file = './train/anns.csv'
        self.temp_data_file = './train/data.npy'

        # about the evaluation
        self.eval_image_dir = './val/images/'
        self.eval_caption_file = './val/captions_val2014.json'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False

        # about the testing
        self.test_image_dir = './img_cap/test/images/'
        self.test_result_dir = './img_cap/test/results/'
        self.test_result_file = './img_cap/test/results.csv'

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', './img_cap/models/289999.npy',
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main():
    config = Config()
    config.phase = 'test'
    config.train_cnn = False
    config.beam_size = 3

    sess = tf.Session()
    data, vocabulary = prepare_test_data(config)
    model = CaptionGenerator(config)
    model.load(sess, FLAGS.model_file)
    sess.graph.finalize()
    return model,sess,data,vocabulary       


def imgcaptest(model,sess,data,vocabulary):
    model.test(sess, data, vocabulary)