import tensorflow as tf
from Upsampling.configs import FLAGS
from datetime import datetime
import os
import logging
import pprint


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pp = pprint.PrettyPrinter()


def run():
    if FLAGS.phase=='train':
        print('train_file:', FLAGS.train_file)
        
        # Training based on previous results
        if not FLAGS.restore:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            FLAGS.log_dir = os.path.join(FLAGS.log_dir, current_time)
            try:
                os.makedirs(FLAGS.log_dir)
            except os.error:
                pass
    else:
        FLAGS.test_data = os.path.join(FLAGS.data_dir, '*.xyz')
        if not os.path.exists(FLAGS.out_folder):
            os.makedirs(FLAGS.out_folder)
        print('test_data:',FLAGS.test_data)

    print('checkpoints:',FLAGS.log_dir)
    pp.pprint(FLAGS)
    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        if FLAGS.which_model == 'unsupervised':
            from Upsampling.UPUOT import Model as M
        else:
            from Upsampling.model import Model as M
        model = M(FLAGS, sess)
        if FLAGS.phase == 'train':
            model.train()
        else:
            model.test()


def main(unused_argv):
    run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
