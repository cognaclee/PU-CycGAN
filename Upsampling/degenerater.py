# -*- coding: utf-8 -*-
# @Time        : 8/3/2021 21:22 PM
# @Description :
# @Author      : li ze zeng
import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
class Degenerater(object):
    def __init__(self, opts,is_training, name="Degenerater"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.down_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point*self.down_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction(inputs, scope='down_feature_extraction', is_training=self.is_training, bn_decay=None,up_feature=False)
            #features = ops.feature_extraction(inputs, scope='down_feature_extraction', is_training=self.is_training, bn_decay=None)

            D = ops.down_projection_unit(features, self.down_ratio, scope="down_projection_unit", is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(D, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None)

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])
            
            #sess = tf.Session()
            #print('#### outputs.shape=',sess.run(tf.shape(outputs)))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs