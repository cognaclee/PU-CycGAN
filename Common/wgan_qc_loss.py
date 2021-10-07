#-*- coding:utf-8 -*-
""" 
Implements WGAN_QC regulation and loss:
Wasserstein Distance 
Author: Zezeng Li
Date: March 2021
"""
import tensorflow as tf
import math
from Common.loss_utils import pc_distance

'''
input:
    pred_d_real: Output of discriminator at realImg
    pred_d_fake: Output of discriminator at fakeImg
    HStar_real: real image H  value
    HStar_fake: fake image H value
output:
    loss : Wasserstein Distance
'''
def wgan_QCloss(pred_d_real, pred_d_fake,HStar_real,HStar_fake):
    mean_HStar_real = tf.reduce_mean(HStar_real)
    diff = pred_d_fake- HStar_fake
    diffSqu = tf.square(diff)

    loss = 0.5*tf.square(tf.reduce_mean(pred_d_real) - mean_HStar_real) + 0.5*tf.reduce_mean(diffSqu)
    return loss

'''
input:
    fake_points: fake points or target domain points
    KCoef: Lipschitz condition of K Coefficient
output:  
    gradient_penalty: WGAN_QC regulation
'''   
def gradient_penalty_WQC(discriminator, fake_points, KCoef, pc_dist):
    gradients = tf.gradients(discriminator(fake_points), [fake_points])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - KCoef*pc_dist)**2)/math.sqrt(KCoef)
    return gradient_penalty
    
    
   
def gradient_penalty_WGP(discriminator, fake_points, real_points):
    alpha = tf.random_uniform(shape=[1], minval=0., maxval=1.)
    differences = real_points - fake_points
    interpolates = real_points+ (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)

    return gradient_penalty
