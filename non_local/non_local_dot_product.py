from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from utils.layers import ConvBlock, MaxPool

class NonLocalBlockND(object):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, is_training=True):
        assert dimension in [1, 2, 3], "Unknown dimension..."

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.sub_sample = sub_sample

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = ConvBlock(self.inter_channels, dimension, 1)

        self.W = ConvBlock(self.in_channels, dimension, 1, use_bn=bn_layer, is_training=is_training)

        self.theta = ConvBlock(self.inter_channels, dimension, 1)

        self.phi = ConvBlock(self.inter_channels, dimension, 1)

        if sub_sample:
            self.g_pool = MaxPool(dimension)
            self.phi_pool = MaxPool(dimension)

    def forward(self, x):
        residual = x

        batch_size = x.get_shape().as_list()[0]

        g_x = self.g(x)
        if self.sub_sample:
            g_x = self.g_pool(g_x)
        g_x = tf.reshape(g_x, [batch_size, -1, self.inter_channels])

        theta_x = self.theta(x)
        theta_x = tf.reshape(theta_x, (batch_size, -1, self.inter_channels))

        phi_x = self.phi(x)
        if self.sub_sample:
            phi_x = self.phi_pool(phi_x)
        phi_x = tf.reshape(phi_x, (batch_size, self.inter_channels, -1))

        f = tf.matmul(theta_x, phi_x)
        N = f.get_shape().as_list()[-1]
        f_div_C = f / N

        y = tf.matmul(f_div_C, g_x)
        y = tf.reshape(y, [batch_size] + x.get_shape().as_list()[1:-1] + [self.inter_channels])
        W_y = self.W(y)
        z = W_y + residual

        return z

    def __call__(self, inputs):
        return self.forward(inputs)


class NonLocalBlock1D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True, is_training=True):
        super(NonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=dimension,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              is_training=is_training)


class NonLocalBlock2D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, is_training=True):
        super(NonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=dimension,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              is_training=is_training)


class NonLocalBlock3D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, is_training=True):
        super(NonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=dimension,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              is_training=is_training)
