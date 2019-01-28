from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


def batch_normalization(x, is_training):
    return tf.layers.batch_normalization(x, training=is_training)


class ConvBlock(object):
    '''
    Conv -> BN -> ReLU
    '''

    def __init__(self, out_channels, dimention=2, kernel_size=3, stride=1, dilation=1, padding='same', use_bias=True,
                 use_bn=False, use_act=False, is_training=True):
        self.out_channels = out_channels
        self.dimention = dimention
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.use_act = use_act
        self.is_training = is_training

    def forward(self, inputs):
        if self.dimention == 1:
            out = tf.layers.conv1d(inputs, self.out_channels, self.kernel_size, self.stride, self.padding,
                                   dilation_rate=self.dilation, use_bias=self.use_bias,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        elif self.dimention == 2:
            out = tf.layers.conv2d(inputs, self.out_channels, self.kernel_size, self.stride, self.padding,
                                   dilation_rate=self.dilation, use_bias=self.use_bias,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        else:
            out = tf.layers.conv3d(inputs, self.out_channels, self.kernel_size, self.stride, self.padding,
                                   dilation_rate=self.dilation, use_bias=self.use_bias,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if self.use_bn:
            out = batch_normalization(out, self.is_training)
        if self.use_act:
            out = tf.nn.relu(out)

        return out

    def __call__(self, inputs):
        return self.forward(inputs)


class MaxPool(object):
    def __init__(self, dimention=2, pool_size=2, stride=2):
        self.dimention = dimention
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        if self.dimention == 1:
            out = tf.layers.max_pooling1d(inputs, self.pool_size, self.stride)
        elif self.dimention == 2:
            out = tf.layers.max_pooling2d(inputs, self.pool_size, self.stride)
        else:
            out = tf.layers.max_pooling3d(inputs, self.pool_size, self.stride)

        return out

    def __call__(self, inputs):
        return self.forward(inputs)


class DeconvBlock(object):
    def __init__(self, output_channels, kernel_size=2, stride=2, padding='same', use_bias=True, use_bn=True,
                 use_act=True, is_training=True):
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.use_act = use_act
        self.is_training = is_training

    def forward(self, inputs):
        out = tf.layers.conv2d_transpose(inputs, self.out_channels, self.kernel_size, self.stride, padding=self.padding,
                                         use_bias=self.use_bias,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if self.use_bn:
            out = batch_normalization(out, self.is_training)
        if self.use_act:
            out = tf.nn.relu(out)

        return out


class DecoderMudule(object):
    '''
    Reference decoder method if deeplab v3+
    '''

    def __init__(self, channels=48, num_classes=5, use_bilinear=True, is_training=True):
        self.use_bilinear = use_bilinear

        self.conv1 = ConvBlock(256, kernel_size=1, use_bn=True, use_act=True, is_training=is_training)
        self.conv1x1 = ConvBlock(channels, kernel_size=1, use_bn=True, use_act=True, is_training=is_training)
        self.conv3x3 = ConvBlock(256, kernel_size=3, use_bn=True, use_act=True, is_training=is_training)
        self.deconv = DeconvBlock(channels, 2, 2, is_training=is_training)

        self.conv_out = ConvBlock(num_classes, kernel_size=1, is_training=is_training)

    def forward(self, encoder_output, encoder_features):
        out = self.conv1.forward(encoder_output)

        for i in xrange(len(encoder_features) - 1, -1, -1):
            encoder_feature = encoder_features[i]
            encoder_feature_shape = encoder_feature.get_shape().as_list()
            encoder_height, encoder_width = encoder_feature_shape[1], encoder_feature_shape[2]

            # high-level features
            if self.use_bilinear:
                out = tf.image.resize_bilinear(out, (encoder_height, encoder_width))
            else:
                out = self.deconv.forward(out)

            # low-level features
            low_level_features = self.conv1x1.forward(encoder_feature)

            # concat
            out = tf.concat([out, low_level_features], axis=3)
            out = self.conv3x3.forward(out)
            out = self.conv3x3.forward(out)

        out = self.conv_out.forward(out)

        return out
