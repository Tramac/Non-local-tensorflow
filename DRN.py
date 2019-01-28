from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from utils.layers import ConvBlock, DecoderMudule
from non_local.non_local_embedded_gaussian import NonLocalBlock2D


class BasicBlock(object):
    expansion = 1

    def __init__(self, out_channels, stride=1, dilation=(1, 1), residual=True, is_training=True):
        self.out_channels = out_channels
        self.stride = stride
        self.residual = residual
        self.is_training = is_training

        self.conv_block1 = ConvBlock(out_channels, stride=stride, dilation=dilation[0], use_bn=True, use_act=True,
                                     is_training=is_training)
        self.conv_block2 = ConvBlock(out_channels, dilation=dilation[1], use_bn=True, is_training=is_training)

    def forward(self, inputs):
        in_channels = tf.shape(inputs)[-1]
        res = inputs

        out = self.conv_block1.forward(inputs)
        out = self.conv_block2.forward(out)

        if self.stride != 1 or in_channels != self.out_channels:
            downsample = ConvBlock(self.out_channels, kernel_size=1, stride=self.stride, is_training=self.is_training)
            res = downsample.forward(inputs)

        if self.residual:
            out += res
        out = tf.nn.relu(out)

        return out


class MakeLayer(object):
    def __init__(self, block, out_channels, num_blocks, stride=1, dilation=1, new_level=True, residual=True,
                 is_training=True):
        assert dilation == 1 or dilation % 2 == 0
        self.num_blocks = num_blocks

        self.block0 = block(out_channels, stride,
                            dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation),
                            residual=residual, is_training=is_training)

        for i in range(1, num_blocks):
            block_i = block(out_channels, dilation=(dilation, dilation), residual=residual, is_training=is_training)
            setattr(self, 'block{}'.format(i), block_i)

    def forward(self, inputs):
        out = inputs

        for i in xrange(self.num_blocks):
            block_i = getattr(self, 'block{}'.format(i))
            out = block_i.forward(out)

        return out


class DRN(object):
    def __init__(self, block, layers, num_classes=5, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False,
                 out_middle=False, pool_size=32, is_training=True):
        self.out_map = out_map
        self.pool_size = pool_size
        self.out_middle = out_middle

        self.conv1 = ConvBlock(channels[0], kernel_size=7, use_bias=False, use_bn=True, use_act=True, is_training=is_training)
        self.layer1 = MakeLayer(BasicBlock, channels[0], layers[0], stride=1, is_training=is_training)
        self.layer2 = MakeLayer(BasicBlock, channels[1], layers[1], stride=2, is_training=is_training)

        self.layer3 = MakeLayer(block, channels[2], layers[2], stride=2, is_training=is_training)
        self.layer4 = MakeLayer(block, channels[3], layers[3], stride=2, is_training=is_training)

        self.layer5 = MakeLayer(block, channels[4], layers[4], dilation=2, new_level=False, is_training=is_training)
        if layers[5] == 0:
            self.layer6 = None
        else:
            self.layer6 = MakeLayer(block, channels[5], layers[5], dilation=4, new_level=False, is_training=is_training)

        if layers[6] == 0:
            self.layer7 = None
        else:
            self.layer7 = MakeLayer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False,
                                    is_training=is_training)
        if layers[7] == 0:
            self.layer8 = None
        else:
            self.layer8 = MakeLayer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False,
                                    is_training=is_training)

        if num_classes > 0:
            self.fc = ConvBlock(num_classes, kernel_size=1, is_training=is_training)

    def forward(self, inputs):
        end_points = []

        out = self.conv1.forward(inputs)

        out = self.layer1.forward(out)
        end_points.append(out)

        out = self.layer2.forward(out)
        end_points.append(out)

        out = self.layer3.forward(out)
        end_points.append(out)

        out = self.layer4.forward(out)
        end_points.append(out)

        out = self.layer5.forward(out)
        end_points.append(out)

        if self.layer6 is not None:
            out = self.layer6.forward(out)
            end_points.append(out)

        if self.layer7 is not None:
            out = self.layer7.forward(out)
            end_points.append(out)

        if self.layer8 is not None:
            out = self.layer8.forward(out)
            end_points.append(out)

        if self.out_map:
            out = self.fc.forward(out)
        else:
            out = tf.layers.average_pooling2d(out, self.pool_size, self.pool_size)
            out = self.fc.forward(out)
            out = tf.reshape(out, (out.shape[0], -1))

        if self.out_middle:
            return out, end_points
        else:
            return out


class DRNSeg(object):
    def __init__(self, model_name, num_classes, use_non_local=True, use_bilinear=True, is_training=True):
        self.num_classes = num_classes
        self.use_bilinear = use_bilinear
        self.use_non_local = use_non_local

        if model_name == 'drn_c_26':
            self.base = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], num_classes=num_classes, out_map=True,
                            out_middle=True, is_training=is_training)
        else:
            raise Exception("Unknown architecture...")

        if use_non_local:
            self.nonLocal = NonLocalBlock2D(512)

        self.seg = ConvBlock(num_classes, kernel_size=1)
        # self.decoder = DecoderMudule(48, num_classes, use_bilinear=use_bilinear, is_training=is_training)

    def forward(self, inputs):
        inputs_shape = inputs.get_shape().as_list()
        height, width = inputs_shape[1], inputs_shape[2]

        _, end_points = self.base.forward(inputs)
        out = end_points[-1]
        encoder_features = end_points[1:3]

        if self.use_non_local:
            out = self.nonLocal(out)

        out = self.seg.forward(out)
        # out = self.decoder.forward(out, encoder_features)

        if self.use_bilinear:
            out = tf.image.resize_bilinear(out, (height, width))
        else:
            out = tf.layers.conv2d_transpose(out, self.num_classes, 2, 2, padding='same', use_bias=False,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        pred = tf.argmax(out, axis=3, name='prediction')
        pred = tf.expand_dims(pred, axis=3)

        return out, pred

    def __call__(self, inputs):
        return self.forward(inputs)
