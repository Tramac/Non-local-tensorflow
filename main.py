'''
Date: 2019.01.24
Author: zxm
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from utils.data_generator import DataGenerator
from model import Model
from trainer import Trainer

parse = argparse.ArgumentParser()
parse.add_argument("--data_dir", default='./data', help="Path to data")
parse.add_argument("--dataset", default='VOC2012', help="Name of dataset, VOC2012/VOC2007")
parse.add_argument("--input_height", type=int, default=224, help="Input height of image and label")
parse.add_argument("--input_width", type=int, default=224, help="Input width of image and label")
parse.add_argument("--model", default='DRN', help="Name of model")
parse.add_argument("--checkpoint_dir", default='./checkpoint', help="Path to save model")
parse.add_argument("--num_classes", type=int, default=21, help="Number of classes.")
parse.add_argument("--learning_rate", default=1e-2, help="Learning rate for training")
parse.add_argument("--num_epochs", type=int, default=200, help="Epochs of train")
parse.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parse.add_argument("--is_training", type=bool, default=True, help="train or test")
config = parse.parse_args()


def main():
    sess = tf.Session()

    train_data_loader = DataGenerator(config, 'train')
    valid_data_loader = DataGenerator(config, 'valid')

    model = Model(sess, config)

    trainer = Trainer(sess, model, train_data_loader, valid_data_loader, config)

    if config.is_training:
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    main()
