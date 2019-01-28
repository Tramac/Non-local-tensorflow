from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


def hist_info(num_classes, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < num_classes)
    labeled = np.sum(k)
    correct = np.sum(pred[k] == gt[k])

    return np.bincount(num_classes * gt[k].astype(int) + pred[k].astype(int), minlength=num_classes ** 2).reshape(
        num_classes, num_classes), labeled, correct


def compute_mean_iou(pred, label, num_classes):
    batch_size = pred.shape[0]
    pred_flat = tf.reshape(pred, (batch_size, -1))
    label_flat = tf.reshape(label, (batch_size, -1))

    miou = tf.metrics.mean_iou(label_flat, pred_flat, num_classes)

    return miou


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100, 'mean_IU_no_back', mean_IU_no_back * 100,'mean_pixel_ACC', mean_pixel_acc * 100))
    else:
        print(mean_pixel_acc)
        lines.append('%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100, 'mean_pixel_ACC', mean_pixel_acc * 100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line
