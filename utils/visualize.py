import os
import numpy as np
from PIL import Image


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


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
        lines.append('%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
            'mean_IU', mean_IU * 100, 'mean_IU_no_back', mean_IU_no_back * 100, 'mean_pixel_ACC', mean_pixel_acc * 100))
    else:
        print(mean_pixel_acc)
        lines.append('%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IU', mean_IU * 100, 'mean_pixel_ACC', mean_pixel_acc * 100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line
