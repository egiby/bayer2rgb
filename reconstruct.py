import numpy as np

import argparse
import sys
import cv2
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from keras.backend import log, mean, square, set_session
import tensorflow as tf

masks = dict()


def get_masks(x_size, y_size):
    green_mask = np.zeros(x_size * y_size, dtype=np.uint8).reshape(x_size, y_size)
    blue_mask = np.zeros(x_size * y_size, dtype=np.uint8).reshape(x_size, y_size)
    red_mask = np.zeros(x_size * y_size, dtype=np.uint8).reshape(x_size, y_size)

    if (x_size, y_size) in masks:
        return masks[(x_size, y_size)]

    for i in range(x_size):
        for j in range(y_size):
            if i % 2 == 0 and j % 2 == 1:
                red_mask[i][j] = 1
            elif i % 2 == 1 and j % 2 == 0:
                blue_mask[i][j] = 1
            else:
                green_mask[i][j] = 1
    masks[(x_size, y_size)] = green_mask, blue_mask, red_mask
    return green_mask, blue_mask, red_mask


def get_bayer_image(source):
    green_mask, blue_mask, red_mask = get_masks(source.shape[0], source.shape[1])
    bayer_image = np.zeros(source.shape, dtype=np.uint8)
    bayer_image[:, :, 0] = source[:, :, 0] * blue_mask
    bayer_image[:, :, 1] = source[:, :, 1] * green_mask
    bayer_image[:, :, 2] = source[:, :, 2] * red_mask
    return bayer_image


def psnr(y_true, y_pred):
    return 10. * log(mean(square(y_pred - y_true))) / np.log(10)


def process(input_path, output_path, model):
    img = cv2.imread(input_path)
    result = model.predict(img.reshape(1, *img.shape) / 255)
    cv2.imwrite(output_path, (result[0] * 255).astype(np.uint8))


def test(input_path, output_path, model):
    img = cv2.imread(input_path)
    bayer_image = get_bayer_image(img)
    start = time.time()
    metrics = model.evaluate(bayer_image.reshape(1, *img.shape) / 255, img.reshape(1, *img.shape) / 255)
    end = time.time()
    metrics[0] = -metrics[0]

    print('psnr = {:.3f}'.format(metrics[0]))
    print('mse =', metrics[1])
    print('time = {:.3f}s'.format(end - start))
    print('performance = {:.3f}MB/s'.format(img.shape[0] * img.shape[1] * 3 / 1024 / 1024 / (end - start)))

    result = model.predict(bayer_image.reshape(1, *img.shape) / 255)
    cv2.imwrite(output_path, (result[0] * 255).astype(np.uint8))


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='Input path. Should be color image')
    parser.add_argument('--model', help='Model path. Default is ./model.h5', default='./model.h5')
    parser.add_argument('--mode', help='Mode. Could be "test" or "process". Default is "process"', default='process')
    parser.add_argument('--output', help='Output path. Default is ./Reconstructed.bmp', default='./Reconstructed.bmp')
    parser.add_argument('--gpu', help='Flag for enabling gpu usage', action='store_true')
    parser.add_argument('--num_workers', type=int, help='Number of cpu cores used for computing', default=-1)

    return parser.parse_args(argv)


def configure_tensorflow(num_workers, use_gpu):
    num_gpu = 1 if use_gpu else 0
    if num_workers == -1:
        config = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': num_gpu})
    else:
        config = tf.ConfigProto(intra_op_parallelism_threads=num_workers, inter_op_parallelism_threads=num_workers,
                                allow_soft_placement=True, device_count={'CPU': 1, 'GPU': num_gpu})

    session = tf.Session(config=config)
    set_session(session)


def main(args):
    configure_tensorflow(args.num_workers, args.gpu)
    model = load_model(args.model, custom_objects={'psnr': psnr})

    if args.mode == 'process':
        process(args.input, args.output, model)
    elif args.mode == 'test':
        test(args.input, args.output, model)
    else:
        raise ValueError('mode should be "process" or "test"')


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
