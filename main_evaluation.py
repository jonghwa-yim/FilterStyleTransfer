""" Extract target style and stylize target image

This module extracts parameters of style from style image, and stylize target images.
Both style image and target image can be multiple images in each directories.
"""
__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import argparse
import os
import time

import numpy as np
import torch.optim

from dataset import default_loader
from stylizer import style_prediction as Stylizer


def argument_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Stylizer')
    parser.add_argument('--styleDir', default='', type=str, metavar='PATH',
                        help='path to dataset.')
    parser.add_argument('--targetDir', default='', type=str, metavar='PATH',
                        help='path to dataset.')
    parser.add_argument('--styleSetTxt', default='', type=str, metavar='PATH',
                        help='path to style set list file.')
    parser.add_argument('--targetSetTxt', default='', type=str, metavar='PATH',
                        help='path to target set list file.')
    parser.add_argument('-m', '--modelPath', default='', type=str, metavar='PATH',
                        help='path to CNN auto-encoder model')
    parser.add_argument('--outputDir', default='', type=str, metavar='PATH',
                        help='path to store output canvas')
    parser.add_argument('-s', '--imageSize', default=256, type=int, metavar='N',
                        help='input image size. Default: 256')
    parser.add_argument('--numSampling', default=4096, type=int, metavar='N',
                        help='number of sampling pixels from input-output pair')

    parser.add_argument('--reg_weight', default=0.5, type=float, metavar='VAL',
                        help='Weight of regularization term. 0 is no regularization.')

    parser.add_argument('--uncertainty', type=str, choices=['default', 'default2', 'mcdrop', 'aleatoric', 'aleatoric_combined'], help='Type of uncertainty as variance. [default, mcdrop, aleatoric].')
    parser.add_argument('-c', '--correlation', action='store_true')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()
    return args


def get_error_metric_value(f_gt_image, input_stylized, img_size):
    # ========================== Loading GT image ==========================

    gt_img = default_loader(f_gt_image)
    gt_img = gt_img.resize((img_size, img_size)) #, PIL.Image.ANTIALIAS)
    gt_img = np.array(gt_img) / 255
    if np.issubdtype(input_stylized.dtype, np.uint8):
        input_stylized = input_stylized / 255

    # ======================= Computing psnr metric ========================
    ret = Stylizer.get_error_metric(input_stylized, gt_img)
    # ======================================================================
    return ret


def read_style_dataset(styleset_txt):
    fp = open(styleset_txt, 'r')
    buf = fp.readlines()
    fp.close()

    image_names = []
    for line in buf:
        image_names.append(line.strip().split(' ')[1])

    # style_filenames = [os.path.join(args.testDir, x.strip().split(' ')[1]) for x in image_names]
    return image_names


def read_target_dataset(targetset_txt):
    fp = open(targetset_txt, 'r')
    buf = fp.readlines()
    fp.close()

    image_names = []
    for i, line in enumerate(buf):
        if i%22 == 0:
            image_names.append(line.strip().split(' ')[0])

    # style_filenames = [os.path.join(args.testDir, x.strip().split(' ')[1]) for x in image_names]
    return image_names


def read_dataset(style_dir_path, target_dir_path):
    styles = os.listdir(style_dir_path)
    targets = []

    for fname in os.listdir(style_dir_path):
        if fname.find("_lut_") != -1:
            orgname = fname.split('_lut_')[0] + fname[fname.rfind('.'):]
            targets.append(orgname)
        elif fname.find("_pil_") != -1:
            orgname = fname.split('_pil_')[0] + fname[fname.rfind('.'):]
            targets.append(orgname)

        assert os.path.exists(target_dir_path + '/' + orgname)
    return styles, targets


def main():
    img_size = args.imageSize
    num_styles = 26

    print('Regularization Weight : ', args.reg_weight)
    print('MC Dropout : ', args.uncertainty)
    print('Correlation : ', args.correlation)

    print('===> Loading datasets')
    if len(args.styleSetTxt) == 0 or len(args.targetSetTxt) == 0:
        style_img_names, target_img_names = read_dataset(args.styleDir, args.targetDir)
    else:
        style_img_names = read_style_dataset(args.styleSetTxt)
        target_img_names = read_target_dataset(args.targetSetTxt)

    total_psnr = 0
    inter_psnr = 0
    total_deltaE = 0
    inter_deltaE = 0
    total_extraction_time = 0
    total_application_time = 0
    cnt = 0
    total_cnt = 0

    for idx, f_style_img in enumerate(style_img_names):
        # start = time.time()
        id_num = idx % num_styles
        if f_style_img.find('_pil_') != -1:
            sty = f_style_img[f_style_img.find('_pil_'):f_style_img.rfind('.')]
        elif f_style_img.find('_lut_') != -1:
            sty = f_style_img[f_style_img.find('_lut_'):f_style_img.rfind('.')]
        else:
            sty = '_' + str(id_num)

        #if id_num == 25 or id_num == 5 or id_num == 7 or id_num == 11 or id_num==13 or id_num==18 or id_num==21 or id_num==24:   # temporary code due to bug in filterizing code
        #    continue

        f_style_img_path = os.path.join(args.styleDir, f_style_img)
        stylized = default_loader(f_style_img_path)

        start_time = time.time()
        restored, res_mean, res_var = Stylizer.ReconstructorCNN.infer_original_image_w_uncertainty(cnn_model, stylized, img_size, args.uncertainty)
        total_extraction_time += (time.time() - start_time)

        stylized = stylized.resize((img_size, img_size)) #, PIL.Image.ANTIALIAS)
        stylized = np.array(stylized, dtype=np.float32) / 255

        # ===================== Computing stylization model =====================
        sampled_restored, sampled_stylized, sampled_var = Stylizer.get_sampled_data_ext(restored, stylized, res_var, num_sampling=args.numSampling)
        stylization_model = Stylizer.get_stylization_model(sampled_restored, sampled_stylized,
                                                           args.reg_weight, sampled_var)
        # ======================================================================

        m_psnr = 0
        m_deltaE = 0
        m_application_time = 0

        for f_target_img in target_img_names:
            f_target_img_path = os.path.join(args.targetDir, f_target_img)
            target_in = default_loader(f_target_img_path)

            target_in = target_in.resize((img_size, img_size)) #, PIL.Image.ANTIALIAS)
            target_in = np.array(target_in, dtype=np.float32) / 255

            # ========== Applying stylizing parameters on the input image ==========

            start_time = time.time()
            target_stylized = Stylizer.get_stylized_image_gpu(target_in, stylization_model)
            m_application_time += (time.time() - start_time)

            f_target_gt = f_target_img[:f_target_img.rfind('.')] + sty + f_target_img[f_target_img.rfind('.'):]
            f_target_gt_path = os.path.join(args.styleDir, f_target_gt)
            metric = get_error_metric_value(f_target_gt_path, target_stylized, img_size)
            m_psnr += metric['psnr']
            m_deltaE += metric['deltaE']

        m_application_time /= len(target_img_names)
        total_application_time += m_application_time
        m_psnr /= len(target_img_names)
        m_deltaE /= len(target_img_names)
        total_psnr += m_psnr
        total_deltaE += m_deltaE
        total_cnt += 1
        inter_psnr += m_psnr
        inter_deltaE += m_deltaE
        cnt += 1
        # if idx % int(num_styles/2) == 0:
        if id_num == 0 or id_num == 13:
            # print('iteration ', idx + 1, ' / ' + str(len(style_img_names)) + ' , average psnr : ', total_psnr / (idx+1))
            print('iter', idx + 1, '/ ' + str(len(style_img_names)) + ' , batch psnr :',
                  '%.4f' %(inter_psnr / cnt), ', avg psnr :', '%.4f' %(total_psnr / total_cnt), ', batch dE00 :',
                  '%.4f' %(inter_deltaE / cnt), ', avg dE00 :', '%.4f' %(total_deltaE / total_cnt))
            cnt = 0
            inter_psnr = 0
            inter_deltaE = 0
        # print('time : ', time.time() - start)
    # total_psnr /= len(style_img_names)
    total_psnr /= total_cnt
    total_deltaE /= total_cnt
    total_extraction_time /= total_cnt
    total_application_time /= total_cnt

    return total_psnr, total_deltaE, total_extraction_time, total_application_time


if __name__ == '__main__':
    args = argument_parser()

    device = torch.device("cuda")
    Stylizer.MODE_CORRELATION = args.correlation

    # ========================= Loading CNN model ==========================
    cnn_model = Stylizer.ReconstructorCNN.load_cnn_model(args.modelPath, args.uncertainty)

    psnr, deltaE, avg_time_extr, avg_time_appl = main()

    print('Total Average PSNR : ', psnr)
    print('Total Average deltaE : ', deltaE)
    print('Total Average runtime (style extraction) : ', avg_time_extr)
    print('Total Average runtime (style application) : ', avg_time_appl)
    print('done')
