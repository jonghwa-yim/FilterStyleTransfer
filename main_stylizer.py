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

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np

from dataset import default_loader
from stylizer import style_prediction as Stylizer


def argument_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Stylizer')
    parser.add_argument('-S', '--styleImageDir', default='', type=str, metavar='PATH',
                        help='path to stylized images.')
    parser.add_argument('--styleImageOrgDir', default='', type=str, metavar='PATH',
                        help='path to originals of stylized images.')
    parser.add_argument('-T', '--targetImageDir', default='', type=str, metavar='PATH',
                        help='path to target images')
    parser.add_argument('--targetGTDir', default='', type=str, metavar='PATH',
                        help='path to target-to-stylize GT images')
    parser.add_argument('-m', '--modelPath', default='', type=str, metavar='PATH',
                        help='path to CNN auto-encoder model')
    parser.add_argument('-s', '--imageSize', default=256, type=int, metavar='N',
                        help='input image size. Default: 256')
    parser.add_argument('--numSampling', default=4096, type=int, metavar='N',
                        help='number of sampling pixels from input-output pair')
    parser.add_argument('-O', '--drawOutput', dest='drawOutput', action='store_true',
                        help='Draw output canvas')
    parser.add_argument('--outputDir', default='', type=str, metavar='PATH',
                        help='path to store output canvas')
    parser.add_argument('-L', '--LargeOutput', dest='LargeOutput', action='store_true',
                        help='Draw large output canvas')
    parser.add_argument('-M', '--outputMode', default=1, type=int, metavar='N',
                        help='Drawing mode. 1 is large output, 2 is single output')
    parser.add_argument('--reg_weight', default=1, type=float, metavar='VAL',
                        help='Weight of regularization term. 0 is no regularization.')

    parser.add_argument('--uncertainty', type=str, choices=['default', 'mcdrop', 'aleatoric', 'aleatoric_combined'], default='default', help='Type of uncertainty as variance. [default, mcdrop, aleatoric].')
    # parser.add_argument('-v', '--variance', action='store_true')
    args = parser.parse_args()
    return args


def create_merged_image_base(base_images, stylization_model, base_img_size=None, f_style_img=''):
    """

    :param base_images: Images comprising the top row of output canvas. (Usually style-related images)
    :param stylization_model: regression models
    :param base_img_size: set if manual canvas size is required
    :param f_style_img:
    :return:
    """
    if base_img_size is None:
        width = img_size * 6
        height = img_size * 3
    else:
        width = base_img_size[0]
        height = base_img_size[1]
    merged_img = PIL.Image.new("RGB", (width, height))

    if len(args.styleImageOrgDir) != 0:
        f_org_of_style_img = get_name_of_original_of_stylized(f_style_img)
        if f_org_of_style_img is not None:
            f_org_of_style_path = os.path.join(args.styleImageOrgDir, f_org_of_style_img)
            org = default_loader(f_org_of_style_path)
            org = org.resize((img_size, img_size), PIL.Image.ANTIALIAS)
            merged_img.paste(im=org, box=(0, 0))
            merged_img.paste(im=org, box=(img_size * 4, 0))

            input_org = np.array(org) / 255
            input_stylized = Stylizer.get_stylized_image(input_org, stylization_model)
            tmp = np.clip(input_stylized, 0, 1)
            tmp = np.uint8(tmp * 255)
            tmp = PIL.Image.fromarray(tmp, 'RGB')
            merged_img.paste(im=tmp, box=(img_size * 5, 0))

    for i, tmp in enumerate(base_images):
        tmp = np.clip(tmp, 0, 1)
        tmp = np.uint8(tmp * 255)
        tmp = PIL.Image.fromarray(tmp, 'RGB')
        merged_img.paste(im=tmp, box=(img_size * (i + 1), 0))

    return merged_img


def paste_sub_image_on_canvas(merged_img, source_image, location):
    if not np.issubdtype(source_image.dtype, np.uint8):
        tmp = np.clip(source_image, 0, 1)
        tmp = np.uint8(tmp * 255)
    else:
        tmp = source_image
    tmp = PIL.Image.fromarray(tmp, 'RGB')
    merged_img.paste(im=tmp, box=location)
    return True


def save_output_canvas(merged_img, cnt):
    output_img_name = output_folder + f_style_img[:-4] + '_stylized_' + str(
        cnt) + '.png'
    # plt.imsave(output_img_name, merged_img, dpi=250)
    plt.imsave(output_img_name, np.asarray(merged_img), dpi=250)
    return True


# def get_error_metric_value(gt_img_name, input_stylized):
#     # ========================== Loading GT image ==========================
#     f_gt_image = os.path.join(target_gt_path, gt_img_name)
#
#     gt_img = default_loader(f_gt_image)
#     gt_img = gt_img.resize((img_size, img_size), PIL.Image.ANTIALIAS)
#     if not np.issubdtype(input_stylized.dtype, np.uint8):
#         gt_img = np.array(gt_img) / 255
#     else:
#         gt_img = np.array(gt_img)
#
#     # ======================= Computing psnr metric ========================
#     ret = Stylizer.get_error_metric(input_stylized, gt_img)
#     # ======================================================================
#     return ret


def draw_large_output(input_img, stylized, output, input_stylized, cnt):
    img_size = input_img.shape
    merged_img = PIL.Image.new("RGB", (img_size[1] * 4, img_size[0]))

    loc = (0, 0)
    paste_sub_image_on_canvas(merged_img, input_img, loc)

    loc = (img_size[1], 0)
    paste_sub_image_on_canvas(merged_img, stylized, loc)

    loc = (img_size[1]*2, 0)
    paste_sub_image_on_canvas(merged_img, output, loc)

    loc = (img_size[1]*3, 0)
    paste_sub_image_on_canvas(merged_img, input_stylized, loc)

    save_output_canvas(merged_img, cnt)
    return


def draw_large_output_rect(input_img, stylized, output, input_stylized, cnt):
    img_size = input_img.shape
    merged_img = PIL.Image.new("RGB", (img_size[1] * 2, img_size[0] * 2))

    loc = (img_size[1], 0)
    paste_sub_image_on_canvas(merged_img, input_img, loc)

    loc = (0, img_size[0])
    paste_sub_image_on_canvas(merged_img, stylized, loc)

    loc = (0, 0)
    paste_sub_image_on_canvas(merged_img, output, loc)

    loc = (img_size[1], img_size[0])
    paste_sub_image_on_canvas(merged_img, input_stylized, loc)

    save_output_canvas(merged_img, cnt)
    return


def draw_single_output(input_stylized, target_img):
    img_size = input_stylized.shape
    merged_img = PIL.Image.new("RGB", (img_size[1], img_size[0]))

    loc = (0, 0)
    paste_sub_image_on_canvas(merged_img, input_stylized, loc)

    save_output_canvas(merged_img, target_img)
    return


def get_name_of_original_of_stylized(f_style_img):
    ret = None
    if f_style_img.find('_pil_') != -1:
        ret = f_style_img[:f_style_img.find('_pil_')] + f_style_img[-4:]
    elif f_style_img.find('_lut_') != -1:
        ret = f_style_img[:f_style_img.find('_lut_')] + f_style_img[-4:]
    return ret


def get_name_of_target_stylized_gt(f_target_img, f_style_img):
    ret = None
    if f_style_img.find('_pil_') != -1:
        ret = f_target_img[:f_target_img.rfind('.')] + f_style_img[f_style_img.find('_pil_'):]
    elif f_style_img.find('_lut_') != -1:
        ret = f_target_img[:f_target_img.rfind('.')] + f_style_img[f_style_img.find('_lut_'):]
    return ret


def main(f_style_img):
    img_size = args.imageSize
    # ======================= Computing model output =======================
    f_style_img_path = os.path.join(args.styleImageDir, f_style_img)
    # print(f_style_img)
    stylized = default_loader(f_style_img_path)

    restored, res_mean, res_var = Stylizer.ReconstructorCNN.infer_original_image_w_uncertainty(cnn_model, stylized, img_size, args.uncertainty)

    stylized = stylized.resize((img_size, img_size), PIL.Image.ANTIALIAS)
    stylized = np.array(stylized) / 255

    # ===================== Computing stylization model =====================
    sampled = Stylizer.get_sampled_data_ext(restored, stylized, res_var, num_sampling=args.numSampling)
    sampled_restored, sampled_stylized, sampled_var = sampled
    stylization_model = Stylizer.get_stylization_model(sampled_restored, sampled_stylized, args.reg_weight, sampled_var)
    # ======================================================================
    if False:
        # save parameters in model
        fp = open(f_style_img + '.txt', 'w')
        for i in range(3):
            coef = stylization_model[i].coef_
            fp.write(str(coef).replace('\n', '') + '\n')
        fp.close()
        return 0

    # ======================================================================
    # Some variables
    idx = 0
    cnt = 0
    total_time = 0
    base_images = []
    m_psnr = 0
    m_deltaE = 0
    metric_calc_cnt = 0

    # =============== Generating the top row of merged image ===============
    if args.drawOutput:
        if args.outputMode == 0:
            output_stylized = Stylizer.get_stylized_image(restored, stylization_model)
            base_images = [stylized, restored, output_stylized]
            starting_row = 1
            # else:
            #     base_img_size = (img_size * 2, img_size)
            #     starting_row = 0
            merged_img = create_merged_image_base(base_images, stylization_model, f_style_img=f_style_img)
        elif args.outputMode == 2:
            draw_single_output(stylized, 'reference')
            draw_single_output(restored, 'restored_original')
    # ======================================================================

    for i, target_img in enumerate(os.listdir(target_path)):
        # f_input_image = path_org + in_num + '.png'
        f_input_image = os.path.join(target_path, target_img)

        input_img = default_loader(f_input_image)
        if args.outputMode != 0:
            target_size = (img_size, img_size) # (int(input_img.size[0] / 2), int(input_img.size[1] / 2))
        else:
            target_size = (img_size, img_size)
        if args.outputMode != 2:
            input_img = input_img.resize(target_size, PIL.Image.ANTIALIAS)
        input_img = np.array(input_img, dtype=np.float32) / 255

        # ========== Applying stylizing parameters on the input image ==========
        start_time = time.time()
        input_stylized = Stylizer.get_stylized_image(input_img, stylization_model)
        total_time += (time.time() - start_time)

        # ======================= Calculate error metric =======================
        if len(args.targetGTDir) != 0:
            f_target_stylized_gt = get_name_of_target_stylized_gt(target_img, f_style_img)
            if f_target_stylized_gt is not None:
                f_target_stylized_gt_path = os.path.join(args.targetGTDir, f_target_stylized_gt)
                try:
                    target_gt_img = default_loader(f_target_stylized_gt_path)
                except FileNotFoundError:
                    # print('Error : target,' + target_img + ',\'s GT image does not exits. Style image: ' + f_style_img)
                    pass
                else:
                    if target_gt_img.size != input_stylized.shape[0:2]:
                        target_gt_img = target_gt_img.resize(input_stylized.shape[0:2], PIL.Image.ANTIALIAS)
                    target_gt_img = np.array(target_gt_img, dtype=np.float32) / 255
                    metric = Stylizer.get_error_metric(input_stylized, target_gt_img)
                    m_psnr += metric['psnr']
                    m_deltaE += metric['deltaE']
                    metric_calc_cnt += 1

        # =================== Pasting onto the merged image ====================
        if args.drawOutput:
            if args.outputMode == 1:
                draw_large_output_rect(input_img, stylized, restored, input_stylized, cnt)
                cnt += 1
                continue
            elif args.outputMode == 2:
                draw_single_output(input_img, 'input_'+target_img)
                draw_single_output(input_stylized, 'output_'+target_img)
                continue
            idx += 1

            loc = (img_size * ((idx - 1) % 3) * 2, img_size * (starting_row + ((idx - 1)) // 3))
            paste_sub_image_on_canvas(merged_img, input_img, loc)

            loc = (img_size * ((2 * idx - 1) % 6), img_size * (starting_row + ((idx - 1)) // 3))
            paste_sub_image_on_canvas(merged_img, input_stylized, loc)

            if idx == NUM_OUTPUT_PER_IMG:
                save_output_canvas(merged_img, cnt)
                if cnt >= MAX_OUTPUT_NUM:
                    break
                merged_img = create_merged_image_base(base_images, stylization_model, f_style_img=f_style_img)
                idx = 0
                cnt += 1
            elif (i + 1) == len(os.listdir(target_path)):
                save_output_canvas(merged_img, cnt)
                # ========================== End of for loop ===========================

    if metric_calc_cnt != 0:
        m_psnr /= metric_calc_cnt
        m_deltaE /= metric_calc_cnt
    print('Style : ', f_style_img, ', Average psnr : ', m_psnr)
    # print('Average time spent on stylization : ', total_time / (i + 1))

    return m_psnr, m_deltaE


if __name__ == '__main__':
    args = argument_parser()

    img_size = args.imageSize
    MAX_OUTPUT_NUM = 2
    NUM_OUTPUT_PER_IMG = 6
    if len(args.outputDir) > 0 and not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    # ========================= Loading CNN model ==========================
    cnn_model = Stylizer.ReconstructorCNN.load_cnn_model(args.modelPath, args.uncertainty)

    psnr = 0.0
    deltaE = 0.0
    style_count = 0
    cur_stage = 1

    for f_style_img in os.listdir(args.styleImageDir):
        if len(args.outputDir) > 0 and not os.path.exists(
                os.path.join(args.outputDir, f_style_img[:-4])):
            os.mkdir(os.path.join(args.outputDir, f_style_img[:-4]))
        output_folder = os.path.join(args.outputDir, f_style_img[:-4] + '/')

        if True:
            # target_folder = 'ux_test_content'
            target_path = args.targetImageDir
            target_gt_path = args.targetGTDir
            ret = main(f_style_img)
            psnr += ret[0]
            deltaE += ret[1]
        else:
            for target_folder in os.listdir(args.targetImageDir):
                target_path = os.path.join(args.targetImageDir, target_folder)
                target_gt_path = os.path.join(args.targetGTDir, target_folder)
                main(f_style_img)

        style_count += 1
        if style_count >= (cur_stage / 10 * len(os.listdir(args.styleImageDir))):
            print(cur_stage * 10, '% done. Intermediate PSNR : ', psnr / style_count, '. dE00 : ', deltaE / style_count)
            cur_stage += 1

    print('Total Average PSNR : ', psnr / len(os.listdir(args.styleImageDir)))
    print('Total Average deltaE00 : ', deltaE / len(os.listdir(args.styleImageDir)))
    print('done')
