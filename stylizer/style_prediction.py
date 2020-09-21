""" Style Prediction

This is the major methods to get style parameters and apply styles on input image.
You can first infer original image using class ReconstructorCNN
Then you will get stylization model using get_stylization_model
Finally, you will want to apply style on target image using get_stylized_image

Example:
    # Getting stylization model:
    cnn_model = ReconstructorCNN.load_cnn_model(model_path)
    stylized_image = default_loader(stylized_image_full_path)
    inferred_original = ReconstructorCNN.infer_original_image(cnn_model, stylized_image, 256)
    stylized_image = stylized_image.resize((img_size, img_size), PIL.Image.ANTIALIAS)
    stylized_image = np.array(stylized) / 255
    sampled_original, sampled_stylized = get_sampled_data(inferred_original, stylized_image)
    stylization_model = get_stylization_model(inferred_original_image, stylized_image, 0.5)

    # or simply you can call:
    cnn_model = ReconstructorCNN.load_cnn_model(model_path)
    stylized_image = default_loader(image_full_path)
    stylization_model = get_stylization_model_auto(cnn_model, image_size, stylized_image, num_sampling, 0.5)

    # Applying styles on target image:
    stylized_target_image = get_stylized_image(target_image, stylization_model)

"""
__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import math
import random

import PIL.Image
import numpy as np
import scipy.stats
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from data import input_transform
from stylizer.color_regression import JHColorRegression
from architecture.transformer_net import TransformerNet
# from architecture.unet_mod import UNetMod
# from architecture.transformer_net_mcdrop import TransformerNet_MCDrop
# from architecture.transformer_net_aleatoric import TransformerNet_Aleatoric
# from architecture.transformer_net_aleatoric2 import TransformerNet_Aleatoric2

# MODE_UNCERTAINTY = {'default': 'TransformerNet',
#                     'default2': 'TransformerNet_Aleatoric',
#                     'mcdrop': 'TransformerNet_MCDrop',
#                     'aleatoric': 'TransformerNet_Aleatoric',
#                     'aleatoric_combined': 'TransformerNet_Aleatoric'}
POLY_DEGREE = 3
MODE_CORRELATION = False


def get_sampled_data(image1, image2, num_sampling=4096):
    image1 = image1.reshape((-1, 3))
    image2 = image2.reshape((-1, 3))

    if image1.shape != image2.shape:
        raise Exception('Discrepancy in shape between two input images.')
    if num_sampling > image1.shape[0]:
        num_sampling = image1.shape[0]

    img_norm_sampled = np.zeros((num_sampling, 3), dtype=np.float32)
    img_stylized_sampled = np.zeros((num_sampling, 3), dtype=np.float32)
    xs = random.sample(range(image1.shape[0]), num_sampling)

    for i in range(num_sampling):
        img_norm_sampled[i] = (image1[xs[i]])
        img_stylized_sampled[i] = (image2[xs[i]])

    return img_norm_sampled, img_stylized_sampled


def get_sampled_data_ext(image1, image2, var_map, num_sampling=4096):
    image1 = image1.reshape((-1, 3))
    image2 = image2.reshape((-1, 3))
    if var_map is not None:
        var_map = var_map.reshape((-1))

    if image1.shape != image2.shape:
        raise Exception('Discrepancy in shape between two input images.')
    if num_sampling > image1.shape[0]:
        num_sampling = image1.shape[0]

    img_norm_sampled = np.zeros((num_sampling, 3), dtype=np.float32)
    img_stylized_sampled = np.zeros((num_sampling, 3), dtype=np.float32)
    if var_map is not None:
        var_sampled = np.zeros((num_sampling), dtype=np.float32)
    else:
        var_sampled = None

    # =========================== Index Sampling ===========================
    if False:
        xs = random.sample(range(image1.shape[0]), num_sampling)
    elif False:
        xs = np.argsort(var_map) # working on it
        var_map = None
        var_sampled = None
    elif True:
        xs = []
        for i in range(num_sampling):
            idx = int(image1.shape[0] / num_sampling * i)
            xs.append(idx)
    # ======================================================================

    for i in range(num_sampling):
        img_norm_sampled[i] = (image1[xs[i]])
        img_stylized_sampled[i] = (image2[xs[i]])
        if var_map is not None:
            var_sampled[i] = (var_map[xs[i]])

    return img_norm_sampled, img_stylized_sampled, var_sampled


def get_stylization_model_auto(cnn_model, img_size, img_stylized, num_sampling, reg_weight=0.5):
    # ======================= Computing model output =======================
    output = ReconstructorCNN.infer_original_image(cnn_model, img_stylized, img_size)

    # ===================== Computing stylization model =====================
    stylized = img_stylized.resize((img_size, img_size), PIL.Image.ANTIALIAS)
    stylized = np.array(stylized) / 255
    sampled_output, sampled_stylized = get_sampled_data(output, stylized,
                                                        num_sampling=num_sampling)

    img_x = sampled_output.transpose(1, 0)
    img_y = sampled_stylized.transpose(1, 0)

    img_input_r, img_input_g, img_input_b = _get_polynomial_features_rgb(img_x)

    if True:
        model_r = JHColorRegression().fit(img_input_r, img_y[0], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_g = JHColorRegression().fit(img_input_g, img_y[1], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_b = JHColorRegression().fit(img_input_b, img_y[2], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
    else:
        # Ordinary least squares Linear Regression.
        model_r = LinearRegression().fit(img_input_r, img_y[0])
        model_g = LinearRegression().fit(img_input_g, img_y[1])
        model_b = LinearRegression().fit(img_input_b, img_y[2])

    return model_r, model_g, model_b


def get_stylization_model(image_in, img_stylized, reg_weight=0.5, vars=None):
    img_x = image_in.transpose(1, 0)
    img_y = img_stylized.transpose(1, 0)

    img_input_r, img_input_g, img_input_b = _get_polynomial_features_rgb(img_x)

    if vars is None:
        model_r = JHColorRegression().fit(img_input_r, img_y[0], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_g = JHColorRegression().fit(img_input_g, img_y[1], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_b = JHColorRegression().fit(img_input_b, img_y[2], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
    elif vars is not None:
        var_map = np.zeros((len(vars), len(vars)), np.float32)
        for i in range(len(vars)):
            var_map[i][i] = vars[i]
        model_r = JHColorRegression().fit_var(img_input_r, var_map, img_y[0], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_g = JHColorRegression().fit_var(img_input_g, var_map, img_y[1], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
        model_b = JHColorRegression().fit_var(img_input_b, var_map, img_y[2], alpha=reg_weight, MODE_CORRELATION=MODE_CORRELATION)
    # else:
        # Ordinary least squares Linear Regression.
        # model_r = LinearRegression().fit(img_input_r, img_y[0])
        # model_g = LinearRegression().fit(img_input_g, img_y[1])
        # model_b = LinearRegression().fit(img_input_b, img_y[2])

    return model_r, model_g, model_b


def get_stylized_image(image_in, stylization_model):
    input_shape = image_in.shape

    img_x = image_in.reshape((-1, 3))
    img_x = img_x.transpose(1, 0)

    img_input_r, img_input_g, img_input_b = _get_polynomial_features_rgb(img_x)

    # rvalue = stylization_model[0].predict(img_input_r)
    # gvalue = stylization_model[1].predict(img_input_g)
    # bvalue = stylization_model[2].predict(img_input_b)
    rvalue = np.dot(img_input_r, stylization_model[0].coef_)
    gvalue = np.dot(img_input_g, stylization_model[1].coef_)
    bvalue = np.dot(img_input_b, stylization_model[2].coef_)
    image_out = np.stack((rvalue, gvalue, bvalue))

    image_out = image_out.transpose(1, 0)
    image_out = image_out.reshape(input_shape)

    return image_out


def get_stylized_image_gpu(image_in, stylization_model):
    input_shape = image_in.shape

    img_x = image_in.reshape((-1, 3))
    img_x = img_x.transpose(1, 0)

    img_input_r, img_input_g, img_input_b = _get_polynomial_features_rgb_gpu(img_x)

    rvalue = np.dot(img_input_r, stylization_model[0].coef_)
    gvalue = np.dot(img_input_g, stylization_model[1].coef_)
    bvalue = np.dot(img_input_b, stylization_model[2].coef_)
    image_out = np.stack((rvalue, gvalue, bvalue))

    image_out = image_out.transpose(1, 0)
    image_out = image_out.reshape(input_shape)

    return image_out


def _get_polynomial_features_rgb_gpu(img_x):

    with torch.no_grad():
        img_tt = torch.Tensor(img_x).cuda()
        img_input_r = torch.ones((10, img_tt.shape[1]))
        img_input_g = torch.ones((10, img_tt.shape[1]))
        img_input_b = torch.ones((10, img_tt.shape[1]))

        img_x_r = torch.zeros((3, img_tt.shape[1]))

        img_x_r[0] = torch.pow(img_tt[0], 1)
        img_x_r[1] = torch.pow(img_tt[0], 2)
        img_x_r[2] = torch.pow(img_tt[0], 3)

        img_x_g = torch.zeros((3, img_tt.shape[1]))

        img_x_g[0] = torch.pow(img_tt[1], 1)
        img_x_g[1] = torch.pow(img_tt[1], 2)
        img_x_g[2] = torch.pow(img_tt[1], 3)

        img_x_b = torch.zeros((3, img_tt.shape[1]))

        img_x_b[0] = torch.pow(img_tt[2], 1)
        img_x_b[1] = torch.pow(img_tt[2], 2)
        img_x_b[2] = torch.pow(img_tt[2], 3)

        img_input_r[1] = img_x_r[0]
        img_input_r[2] = img_x_r[1]
        img_input_r[3] = img_x_r[2]
        img_input_r[4] = img_x_g[0]
        img_input_r[5] = img_x_g[1]
        img_input_r[6] = img_x_g[2]
        img_input_r[7] = img_x_b[0]
        img_input_r[8] = img_x_b[1]
        img_input_r[9] = img_x_b[2]

        img_input_g[1] = img_x_g[0]
        img_input_g[2] = img_x_g[1]
        img_input_g[3] = img_x_g[2]
        img_input_g[4] = img_x_b[0]
        img_input_g[5] = img_x_b[1]
        img_input_g[6] = img_x_b[2]
        img_input_g[7] = img_x_r[0]
        img_input_g[8] = img_x_r[1]
        img_input_g[9] = img_x_r[2]

        img_input_b[1] = img_x_b[0]
        img_input_b[2] = img_x_b[1]
        img_input_b[3] = img_x_b[2]
        img_input_b[4] = img_x_r[0]
        img_input_b[5] = img_x_r[1]
        img_input_b[6] = img_x_r[2]
        img_input_b[7] = img_x_g[0]
        img_input_b[8] = img_x_g[1]
        img_input_b[9] = img_x_g[2]

    return img_input_r.transpose(1, 0), img_input_g.transpose(1, 0), img_input_b.transpose(1, 0)


def _get_polynomial_features_rgb(img_x):

    img_x_r = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False).fit_transform(
        img_x[0].reshape(-1, 1))
    img_x_g = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False).fit_transform(
        img_x[1].reshape(-1, 1))
    img_x_b = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False).fit_transform(
        img_x[2].reshape(-1, 1))
    if MODE_CORRELATION:
        img_x_rg = img_x[0] * img_x[1]
        img_x_rb = img_x[0] * img_x[2]
        img_x_gb = img_x[1] * img_x[2]

    img_tmp = np.append(img_x_r, np.append(img_x_g, img_x_b, 1), 1)
    if MODE_CORRELATION:
        img_tmp = np.append(img_tmp, np.append(np.append(img_x_rg.reshape(-1, 1), img_x_rb.reshape(-1, 1), 1), img_x_gb.reshape(-1, 1), 1), 1)
    img_input_r = np.append(np.ones((img_tmp.shape[0], 1), np.float32), img_tmp, 1)
    #img_input_r = np.append(np.ones((img_tmp.shape[0], 1)), img_x_r, 1)    # black-and-white mode

    img_tmp = np.append(img_x_g, np.append(img_x_b, img_x_r, 1), 1)
    if MODE_CORRELATION:
        img_tmp = np.append(img_tmp, np.append(np.append(img_x_rg.reshape(-1, 1), img_x_rb.reshape(-1, 1), 1), img_x_gb.reshape(-1, 1), 1), 1)
    img_input_g = np.append(np.ones((img_tmp.shape[0], 1), np.float32), img_tmp, 1)
    #img_input_g = np.append(np.ones((img_tmp.shape[0], 1)), img_x_g, 1)    # black-and-white mode

    img_tmp = np.append(img_x_b, np.append(img_x_r, img_x_g, 1), 1)
    if MODE_CORRELATION:
        img_tmp = np.append(img_tmp, np.append(np.append(img_x_rg.reshape(-1, 1), img_x_rb.reshape(-1, 1), 1), img_x_gb.reshape(-1, 1), 1), 1)
    img_input_b = np.append(np.ones((img_tmp.shape[0], 1), np.float32), img_tmp, 1)
    #img_input_b = np.append(np.ones((img_tmp.shape[0], 1)), img_x_b, 1)    # black-and-white mode

    return img_input_r, img_input_g, img_input_b


def get_stylized_image_lookuptable(image_in, lookup_table):
    for i in range(image_in.shape[0]):
        for j in range(image_in.shape[1]):
            xr = image_in[i][j][0]
            xg = image_in[i][j][1]
            xb = image_in[i][j][2]

            xm = xr
            xs = int((xg + xb) / 2)
            image_in[i][j][0] = lookup_table[0][xm][xs]

            xm = xg
            xs = int((xb + xr) / 2)
            image_in[i][j][1] = lookup_table[1][xm][xs]

            xm = xb
            xs = int((xr + xg) / 2)
            image_in[i][j][2] = lookup_table[2][xm][xs]
    return


def get_lookup_table(stylization_model):
    # TODO: Need to create function
    table_r = np.zeros((256, 256), dtype=np.uint8)
    table_g = np.zeros((256, 256), dtype=np.uint8)
    table_b = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            xm = np.zeros(1, np.float)
            xm[0] = i / 255
            xm = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False).fit_transform(xm.reshape(-1, 1))

            xs = np.zeros(1, np.float)
            xs[0] = j / 255
            xs = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False).fit_transform(xs.reshape(-1, 1))

            tmp = np.append(xm, xs)
            x = np.append(np.ones(1), tmp)

            table_r[i][j] = np.clip(np.dot(stylization_model[0].coef_, x), 0, 1) * 255
            table_g[i][j] = np.clip(np.dot(stylization_model[1].coef_, x), 0, 1) * 255
            table_b[i][j] = np.clip(np.dot(stylization_model[2].coef_, x), 0, 1) * 255
            # ======================================================================

    print('done')
    return table_r, table_g, table_b


def deltaE76(img1, img2):
    from skimage import color
    img1 = color.rgb2lab(img1)
    img2 = color.rgb2lab(img2)
    img1 = np.reshape(img1, [-1, 3]).astype(np.float32)
    img2 = np.reshape(img2, [-1, 3]).astype(np.float32)
    delta_e = np.sqrt(np.sum(np.power(img1 - img2, 2), 1))
    return sum(delta_e) / (np.shape(delta_e)[0])


def deltaE00(img1, img2):
    """
    Calculate deltaE 2000 over two images and returns average deltaE 2000
    :param img1: numpy
    :param img2: numpy
    :return: float
    """
    from skimage import color
    import colour

    img1 = color.rgb2lab(img1)
    img2 = color.rgb2lab(img2)
    img1 = np.reshape(img1, [-1, 3]).astype(np.float32)
    img2 = np.reshape(img2, [-1, 3]).astype(np.float32)

    delta_e = colour.delta_E(img1, img2, method='CIE 2000')

    return np.mean(delta_e)


def get_error_metric(image_x, image_y):
    m, n, ch = image_x.shape
    if not np.issubdtype(image_x.dtype, np.uint8):
        max_val = 1
    else:
        max_val = 255

    mse = 0.0
    mse = np.subtract(image_x, image_y)**2 / n
    mse = np.sum(np.sum(mse, axis=0), axis=0)
    """
    for i in range(m):
        tmp = 0.0
        for j in range(n):
            tmp += (image_y[i][j] - image_x[i][j]) ** 2
        mse += (tmp / n)
    """
    mse /= m
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.mean())
    ret = {'psnr': psnr}

    #ret['deltaE'] = deltaE76(image_x, image_y)
    ret['deltaE'] = deltaE00(image_x, image_y)
    return ret


class ReconstructorCNN:
    def __init__(self, modelPath):
        pass

    @staticmethod
    def load_cnn_model(modelPath, uncertainty='mcdrop'):
        # model = eval(MODE_UNCERTAINTY[uncertainty])().cuda()
        if uncertainty.startswith('aleatoric'):
            aleatoric = True
        else:
            aleatoric = False
        model = TransformerNet(aleatoric=aleatoric).cuda()
        # model = UNetMod(nclass=133, aleatoric=aleatoric).cuda()
        print("=> loading checkpoint '{}', uncertainty {}".format(modelPath, uncertainty))
        checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint.state_dict(), strict=False)
        model.eval()
        return model

    @staticmethod
    def infer_original_image(model, stylized_in, img_size=256):
        stylized_in = input_transform((img_size, img_size))(stylized_in)

        with torch.no_grad():
            stylized_in = stylized_in.cuda().unsqueeze(0)
            prediction = model(stylized_in)
            prediction = prediction.to('cpu')

        prediction = prediction[0].permute(1, 2, 0).numpy()
        prediction = prediction.reshape(-1, 3) * np.array([0.229, 0.224, 0.225])
        prediction = prediction + np.array([0.485, 0.456, 0.406])
        prediction = prediction.reshape(img_size, img_size, 3)

        return prediction

    @staticmethod
    def infer_original_image_w_uncertainty(model, stylized_in, img_size=256, uncertainty='mcdrop', n_mc_iter=10):
        stylized_in = input_transform((img_size, img_size))(stylized_in)

        num_mc_iteration = n_mc_iter if uncertainty=='mcdrop' or uncertainty=='aleatoric_combined' else 0
        pred_map = np.zeros((num_mc_iteration, img_size, img_size, 3), np.float32)
        pred_var_map = None
        if uncertainty == 'aleatoric_combined':
            pred_var_map = np.zeros((num_mc_iteration, img_size, img_size, 3), np.float32)
        mean_map = None
        var_ratio = None

        with torch.no_grad():
            stylized_in = stylized_in.cuda().unsqueeze(0)
            prediction = model(stylized_in)
            if uncertainty.startswith('aleatoric'):
                prediction, var_ratio = prediction
                var_ratio = var_ratio.to('cpu')
            elif uncertainty == 'default2':
                prediction, _ = prediction
            prediction = prediction.to('cpu')

        # prediction = prediction[0].permute(1, 2, 0).numpy()
        prediction = np.float32(prediction[0].permute(1, 2, 0))
        prediction = prediction.reshape(-1, 3) * np.array([0.229, 0.224, 0.225], dtype=np.float32)
        prediction = prediction + np.array([0.485, 0.456, 0.406], dtype=np.float32)
        prediction = prediction.reshape(img_size, img_size, 3)

        if uncertainty.startswith('aleatoric'):
            var_ratio = np.exp(var_ratio[0].permute(1, 2, 0).numpy()).mean(2)

        for i in range(num_mc_iteration):
            with torch.no_grad():
                # stylized_in = stylized_in.cuda().unsqueeze(0)
                mc_pred = model(stylized_in, mcdrop=True)
                if uncertainty == 'aleatoric_combined':
                    mc_pred, mc_var = mc_pred
                    mc_var = mc_var.to('cpu')
                mc_pred = mc_pred.to('cpu')
            mc_pred = np.float32(mc_pred[0].permute(1, 2, 0))
            mc_pred = mc_pred.reshape(-1, 3) * np.array([0.229, 0.224, 0.225], dtype=np.float32)
            mc_pred = mc_pred + np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mc_pred = mc_pred.reshape(img_size, img_size, 3)
            pred_map[i] = mc_pred
            if uncertainty == 'aleatoric_combined':
                mc_var = np.exp(mc_var[0].permute(1, 2, 0).numpy())
                mc_var = mc_var.reshape((img_size, img_size, 3)) #.mean(2)
                pred_var_map[i] = mc_var

        if uncertainty == 'aleatoric_combined':
            mean_of_square = np.average(np.square(pred_map), axis=0)
            square_of_mean = np.square(np.average(pred_map, axis=0))
            var_ratio = np.average(pred_var_map, axis=0)
            var_ratio = (mean_of_square - square_of_mean + var_ratio).mean(2)

        elif uncertainty == 'mcdrop':
            mean_map = np.average(pred_map, axis=0)
            if True:
                var_ratio = np.var(pred_map, axis=0, dtype=np.float64).mean(2)    # mean over channel
            else:
                var_map_bin = pred_map.mean(3) // 0.03
                mode_var_bin = scipy.stats.mode(var_map_bin)
                var_ratio = np.zeros((img_size, img_size), np.float32)
                for i in range(img_size):
                    for j in range(img_size):
                        for k in range(num_mc_iteration):
                            if var_map_bin[k][i][j] == mode_var_bin[0][0][i][j]:
                                var_ratio[i][j] += 1 # / num_mc_iteration
                        var_ratio[i][j] = num_mc_iteration / var_ratio[i][j]

        return prediction, mean_map, var_ratio

