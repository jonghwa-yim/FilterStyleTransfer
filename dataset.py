__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

from os.path import join, exists

import numpy as np
import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFromTXT(data.Dataset):
    def __init__(self, org_dir, style_dir, txt_file,
                 input_transform=None, target_transform=None,
                 segmentation=False, seg_transform=None):
        super(DatasetFromTXT, self).__init__()

        fp = open(txt_file, 'r')
        image_names = fp.readlines()
        fp.close()

        COCO_HOME = '/f_data1/TrainingSets/coco/'
        if org_dir.find('train') > 0:
            panoptic_dir = join(COCO_HOME, 'annotations', 'panoptic_train2017_semantic_trainid_stff')
        else:
            panoptic_dir = join(COCO_HOME, 'annotations', 'panoptic_val2017_semantic_trainid_stff')

        self.image_filenames = [(join(org_dir, x.strip().split(' ')[0]),
                                 join(style_dir, x.strip().split(' ')[1]),
                                 join(panoptic_dir, x.strip().split(' ')[0].replace('jpg', 'png')))
                                if segmentation else
                                (join(org_dir, x.strip().split(' ')[0]),
                                 join(style_dir, x.strip().split(' ')[1])) for x in image_names]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.segmentation = segmentation
        self.seg_transform = seg_transform
        pass

    def __getitem__(self, index):
        input_img = default_loader(self.image_filenames[index][1])
        target_img = default_loader(self.image_filenames[index][0])
        if self.input_transform:
            input_img = self.input_transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)

        seg_gt = 0
        if self.segmentation and self.seg_transform != None:
            if exists(self.image_filenames[index][2]):
                seg_gt = Image.open(self.image_filenames[index][2])
                seg_gt = np.array(self.seg_transform(seg_gt)).astype(np.long)
            else:
                seg_gt = np.full((256, 256), 255, np.long)

        return input_img, target_img, seg_gt

    def __len__(self):
        return len(self.image_filenames)


# import os
# import numpy as np
# import cv2
# import json
# from pycocotools.cocoeval import COCOeval
# from collections import defaultdict
#
# from dataset_.json_dataset import JsonDataset, extend_with_flipped_entries, filter_for_training, \
#     add_bbox_regression_targets
# from dataset_.base_dataset import BaseDataset
# from PIL import Image, ImageDraw
#
# import pycocotools.mask as mask_util
#
#
# class coco(BaseDataset):
#     def __init__(self, image_sets, flip=False, proposal_files=None, phase='train', result_path=''):
#
#         super(coco, self).__init__()
#
#         image_dirs = {
#             'train2017': os.path.join('/f_data2/TrainingSets/coco/', 'images', 'train2017'),
#             'val2017': os.path.join('/f_data2/TrainingSets/coco/', 'images', 'val2017'),
#             'test-dev2017': os.path.join('/f_data2/TrainingSets/coco/', 'images', 'test2017'),
#         }
#
#         anno_files = {
#             'train2017': 'instances_train2017.json',
#             'val2017': 'instances_val2017.json',
#             'test-dev2017': 'image_info_test-dev2017.json',
#         }
#
#         if image_sets[0] == 'test-dev2017':
#             self.panoptic_json_file = os.path.join('/f_data2/TrainingSets/coco/', 'annotations',
#                                                    'image_info_test-dev2017.json')
#         else:
#             self.panoptic_json_file = os.path.join('/f_data2/TrainingSets/coco/', 'annotations',
#                                                    'panoptic_val2017_stff.json')
#             self.panoptic_gt_folder = os.path.join('/f_data2/TrainingSets/coco/', 'annotations',
#                                                    'panoptic_val2017')
#
#         if proposal_files is None:
#             proposal_files = [None] * len(image_sets)
#
#         if phase == 'train' and len(image_sets) > 1:
#             # combine multiple datasets
#             roidbs = []
#             for image_set, proposal_file in zip(image_sets, proposal_files):
#                 dataset = JsonDataset('coco_' + image_set,
#                                       image_dir=image_dirs[image_set],
#                                       anno_file=os.path.join('/f_data2/TrainingSets/coco/',
#                                                              'annotations', anno_files[image_set]))
#                 roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file,
#                                           crowd_filter_thresh=config.train.crowd_filter_thresh)
#                 roidbs.append(roidb)
#             roidb = roidbs[0]
#             for r in roidbs[1:]:
#                 roidb.extend(r)
#             roidb = filter_for_training(roidb)
#             add_bbox_regression_targets(roidb)
#
#         else:
#             assert len(image_sets) == 1
#             self.dataset = JsonDataset('coco_' + image_sets[0],
#                                        image_dir=image_dirs[image_sets[0]],
#                                        anno_file=os.path.join(config.dataset.dataset_path,
#                                                               'annotations',
#                                                               anno_files[image_sets[0]]))
#             roidb = self.dataset.get_roidb(gt=True, proposal_file=proposal_files[0],
#                                            crowd_filter_thresh=config.train.crowd_filter_thresh if phase != 'test' else 0)
#             if phase != 'test':
#                 roidb = filter_for_training(roidb)
#                 add_bbox_regression_targets(roidb)
#
#         self.roidb = roidb
#         self.phase = phase
#         self.flip = flip
#         self.result_path = result_path
#         self.num_classes = 81
#
#     def __len__(self):
#         return len(self.roidb)
#
#     def __getitem__(self, index):
#         blob = defaultdict(list)
#         im_blob, im_scales = self.get_image_blob([self.roidb[index]])
#         label = dict()
#
#         if True: # config.network.has_fcn_head:
#             if self.phase != 'test':
#                 seg_gt = np.array(Image.open(
#                     self.roidb[index]['image'].replace('images', 'annotations').replace('train2017',
#                                                                                         'panoptic_train2017_semantic_trainid_stff').replace(
#                         'val2017', 'panoptic_val2017_semantic_trainid_stff').replace('jpg', 'png')))
#                 if self.roidb[index]['flipped']:
#                     seg_gt = np.fliplr(seg_gt)
#                 seg_gt = cv2.resize(seg_gt, None, None, fx=im_scales[0], fy=im_scales[0],
#                                     interpolation=cv2.INTER_NEAREST)
#                 label.update({'seg_gt': seg_gt})
#                 label.update(
#                     {'gt_classes': label['roidb']['gt_classes'][label['roidb']['is_crowd'] == 0]})
#                 label.update({'mask_gt': np.zeros(
#                     (len(label['gt_classes']), im_blob.shape[-2], im_blob.shape[-1]))})
#                 idx = 0
#                 for i in range(len(label['roidb']['gt_classes'])):
#                     if label['roidb']['is_crowd'][i] != 0:
#                         continue
#                     if type(label['roidb']['segms'][i]) is list and type(
#                             label['roidb']['segms'][i][0]) is list:
#                         img = Image.new('L', (int(np.round(im_blob.shape[-1] / im_scales[0])),
#                                               int(np.round(im_blob.shape[-2] / im_scales[0]))), 0)
#                         for j in range(len(label['roidb']['segms'][i])):
#                             ImageDraw.Draw(img).polygon(tuple(label['roidb']['segms'][i][j]),
#                                                         outline=1, fill=1)
#                         label['mask_gt'][idx] = cv2.resize(np.array(img), None, None, fx=im_scales[0],
#                                                            fy=im_scales[0],
#                                                            interpolation=cv2.INTER_NEAREST)
#                     else:
#                         assert type(label['roidb']['segms'][i]) is dict or type(
#                             label['roidb']['segms'][i][0]) is dict
#                         if type(label['roidb']['segms'][i]) is dict:
#                             label['mask_gt'][idx] = cv2.resize(mask_util.decode(
#                                 mask_util.frPyObjects([label['roidb']['segms'][i]],
#                                                       label['roidb']['segms'][i]['size'][0],
#                                                       label['roidb']['segms'][i]['size'][1]))[:, :,
#                                                                0], None, None, fx=im_scales[0],
#                                                                fy=im_scales[0],
#                                                                interpolation=cv2.INTER_NEAREST)
#                         else:
#                             assert len(label['roidb']['segms'][i]) == 1
#                             output = mask_util.decode(label['roidb']['segms'][i])
#                             label['mask_gt'][idx] = cv2.resize(output[:, :, 0], None, None,
#                                                                fx=im_scales[0], fy=im_scales[0],
#                                                                interpolation=cv2.INTER_NEAREST)
#                     idx += 1
#             else:
#                 pass
#
#         return data, label, index
#
#     def evaluate_masks(
#             self,
#             all_boxes,
#             all_segms,
#             output_dir,
#     ):
#         res_file = os.path.join(
#             output_dir, 'segmentations_' + self.dataset.name + '_results.json'
#         )
#         results = []
#         for cls_ind, cls in enumerate(self.dataset.classes):
#             if cls == '__background__':
#                 continue
#             if cls_ind >= len(all_boxes):
#                 break
#             cat_id = self.dataset.category_to_id_map[cls]
#             results.extend(
#                 self.segms_results_one_category(all_boxes[cls_ind], all_segms[cls_ind], cat_id))
#         with open(res_file, 'w') as fid:
#             json.dump(results, fid)
#         coco_dt = self.dataset.COCO.loadRes(str(res_file))
#         coco_eval = COCOeval(self.dataset.COCO, coco_dt, 'segm')
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         self.log_detection_eval_metrics(coco_eval, os.path.join(output_dir, 'detection_results.txt'))
#         return coco_eval
#
#     def evaluate_ssegs(self, pred_segmentations, res_file_folder):
#         self.write_segmentation_result(pred_segmentations, res_file_folder)
#
#         num_seg_classes = 133
#
#         confusion_matrix = np.zeros((num_seg_classes, num_seg_classes))
#         for i, roidb in enumerate(self.roidb):
#             seg_gt = np.array(Image.open(
#                 self.roidb[i]['image'].replace('images', 'annotations').replace('train2017',
#                                                                                 'panoptic_train2017_semantic_trainid_stff').replace(
#                     'val2017', 'panoptic_val2017_semantic_trainid_stff').replace('jpg',
#                                                                                  'png'))).astype(
#                 np.float32)
#
#             seg_pathes = os.path.split(roidb['image'])
#             res_image_name = seg_pathes[-1]
#             res_save_path = os.path.join(res_file_folder, res_image_name + '.png')
#
#             seg_pred = Image.open(res_save_path)
#
#             seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
#             ignore_index = seg_gt != 255
#             seg_gt = seg_gt[ignore_index]
#             seg_pred = seg_pred[ignore_index]
#
#             confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, num_seg_classes)
#
#         pos = confusion_matrix.sum(1)
#         res = confusion_matrix.sum(0)
#         tp = np.diag(confusion_matrix)
#
#         IU_array = (tp / np.maximum(1.0, pos + res - tp))
#         mean_IU = IU_array.mean()
#
#         evaluation_results = {'meanIU': mean_IU, 'IU_array': IU_array,
#                               'confusion_matrix': confusion_matrix}
#
#         def convert_confusion_matrix(confusion_matrix):
#             cls_sum = confusion_matrix.sum(axis=1)
#             confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
#             return confusion_matrix
#
#         # logger.info('evaluate segmentation:')
#         meanIU = evaluation_results['meanIU']
#         IU_array = evaluation_results['IU_array']
#         confusion_matrix = convert_confusion_matrix(evaluation_results['confusion_matrix'])
#         # logger.info('IU_array:')
#         # for i in range(len(IU_array)):
#         #     logger.info('%.5f' % IU_array[i])
#         # logger.info('meanIU:%.5f' % meanIU)
#         np.set_printoptions(precision=3, suppress=True, linewidth=200)
#         import re
#         confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))
#         # logger.info('confusion_matrix:')
#         # logger.info(confusion_matrix)
#
#     def write_segmentation_result(self, segmentation_results, res_file_folder):
#         """
#         Write the segmentation result to result_file_folder
#         :param segmentation_results: the prediction result
#         :param result_file_folder: the saving folder
#         :return: [None]
#         """
#         if not os.path.exists(res_file_folder):
#             os.mkdir(res_file_folder)
#
#         for i, roidb in enumerate(self.roidb):
#             seg_pathes = os.path.split(roidb['image'])
#             res_image_name = seg_pathes[-1]
#             res_save_path = os.path.join(res_file_folder, res_image_name + '.png')
#
#             segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
#             segmentation_result = Image.fromarray(segmentation_result)
#             segmentation_result.save(res_save_path)


if __name__ == '__main__':
    # Sample code
    DatasetFromTXT('/f_data1/TrainingSets/coco/images/val2017/',
                   '/f_data1/TrainingSets/coco_stylized/images/val2017/',
                   '/f_data1/TrainingSets/coco_stylized/images/val2017.txt', segmentation=True)
    DatasetFromTXT('my_images/sample_dataset/train/org', 'my_images/sample_dataset/train/style',
                   'my_images/sample_dataset/train/image_list.md')
