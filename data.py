__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import torchvision.transforms as transforms

from dataset import DatasetFromTXT


def input_transform(input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])


def target_transform(input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize,
    ])


def target_seg_label_transform(input_size):
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=0),
    ])


def get_training_set(org_trainset_dir, style_trainset_dir, img_list_txt, input_size, get_seg=False):
    return DatasetFromTXT(org_trainset_dir, style_trainset_dir, img_list_txt,
                          input_transform=input_transform(input_size),
                          target_transform=target_transform(input_size),
                          segmentation=get_seg,
                          seg_transform=target_seg_label_transform(input_size))


def get_test_set(org_valset_dir, style_valset_dir, img_list_txt, input_size, get_seg=False):
    return DatasetFromTXT(org_valset_dir, style_valset_dir, img_list_txt,
                          input_transform=input_transform(input_size),
                          target_transform=target_transform(input_size),
                          segmentation=get_seg,
                          seg_transform=target_seg_label_transform(input_size))
