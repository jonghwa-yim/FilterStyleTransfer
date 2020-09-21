""" Train CNN to get repaired original image from stylized image.
"""
__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from architecture.transformer_net import TransformerNet
from data import get_training_set, get_test_set
from loss_custom import MSEVarianceLoss


def argument_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Style to Original AutoEncoder')
    parser.add_argument('--stylizedTrainDir', default='', type=str, metavar='PATH',
                        help='path to stylized image set for training.')
    parser.add_argument('--orgTrainDir', default='', type=str, metavar='PATH',
                        help='path to original image set for training')
    parser.add_argument('--stylizedTestDir', default='', type=str, metavar='PATH',
                        help='path to stylized image set for testing.')
    parser.add_argument('--orgTestDir', default='', type=str, metavar='PATH',
                        help='path to original image set for testing')
    parser.add_argument('--trainSetTxt', default='', type=str, metavar='PATH',
                        help='list of train images.')
    parser.add_argument('--valSetTxt', default='', type=str, metavar='PATH',
                        help='list of validation images.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-freq', '-p', default=2000, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--input_size', default=256, type=int, help='input image size. ')

    parser.add_argument('--uncertainty', type=str,
                        choices=['default', 'aleatoric'],
                        help='Type of uncertainty as variance. [default, mcdrop, aleatoric].')
    parser.add_argument('--segmentation', dest='segmentation', action='store_true',
                        help='add segmentation branch')
    parser.add_argument('--outputDir', default='', type=str, metavar='PATH',
                        help='path to save trained model.')

    args = parser.parse_args()
    return args


def train(epoch):
    epoch_loss = 0
    inter_loss = 0
    epoch_loss_seg = 0
    inter_loss_seg = 0

    for iteration, batch in enumerate(training_data_loader):
        input = batch[0].cuda(args.gpu, non_blocking=False)
        target = batch[1].cuda(args.gpu, non_blocking=True)
        target_seg = batch[2].cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        output = model(input)

        if args.uncertainty.startswith('aleatoric') and args.segmentation:
            restored, seg_output, var = output
            loss = criterion_var(restored, var, target)
            loss_seg = criterion_seg(seg_output, target_seg)
            epoch_loss_seg += loss_seg.item()
            inter_loss_seg += loss_seg.item()
        elif args.uncertainty.startswith('aleatoric'):
            restored, var = output
            loss = criterion_var(restored, var, target)
            loss_seg = 0.0
        elif args.segmentation:
            restored, seg_output = output
            loss = criterion_mse(restored, target)
            loss_seg = criterion_seg(seg_output, target_seg) / 8
            epoch_loss_seg += loss_seg.item()
            inter_loss_seg += loss_seg.item()
        else:
            restored = output
            loss = criterion_mse(restored, target)
            loss_seg = 0.0

        epoch_loss += loss.item()
        inter_loss += loss.item()

        loss += loss_seg

        loss.backward()
        optimizer.step()

        if iteration % args.print_freq == 0:
            print("===> Epoch[{}]({}/{}): ".format(epoch, iteration, len(training_data_loader)),
                  "Inter. Avg. Loss: {:.4f}, Seg Loss: {:.4f}".format(inter_loss / args.print_freq,
                                                                      inter_loss_seg / args.print_freq))
            inter_loss = 0
            inter_loss_seg = 0

    print("===> Epoch {} Complete: ".format(epoch),
          "Avg. Loss: {:.4f}, Seg Loss: {:.4f}".format(epoch_loss / len(training_data_loader),
                                                       epoch_loss_seg / len(training_data_loader)))
    return


def test():
    avg_psnr = 0
    avg_loss = 0
    avg_loss_seg = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input = batch[0].cuda(args.gpu, non_blocking=False)
            target = batch[1].cuda(args.gpu, non_blocking=True)
            target_seg = batch[2].cuda(args.gpu, non_blocking=True)

            output = model(input)
            seg_output = None

            if args.uncertainty.startswith('aleatoric') and args.segmentation:
                restored, seg_output, var = output
            elif args.uncertainty.startswith('aleatoric'):
                restored, var = output
            elif args.segmentation:
                restored, seg_output = output
            else:
                restored = output
            mse = criterion_mse(restored, target)
            avg_loss += mse.item()
            if args.segmentation:
                loss_seg = criterion_seg(seg_output, target_seg)
                avg_loss_seg += loss_seg.item()
            psnr = 20 * log10(1 / mse.item())
            avg_psnr += psnr

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. MSE: {:.4f} dB".format(avg_loss / len(testing_data_loader)))
    print("===> Avg. Seg Loss: {:.4f} dB".format(avg_loss_seg / len(testing_data_loader)))
    return


def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, os.path.join(args.outputDir, model_out_path))
    print("Checkpoint saved to {}".format(model_out_path))
    return


def adjust_learning_rate(optimizer, epoch, epoch_ch=4, lr_update=0.2):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (lr_update ** (epoch // epoch_ch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_manual(optimizer, lr, lr_update=0.2):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = lr * lr_update
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new


if __name__ == '__main__':
    args = argument_parser()

    device = torch.device("cuda")

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    print('===> Loading datasets')
    train_set = get_training_set(args.orgTrainDir, args.stylizedTrainDir, args.trainSetTxt,
                                 (args.input_size, args.input_size), get_seg=args.segmentation)
    test_set = get_test_set(args.orgTestDir, args.stylizedTestDir, args.valSetTxt,
                            (args.input_size, args.input_size), get_seg=args.segmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                                      batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=args.workers,
                                     batch_size=args.testBatchSize, shuffle=False)

    print('===> Building model')
    if args.uncertainty.startswith('aleatoric'):
        aleatoric = True
    else:
        aleatoric = False
    aux = False
    model = TransformerNet(aleatoric=aleatoric, nclass=133).cuda(args.gpu)

    criterion_var = MSEVarianceLoss().cuda(args.gpu)
    criterion_mse = nn.MSELoss().cuda(args.gpu)
    criterion_seg = nn.CrossEntropyLoss(ignore_index=255).cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint.state_dict(), strict=True)

    print('===> Training model')
    epoch_ch = 4
    lr = args.lr
    for epoch in range(1, args.nEpochs + 1):
        train(epoch)
        test()
        if epoch == args.nEpochs or epoch % epoch_ch == 0 or epoch % int(epoch_ch/2):
            save_checkpoint(epoch)

        if True:
            adjust_learning_rate(optimizer, epoch, epoch_ch, 0.2)
        else:
            lrupdate = input()
            if lrupdate == 'y':
                lr = adjust_learning_rate_manual(optimizer, lr)
                print('=== Learning rate update. New lr {}'.format(lr))
