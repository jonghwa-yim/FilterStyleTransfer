
import torch

from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss, BCELoss
from torch.nn import _reduction as _Reduction
# from torch._jit_internal import weak_module, weak_script_method
from torch.autograd import Variable


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class MSEVarianceLoss(_Loss):

    def __init__(self):
        super(MSEVarianceLoss, self).__init__()

    def forward(self, input, variance, target):
        loss = torch.mean( ((input - target)**2 / (2*torch.exp(variance))) + (1/2 * variance) )
        return loss


# class SegmentationLosses(CrossEntropyLoss):
#     """2D Cross Entropy Loss with Auxilary Loss"""
#     def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
#                  aux=False, aux_weight=0.4, weight=None,
#                  size_average=True, ignore_index=-1, reduction='mean'):
#         super(SegmentationLosses, self).__init__(weight, ignore_index=ignore_index, reduction=reduction)
#         self.se_loss = se_loss
#         self.aux = aux
#         self.nclass = nclass
#         self.se_weight = se_weight
#         self.aux_weight = aux_weight
#         self.bceloss = BCELoss(weight, reduction=reduction)
#
#     def forward(self, *inputs):
#         if not self.se_loss and not self.aux:
#             return super(SegmentationLosses, self).forward(*inputs)
#         elif not self.se_loss:
#             (pred1, pred2), target = tuple(inputs)
#             loss1 = super(SegmentationLosses, self).forward(pred1, target)
#             loss2 = super(SegmentationLosses, self).forward(pred2, target)
#             return loss1 + self.aux_weight * loss2
#         elif not self.aux:
#             (pred, se_pred), target = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
#             loss1 = super(SegmentationLosses, self).forward(pred, target)
#             loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
#             return loss1 + self.se_weight * loss2
#         else:
#             (pred1, se_pred, pred2), target = tuple(inputs)
#             se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
#             loss1 = super(SegmentationLosses, self).forward(pred1, target)
#             loss2 = super(SegmentationLosses, self).forward(pred2, target)
#             loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
#             return loss1 + self.aux_weight * loss2 + self.se_weight * loss3
#
#     @staticmethod
#     def _get_batch_label_vector(target, nclass):
#         # target is a 3D Variable BxHxW, output is 2D BxnClass
#         batch = target.size(0)
#         tvect = Variable(torch.zeros(batch, nclass))
#         for i in range(batch):
#             hist = torch.histc(target[i].cpu().data.float(),
#                                bins=nclass, min=0,
#                                max=nclass-1)
#             vect = hist>0
#             tvect[i] = vect
#         return tvect

