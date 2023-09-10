import torch.nn as nn
import torch.nn.functional as F
import torch
import SimpleITK as sitk
import numpy as np
import sys
from scipy.ndimage import morphology
sys.dont_write_bytecode = True  # don't generate the binray python file .pyc

hdcomputer = sitk.HausdorffDistanceImageFilter()

def weight_cal(recon,pos):
    a1 = nn.L1Loss(size_average=False, reduce=False, reduction='mean')(recon, pos).detach()
    a2 = torch.mean(a1,dim = [1,2,3,4]).view([recon.shape[0],1,1,1,1])

    return a2


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)



class DiceLoss(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth))



class DiceLossWeight(nn.Module):
    """
    define the dice loss
    """
    def __init__(self):
        super(DiceLossWeight, self).__init__()

    def forward(self, input, target,weight_adapt):
        smooth = 1.
        iflat = input.contiguous().view([input.shape[0],-1])
        tflat = target.contiguous().view([input.shape[0],-1])

        intersection = (iflat * tflat).sum(-1)

        A_sum = torch.sum(iflat * iflat,-1)
        B_sum = torch.sum(tflat * tflat,-1)
        loss_batch = 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth))
        loss = loss_batch*weight.squeeze()

        return loss.sum()


"""dice coefficient"""
def dice(pre, gt, tid=1):
    pre=pre==tid   #make it boolean
    gt=gt==tid     #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    dsc=(2. * intersection.sum() + 1e-07) / (pre.sum() + gt.sum() + 1e-07)

    return dsc

"""positive predictive value"""
def pospreval(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    ppv=(1.0*intersection.sum() + 1e-07) / (pre.sum()+1e-07)

    return ppv

"""sensitivity"""
def sensitivity(pre,gt,tid=1):
    pre=pre==tid #make it boolean
    gt=gt==tid   #make it boolean
    pre=np.asarray(pre).astype(np.bool)
    gt=np.asarray(gt).astype(np.bool)

    if pre.shape != gt.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have the same shape.")

    intersection = np.logical_and(pre, gt)
    sen=(1.0*intersection.sum()+1e-07) / (gt.sum()+1e-07)

    return sen

def seg_metric(pre,gt,itk_info):

    fake = (pre>0.5).astype(np.float32)
    real = (gt>0.5).astype(np.float32)
    DSC = dice(fake,real)
    PPV = pospreval(fake,real)
    SEN = sensitivity(fake,real)
    real_itk = sitk.GetImageFromArray(real)
    fake_itk = sitk.GetImageFromArray(fake)
    if np.sum(fake) !=0:

        real_itk.SetOrigin(itk_info.GetOrigin())
        real_itk.SetSpacing(itk_info.GetSpacing())
        real_itk.SetDirection(itk_info.GetDirection())

        fake_itk.SetOrigin(itk_info.GetOrigin())
        fake_itk.SetSpacing(itk_info.GetSpacing())
        fake_itk.SetDirection(itk_info.GetDirection())

        hdcomputer.Execute(real_itk>0.5, fake_itk>0.5)
        HD = hdcomputer.GetAverageHausdorffDistance()
    else:
        HD = 100

    return DSC,PPV,SEN,HD




