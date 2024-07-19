import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
import torch.nn.functional as F

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-8  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)

        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred, y_true_mark):
        pred = torch.sigmoid(y_pred)
        y_pred = pred * y_true_mark
        a =  self.bce_loss(y_pred, y_true * y_true_mark)
        b =  self.soft_dice_loss(y_true * y_true_mark, y_pred)
        return a, b, a + b
