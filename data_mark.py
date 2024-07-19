"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os

def random_crop(image, crop_shape, mask=None):
    image_shape = image.shape
    image_shape = image_shape[0:2]
    ret = []
    nh = np.random.randint(0, image_shape[0] - crop_shape[0])
    nw = np.random.randint(0, image_shape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    ret.append(image_crop)
    if mask is not None:
        mask_crop = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        ret.append(mask_crop)
        return ret
    return ret[0]

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
    return image, mask

def scribble(mask, p = 17):
    kernels = np.ones((p, p))
    dilate_mask = cv2.dilate(mask, kernels)

    blank_mask = np.random.uniform(0, 1, [768, 768])
    T = np.sum(mask)/(768*768*255.0)
    T_num = 1 - 2*T

    np.putmask(blank_mask, blank_mask >= T_num, 255.0)
    blank_mask[:, :][dilate_mask > 128] = 0
    mask_marking = blank_mask + mask

    return mask_marking

def default_loader(id, root):
    img = cv2.imread(os.path.join(root, '{}.jpg').format(id))
    mask = cv2.imread(os.path.join(root+'{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    img, mask = random_crop(img, (768, 768), mask)

    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask_mark = scribble(mask)

    mask = np.expand_dims(mask, axis=2)
    mask_mark = np.expand_dims(mask_mark, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
    mask_mark = np.array(mask_mark, np.float32).transpose(2, 0, 1)/255.0

    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    mask_mark[mask_mark >= 0.5] = 1
    mask_mark[mask_mark < 0.5] = 0

    return img, mask, mask_mark

class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask, mask_mark = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        mask_mark = torch.Tensor(mask_mark)

        return img, mask, mask_mark

    def __len__(self):
        return len(list(self.ids))
