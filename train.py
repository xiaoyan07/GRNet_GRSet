import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import argparse
import cv2
import os
import numpy as np
import random
from time import time
from networks.linknet import LinkNet50
from framework_mark import MyFrame
from loss_mark import dice_bce_loss
from data_mark import ImageFolder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2333)

SAVE_PATH = '/'
def get_arguments():

    parser = argparse.ArgumentParser(description="LinkNet50 trained on GRSet")
    parser.add_argument("--model", type=str, default='LinkNet50',
                        help="")
    parser.add_argument("--name", type=str, default='LinkNet50_GRSet',
                        help="available options")
    parser.add_argument("--data-dir", type=str, default='',
                        help="Path to the GRSet.")
    parser.add_argument("--weight-dir", type=str, default='weight/',
                        help="Path to save weight.")

    return parser.parse_args()

args = get_arguments()

SHAPE = (768, 768)
batchsize = 4

imagelist = filter(lambda x: x.find('jpg')!=-1, os.listdir(args.data_dir))
trainlist = list(map(lambda x: x[:-4], imagelist))


if args.model == 'LinkNet50':
    solver = MyFrame(LinkNet50, dice_bce_loss, 2e-4)

dataset = ImageFolder(trainlist, args.data_dir)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=1)

tic = time()
MODEL = args.name
weigth_path = args.weight_dir + '/' + MODEL + '/'

def main():

    no_optim = 0
    total_epoch = 12
    train_epoch_best_loss = 100.

    if not os.path.exists(weigth_path):
        os.makedirs(weigth_path)

    mylog = open(weigth_path + MODEL +'.log','w')

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        train_bce_loss = 0
        train_dice_loss = 0
        for img, segments, segments_mark in data_loader_iter:
            solver.set_input(img, segments, segments_mark)
            loss_bce, loss_dice, train_loss = solver.optimize()
            train_bce_loss += loss_bce
            train_dice_loss += loss_dice
            train_epoch_loss += train_loss

        train_bce_loss /= len(data_loader_iter)
        train_dice_loss /= len(data_loader_iter)
        train_epoch_loss /= len(data_loader_iter)

        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('bce_loss:', train_bce_loss, file=mylog)
        print('dice_loss:', train_dice_loss, file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)

        print('SHAPE:', SHAPE, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('bce_loss:', train_bce_loss)
        print('dice_loss:', train_dice_loss)
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save(weigth_path + MODEL + '_latest.th')
        if epoch == 7:
            solver.load(weigth_path + MODEL + '_latest.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
            solver.save(weigth_path + MODEL + '_7.th')

        if epoch == 9:
            solver.load(weigth_path + MODEL + '_latest.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
            solver.save(weigth_path + MODEL + '_9.th')

        if epoch == 11:
            solver.load(weigth_path + MODEL + '_latest.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
            solver.save(weigth_path + MODEL + '_11.th')

        mylog.flush()

    solver.save(weigth_path + MODEL + '_12.th')

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    main()