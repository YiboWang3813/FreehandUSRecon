import os
import cv2
import argparse

from os import path
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import FreehandUS4D  
from engine import train_model 
from networks import define_model 

################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--init_mode',
                    type=str,
                    help="mode of training with different transformation matrics",
                    default='random_SRE2')

parser.add_argument('-t', '--training_mode',
                    type=str,
                    help="mode of training with different starting points",
                    default='scratch')

parser.add_argument('-m', '--model_filename',
                    type=str,
                    help="name of the pre-trained mode file",
                    default='None')

parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='Learning rate',
                    default=5e-6)

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=500)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    default='resnext50')

parser.add_argument('-info', '--information',
                    type=str,
                    help='infomation of this round of experiment',
                    default='Here is the information')

parser.add_argument('-ns', '--neighbour_slice',
                    type=int,
                    help='number of slice that acts as one sample',
                    default='8')

parser.add_argument('-it', '--input_type',
                    type=str,
                    help='input type of the network,'
                         'org_img, diff_img, optical flow',
                    default='org_img')

parser.add_argument('-ot', '--output_type',
                    type=str,
                    help='output type of the network,'
                         'average_dof, separate_dof, sum_dof',
                    default='average_dof')

pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d']

net = 'Generator'
batch_size = 28
use_last_pretrained = False
current_epoch = 0

args = parser.parse_args()
device_no = args.device_no
epochs = args.epochs

training_progress = np.zeros((epochs, 4))

hostname = os.uname().nodename
zion_common = '/zion/guoh9'
on_arc = False
if 'arc' == hostname:
    on_arc = True
    print('on_arc {}'.format(on_arc))
    # device = torch.device("cuda:{}".format(device_no))
    zion_common = '/raid/shared/guoh9'
    batch_size = 64
# device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:{}".format(device_no))
# print('start device {}'.format(device))

fan_mask = cv2.imread('data/avg_img.png', 0)

normalize_dof = True
dof_stats = np.loadtxt('infos/dof_stats.txt')


def save_info():
    file = open('data/experiment_diary/{}.txt'.format(now_str), 'a+')
    file.write('Time_str: {}\n'.format(now_str))
    # file.write('Initial_mode: {}\n'.format(args.init_mode))
    file.write('Training_mode: {}\n'.format(args.training_mode))
    file.write('Model_filename: {}\n'.format(args.model_filename))
    file.write('Device_no: {}\n'.format(args.device_no))
    file.write('Epochs: {}\n'.format(args.epochs))
    file.write('Network_type: {}\n'.format(args.network_type))
    file.write('Learning_rate: {}\n'.format(args.learning_rate))
    file.write('Neighbour_slices: {}\n'.format(args.neighbour_slice))
    file.write('Infomation: {}\n'.format(args.information))
    file.write('Best_epoch: 0\n')
    file.write('Val_loss: {:.4f}\n'.format(1000))
    file.close()
    print('Information has been saved!')


if __name__ == '__main__':

    # set arguments 
    # data_dir = path.join('/home/guoh9/tmp/US_vid_frames')
    # results_dir = path.join('/home/guoh9/tmp/US_vid_frames')

    data_dir = path.join(zion_common, 'US_recon/US_vid_frames')
    pos_dir = path.join(zion_common, 'US_recon/US_vid_pos')
    uronav_dir = path.join(zion_common, 'uronav_data')

    train_ids = np.loadtxt('infos/train_ids.txt')
    val_ids = np.loadtxt('infos/val_ids.txt')
    clean_ids = {'train': train_ids, 'val': val_ids}

    if 'arc' == hostname:
        results_dir = '/home/guoh9/US_recon/results'
    else:
        results_dir = path.join(zion_common, 'US_recon/results')

    init_mode = args.init_mode
    network_type = args.network_type
    print('Transform initialization mode: {}'.format(init_mode))
    print('Training mode: {}'.format(args.training_mode))

    # build dataset and dataloader 
    image_datasets = {x: FreehandUS4D(os.path.join(data_dir, x), init_mode)
                      for x in ['train', 'val']}
    print('image_dataset\n{}'.format(image_datasets))
    # time.sleep(30)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print('Number of training samples: {}'.format(dataset_sizes['train']))
    print('Number of validation samples: {}'.format(dataset_sizes['val']))

    # build model, loss criterion, optimizer and lr scheduler 
    model_folder = '/zion/guoh9/US_recon/results'
    model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft = define_model(model_type=network_type, pretrained_path=model_path)
    model_ft = define_model(model_type=network_type)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.()
    # criterion = nn.L1Loss()

    # mahalanobis_dist = mahalanobis.MahalanobisMetricLoss()

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model_ft.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model_ft.parameters(), lr=1)
    # optimizer = optim.SGD(model_ft.parameters(), lr=lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    now = datetime.now()
    now_str = now.strftime('%m%d-%H%M%S')

    save_info() # save settings info  

    # Train and evaluate
    fn_best_model = path.join(results_dir, '3d_best_{}_{}.pth'.format(net, now_str))
    print('Start training...')
    print('This model is <3d_best_{}_{}_{}.pth>'.format(net, now_str, init_mode))
    txt_path = path.join(results_dir, 'training_progress_{}_{}.txt'.format(net, now_str))

    hist_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, fn_best_model, num_epochs=epochs)

    # fn_hist = os.path.join(results_dir, 'hist_{}_{}_{}.npy'.format(net, now_str, init_mode))
    # np.save(fn_hist, hist_ft)

    np.savetxt(txt_path, training_progress)

    now = datetime.now()
    now_stamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print('#' * 15 + ' Training {} completed at {} '.format(init_mode, now_stamp) + '#' * 15)
