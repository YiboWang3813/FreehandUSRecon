import os 
import torch 
import torch.nn as nn 
import argparse 

from networks import generators, resnext, resnet, mynet, p3d, densenet 

parser = argparse.ArgumentParser() 
args = parser.parse_args() 

network_type = 'resnext50'
device = 'cuda:0' 


def define_model(model_type, pretrained_path='', neighbour_slice=args.neighbour_slice,
                 input_type=args.input_type, output_type=args.output_type):
    """ Define a network. """
    # set input channel 
    if input_type == 'diff_img':
        input_channel = neighbour_slice - 1
    else:
        input_channel = neighbour_slice

    # set network 
    if model_type == 'prevost':
        model_ft = generators.PrevostNet()
    elif model_type == 'resnext50':
        model_ft = resnext.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    elif model_type == 'resnext101':
        model_ft = resnext.resnet101(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # model_ft.conv1 = nn.Conv3d(neighbour_slice, 64, kernel_size=7, stride=(1, 2, 2),
        #                            padding=(3, 3, 3), bias=False)
    elif model_type == 'resnet152':
        model_ft = resnet.resnet152(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet101':
        model_ft = resnet.resnet101(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet50':
        model_ft = resnet.resnet50(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet34':
        model_ft = resnet.resnet34(pretrained=False)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet18':
        model_ft = resnet.resnet18(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'mynet':
        model_ft = mynet.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    elif model_type == 'mynet2':
        model_ft = generators.My3DNet()
    elif model_type == 'p3d':
        model_ft = p3d.P3D63()
        model_ft.conv1_custom = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                          padding=(0, 3, 3), bias=False)
    elif model_type == 'densenet121':
        model_ft = densenet.densenet121()
    else:
        print('network type of <{}> is not supported, use original instead'.format(network_type))
        model_ft = generators.PrevostNet()

    # set output layer in the network 
    num_ftrs = model_ft.fc.in_features

    if model_type == 'mynet':
        num_ftrs = 384
    elif model_type == 'prevost':
        num_ftrs = 576

    if output_type == 'average_dof' or output_type == 'sum_dof':
        # model_ft.fc = nn.Linear(128, 6)
        model_ft.fc = nn.Linear(num_ftrs, 6)
    else:
        # model_ft.fc = nn.Linear(128, (neighbour_slice - 1) * 6)
        model_ft.fc = nn.Linear(num_ftrs, (neighbour_slice - 1) * 6)

    # if args.training_mode == 'finetune':
    #     model_path = path.join(results_dir, args.model_filename)
    #     if path.isfile(model_path):
    #         print('Loading model from <{}>...'.format(model_path))
    #         model_ft.load_state_dict(torch.load(model_path))
    #         print('Done')
    #     else:
    #         print('<{}> not exists! Training from scratch...'.format(model_path))

    # load pretrained network if accessible 
    if pretrained_path:
        if os.path.isfile(pretrained_path):
            print('Loading model from <{}>...'.format(pretrained_path))
            model_ft.load_state_dict(torch.load(pretrained_path, map_location='cuda:0'))
            # model_ft.load_state_dict(torch.load(pretrained_path))
            print('Done')
        else:
            print('<{}> not exists! Training from scratch...'.format(pretrained_path))
    else:
        print('Train this model from scratch!')

    model_ft.cuda()
    model_ft = model_ft.to(device)
    print('define model device {}'.format(device))
    return model_ft