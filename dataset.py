import os 
import cv2 
import random 
import numpy as np 
from os import path 
import argparse 

import SimpleITK as sitk 

import torch 
from torch.utils.data import Dataset 

import tools 

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d']

net = 'Generator'
batch_size = 28
use_last_pretrained = False
current_epoch = 0

parser = argparse.ArgumentParser() 
args = parser.parse_args()
device_no = '0'
epochs = 30 

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

data_dir = path.join(zion_common, 'US_recon/US_vid_frames')
pos_dir = path.join(zion_common, 'US_recon/US_vid_pos')
uronav_dir = path.join(zion_common, 'uronav_data')

train_ids = np.loadtxt('infos/train_ids.txt')
val_ids = np.loadtxt('infos/val_ids.txt')
clean_ids = {'train': train_ids, 'val': val_ids}

network_type = 'resnext50'


def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    # print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = os.path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images


def data_transform(input_img, crop_size=224, resize=224, normalize=False, masked_full=False):
    """
    Crop and resize image as you wish. This function is shared through multiple scripts
    :param input_img: please input a grey-scale numpy array image
    :param crop_size: center crop size, make sure do not contain regions beyond fan-shape
    :param resize: resized size
    :param normalize: whether normalize the image
    :return: transformed image
    """
    if masked_full:
        input_img[fan_mask == 0] = 0
        masked_full_img = input_img[112:412, 59:609]
        return masked_full_img

    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]

    patch_img = cv2.resize(patch_img, (resize, resize))
    # cv2.imshow('patch', patch_img)
    # cv2.waitKey(0)
    if normalize:
        patch_img = patch_img.astype(np.float64)
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.mean(patch_img))

    return patch_img



class FreehandUS4D(Dataset):

    def __init__(self, root_dir, initialization, transform=None):
        """
        """
        samples = filename_list(root_dir)
        # print('samples\n{}'.format(samples))
        # time.sleep(30)
        self.samples = samples
        self.transform = transform
        self.initialization = initialization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        # case_folder = '/zion/guoh9/US_recon/US_vid_frames/train/Case0141'
        case_folder = self.samples[idx]
        case_id = int(case_folder[-4:])

        norm_path = path.normpath(case_folder)
        res = norm_path.split(os.sep)
        status = res[-2]

        """ Make sure we do not use weird BK scans """
        if case_id not in clean_ids[status]:
            case_id = int(np.random.choice(clean_ids[status], 1)[0])
            case_folder = path.join(data_dir, status, 'Case{:04}'.format(case_id))

        aurora_pos = np.loadtxt(path.join(pos_dir, 'Case{:04}.txt'.format(case_id)))
        calib_mat = np.loadtxt(path.join(uronav_dir, '{}/Case{:04}/Case{:04}_USCalib.txt'.format(status, case_id, case_id)))

        frame_num = len(os.listdir(case_folder))
        sample_size = args.neighbour_slice
        # print('Case{:04} have {} frames'.format(case_id, frame_num))
        # print('sample_size {}'.format(sample_size))

        valid_range = frame_num - sample_size
        start_id = np.random.randint(low=0, high=valid_range, size=1)[0]
        start_params = aurora_pos[start_id, :]
        # start_mat = tools.params_to_mat44(trans_params=start_params, cam_cali_mat=calib_mat)
        # print('selected start index {}'.format(rand_num))

        select_ids = tools.sample_ids(slice_num=frame_num, neighbour_num=sample_size,
                                      sample_option='normal',
                                      random_reverse_prob=0, self_prob=0)
        # print('{} slices, select_ids\n{}'.format(frame_num, select_ids))
        # time.sleep(30)

        sample_slices = []
        labels = []
        # for slice_index in range(start_id, start_id + sample_size):
        # for slice_index in select_ids:
        for i in range(select_ids.shape[0]):
            slice_index = select_ids[i]
            slice_path = path.join(case_folder, '{:04}.jpg'.format(slice_index))
            slice_img = cv2.imread(slice_path, 0)
            slice_img = data_transform(slice_img, masked_full=False)
            sample_slices.append(slice_img)
            # print('slice_img shape {}'.format(slice_img.shape))

            # if slice_index != select_ids[0]:
            if i != select_ids.shape[0] - 1:
                first_id = select_ids[i]
                second_id = select_ids[i + 1]
                dof = tools.get_6dof_label(trans_params1=aurora_pos[first_id, :],
                                           trans_params2=aurora_pos[second_id, :],
                                           cam_cali_mat=calib_mat)
                labels.append(dof)
        # format_labels = np.asarray(labels)
        # format_labels = np.around(format_labels, decimals=3)

        if args.input_type == 'diff_img':
            diff_imgs = []
            for sample_id in range(1, len(sample_slices)):
                diff_imgs.append(sample_slices[sample_id] - sample_slices[sample_id - 1])
            sample_slices = np.asarray(diff_imgs)
            # print('sample_slices shape {}'.format(sample_slices.shape))
            # time.sleep(30)
        else:
            sample_slices = np.asarray(sample_slices)

        if args.output_type == 'average_dof':
            labels = np.mean(np.asarray(labels), axis=0)
        elif args.output_type == 'sum_dof':
            end2end_dof = tools.get_6dof_label(trans_params1=aurora_pos[select_ids[0], :],
                                               trans_params2=aurora_pos[select_ids[-1], :],
                                               cam_cali_mat=calib_mat)
            labels = end2end_dof
        else:
            labels = np.asarray(labels).flatten()
        # print('sample_slices shape {}'.format(sample_slices.shape))
        # print('labels shape {}'.format(labels.shape))
        # print('int_label\n{}'.format(format_labels))
        # time.sleep(30)

        if network_type in networks3D:
            sample_slices = np.expand_dims(sample_slices, axis=0)
            # print('sample_slices shape {}'.format(sample_slices.shape))
            # time.sleep(30)

        if normalize_dof:
            labels = (labels - dof_stats[:, 0]) / dof_stats[:, 1]
            # print(labels.shape)
            # print(labels)
            # time.sleep(30)


        sample_slices = torch.from_numpy(sample_slices).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)
        # print('dataloader device {}'.format(device))
        # print('sample_slices shape {}'.format(sample_slices.shape))
        # print('labels shape {}'.format(labels.shape))

        # print('selected_ids\n{}'.format(select_ids))
        # print('labels\n{}'.format(labels))
        # time.sleep(30)
        return sample_slices, labels, case_id, start_params, calib_mat
    


# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized


def normalize_volume(input_volume):
    # print('input_volume shape {}'.format(input_volume.shape))
    mean = np.mean(input_volume)
    std = np.std(input_volume)

    normalized_volume = (input_volume - mean) / std
    # print('normalized shape {}'.format(normalized_volume.shape))
    # time.sleep(30)
    return normalized_volume


def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

# ----- #
def _get_random_value(r, center, hasSign):
    randNumber = random.random() * r + center


    if hasSign:
        sign = random.random() > 0.5
        if sign == False:
            randNumber *= -1

    return randNumber


# ----- #
def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat


# ----- #
def create_transform(aX, aY, aZ, tX, tY, tZ, mat_base=None):
    if mat_base is None:
        mat_base = np.identity(3)

    t_all = np.asarray((tX, tY, tZ))

    # Get the transform
    rotX = sitk.VersorTransform((1, 0, 0), aX / 180.0 * np.pi)
    matX = get_array_from_itk_matrix(rotX.GetMatrix())
    #
    rotY = sitk.VersorTransform((0, 1, 0), aY / 180.0 * np.pi)
    matY = get_array_from_itk_matrix(rotY.GetMatrix())
    #
    rotZ = sitk.VersorTransform((0, 0, 1), aZ / 180.0 * np.pi)
    matZ = get_array_from_itk_matrix(rotZ.GetMatrix())

    # Apply all the rotations
    mat_all = matX.dot(matY.dot(matZ.dot(mat_base[:3, :3])))

    return mat_all, t_all
