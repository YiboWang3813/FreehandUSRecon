import os 
import cv2 
import torch 
import argparse 
import numpy as np 

import tools 

parser = argparse.ArgumentParser() 

pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d']

net = 'Generator'
batch_size = 28
use_last_pretrained = False
current_epoch = 0

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


def get_dist_loss(labels, outputs, start_params, calib_mat):
    # print('labels shape {}'.format(labels.shape))
    # print('outputs shape {}'.format(outputs.shape))
    # print('start_params shape {}'.format(start_params.shape))
    # print('calib_mat shape {}'.format(calib_mat.shape))

    # print('labels_before\n{}'.format(labels.shape))
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    if normalize_dof:
        labels = labels / dof_stats[:, 1] + dof_stats[:, 0]
        outputs = outputs / dof_stats[:, 1] + dof_stats[:, 0]


    start_params = start_params.data.cpu().numpy()
    calib_mat = calib_mat.data.cpu().numpy()

    if args.output_type == 'sum_dof':
        batch_errors = []
        for sample_id in range(labels.shape[0]):
            gen_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                           dof=outputs[sample_id, :],
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                          dof=labels[sample_id, :],
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param = np.expand_dims(gen_param, axis=0)
            gt_param = np.expand_dims(gt_param, axis=0)

            result_pts = tools.params2corner_pts(params=gen_param, cam_cali_mat=calib_mat[sample_id, :, :])
            gt_pts = tools.params2corner_pts(params=gt_param, cam_cali_mat=calib_mat[sample_id, :, :])

            sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
            batch_errors.append(sample_error)

        batch_errors = np.asarray(batch_errors)

        avg_batch_error = np.asarray(np.mean(batch_errors))
        error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
        error_tensor = error_tensor.type(torch.FloatTensor)
        error_tensor = error_tensor.to(device)
        error_tensor = error_tensor * 0.99
        # print('disloss device {}'.format(device))
        # print(error_tensor)
        # time.sleep(30)
        return error_tensor

    if args.output_type == 'average_dof':
        labels = np.expand_dims(labels, axis=1)
        labels = np.repeat(labels, args.neighbour_slice - 1, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        outputs = np.repeat(outputs, args.neighbour_slice - 1, axis=1)
    else:
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1] // 6, 6))
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1] // 6, 6))
    # print('labels_after\n{}'.format(labels.shape))
    # print('outputs\n{}'.format(outputs.shape))
    # time.sleep(30)

    batch_errors = []
    final_drifts = []
    for sample_id in range(labels.shape[0]):
        gen_param_results = []
        gt_param_results = []
        for neighbour in range(labels.shape[1]):
            if neighbour == 0:
                base_param_gen = start_params[sample_id, :]
                base_param_gt = start_params[sample_id, :]
            else:
                base_param_gen = gen_param_results[neighbour - 1]
                base_param_gt = gt_param_results[neighbour - 1]
            gen_dof = outputs[sample_id, neighbour, :]
            gt_dof = labels[sample_id, neighbour, :]
            gen_param = tools.get_next_pos(trans_params1=base_param_gen, dof=gen_dof,
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=base_param_gt, dof=gt_dof,
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param_results.append(gen_param)
            gt_param_results.append(gt_param)
        gen_param_results = np.asarray(gen_param_results)
        gt_param_results = np.asarray(gt_param_results)
        # print('gen_param_results shape {}'.format(gen_param_results.shape))

        result_pts = tools.params2corner_pts(params=gen_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        gt_pts = tools.params2corner_pts(params=gt_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        # print(result_pts.shape, gt_pts.shape)
        # time.sleep(30)

        results_final_vec = np.mean(result_pts[-1, :, :], axis=0)
        gt_final_vec = np.mean(gt_pts[-1, :, :], axis=0)
        final_drift = np.linalg.norm(results_final_vec - gt_final_vec) * 0.2
        final_drifts.append(final_drift)
        # print(results_final_vec, gt_final_vec)
        # print(final_drift)
        # time.sleep(30)

        sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
        batch_errors.append(sample_error)

    batch_errors = np.asarray(batch_errors)
    avg_batch_error = np.asarray(np.mean(batch_errors))

    error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
    error_tensor = error_tensor.type(torch.FloatTensor)
    error_tensor = error_tensor.to(device)
    error_tensor = error_tensor * 0.99
    # print('disloss device {}'.format(device))
    # print(error_tensor)
    # time.sleep(30)

    avg_final_drift = np.asarray(np.mean(np.asarray(final_drifts)))
    final_drift_tensor = torch.tensor(avg_final_drift, requires_grad=True)
    final_drift_tensor = final_drift_tensor.type(torch.FloatTensor)
    final_drift_tensor = final_drift_tensor.to(device)
    final_drift_tensor = final_drift_tensor * 0.99
    return error_tensor, final_drift_tensor


def get_correlation_loss(labels, outputs):
    """ 计算相关系数损失 
    labels 金标准theta [N, 6] 
    outputs 预测的theta [N, 6] """
    x = outputs.flatten()
    y = labels.flatten()

    # compute the covariance between x and y 
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y # 乘积的均值-均值的乘积 

    # compute the standard deviation of x and y 
    var_x = torch.sum((x - mean_x) ** 2) / x.shape[0]
    var_y = torch.sum((y - mean_y) ** 2) / y.shape[0]
    std_x = torch.sqrt(var_x) 
    std_y = torch.sqrt(var_y) 

    corr_xy = cov_xy / (std_x * std_y) # correlation coefficient 

    loss = 1 - corr_xy
    
    return loss
