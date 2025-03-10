import os 
import time 
import torch 
from datetime import datetime 
import numpy as np 

from losses import get_correlation_loss, get_dist_loss 
from dataset import FreehandUS4D 

zion_common = '/zion/guoh9'
data_dir = os.path.join(zion_common, 'US_recon/US_vid_frames')
init_mode = 'random_SRE2'
batch_size = 8 
device = 'cuda:0' 

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

now = datetime.now()
now_str = now.strftime('%m%d-%H%M%S')   

epochs = 30 
training_progress = np.zeros((epochs, 4))
results_dir = '/home/guoh9/US_recon/results'

net = 'Generator'
txt_path = os.path.join(results_dir, 'training_progress_{}_{}.txt'.format(net, now_str))

current_epoch = 0


def update_info(best_epoch, current_epoch, lowest_val_TRE):
    readFile = open('data/experiment_diary/{}.txt'.format(now_str))
    lines = readFile.readlines()
    readFile.close()

    file = open('data/experiment_diary/{}.txt'.format(now_str), 'w')
    file.writelines([item for item in lines[:-2]])
    file.write('Best_epoch: {}/{}\n'.format(best_epoch, current_epoch))
    file.write('Val_loss: {:.4f}'.format(lowest_val_TRE))
    file.close()
    print('Info updated in {}!'.format(now_str))


def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    lowest_loss = 2000
    lowest_dist = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}
    # print('trainmodel device {}'.format(device))

    for epoch in range(num_epochs):
        global current_epoch
        current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Network is in {}...'.format(phase))

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0 # store the sum of mse loss and corr loss 
            running_dist = 0.0 # store dist loss 
            running_corr = 0.0 # store corr loss 
            # running_corrects = 0

            # Iterate over data.
            for inputs, labels, case_id, start_params, calib_mat in dataloaders[phase]:
                # Get images from inputs
                # print('*'*10 + ' printing inputs and labels ' + '*'*10)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.type(torch.FloatTensor)
                # base_mat = base_mat.type(torch.FloatTensor)
                # img_id = img_id.type(torch.FloatTensor)
                labels = labels.to(device)
                inputs = inputs.to(device)
                # base_mat = base_mat.to(device)
                # img_id = img_id.to(device)

                labels.require_grad = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print('inputs shape {}'.format(inputs.shape))
                    # print('labels shape {}'.format(labels.shape))
                    # time.sleep(30)
                    outputs = model(inputs)
                    # print('outputs shape {}'.format(outputs.shape))
                    # time.sleep(30)


                    '''Weighted MSE loss function'''
                    # my_weight = torch.Tensor([0.5/4,0.5/2,0.5/2,0.5/4,0.5/4,0.5/4]).cuda()
                    # loss = weighted_mse_loss(input=outputs, target=labels, weights=my_weight)
                    # loss_weight = torch.Tensor([1, 1, 1, 1, 0, 0]).cuda().to(device)
                    # loss = weighted_mse_loss(input=outputs, target=labels, weights=loss_weight)

                    # print('outputs type {}, labels type {}'.format(outputs.dtype, labels.type))
                    dist_loss, drift_loss = get_dist_loss(labels=labels, outputs=outputs,
                                                          start_params=start_params, calib_mat=calib_mat)
                    corr_loss = get_correlation_loss(labels=labels, outputs=outputs)
                    # corr_loss = loss_functions.get_correlation_loss(labels=labels,
                    #                                                 outputs=outputs,
                    #                                                 dof_based=True)
                    # print('corr_loss {:.5f}'.format(corr_loss))
                    # print('dist_loss {:.5f}'.format(dist_loss))
                    # time.sleep(30)

                    loss = criterion(outputs, labels) # mse loss 

                    # loss = loss_functions.dof_MSE(labels=labels, outputs=outputs,
                    #                               criterion=criterion, dof_based=True)

                    # loss = loss + drift_loss
                    hybrid_loss = loss + corr_loss
                    # hybrid_loss = loss
                    # print('loss {:.5f}'.format(loss))
                    # print('m_dist {:.5f}'.format(m_dist))
                    # time.sleep(30)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # loss.backward()
                        hybrid_loss.backward()
                        # dist.backward()
                        optimizer.step()
                    # print('update loss')
                    # time.sleep(30)
                # statistics
                # running_loss += loss.item() * inputs.size(0)
                # running_loss += loss.data.mean() * inputs.size(0)
                running_loss += hybrid_loss.data.mean() * inputs.size(0)
                running_dist += dist_loss.item() * inputs.size(0)
                running_corr += corr_loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dist = running_dist / dataset_sizes[phase]
            epoch_corr = running_corr / dataset_sizes[phase]
            # print('epoch_dist {}'.format(epoch_dist))

            tv_hist[phase].append([epoch_loss, epoch_dist, epoch_corr])

            # deep copy the model
            # if (phase == 'val' and epoch_loss <= lowest_loss) or current_epoch % 10 == 0:
            if phase == 'val' and epoch_loss <= lowest_loss:
            # if phase == 'val' and epoch_dist <= lowest_dist:
                lowest_loss = epoch_loss
                # lowest_dist = epoch_dist
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
                # print('**** best model updated with dist={:.4f} ****'.format(lowest_dist))
                print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))

        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss)
        print('{}/{}: Tl: {:.4f}, Vl: {:.4f}, Td: {:.4f}, Vd: {:.4f}, Tc: {:.4f}, Vc: {:.4f}'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],
            tv_hist['val'][-1][0],
            tv_hist['train'][-1][1],
            tv_hist['val'][-1][1],
            tv_hist['train'][-1][2],
            tv_hist['val'][-1][2])
        )
        # time.sleep(30)
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        training_progress[epoch][1] = tv_hist['val'][-1][0]
        training_progress[epoch][2] = tv_hist['train'][-1][1]
        training_progress[epoch][3] = tv_hist['val'][-1][1]
        np.savetxt(txt_path, training_progress)

    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*' * 10 + 'Lowest val TRE: {:4f} at epoch {}'.format(lowest_dist, best_ep))
    print()

    return tv_hist