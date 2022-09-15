import numpy as np
import time
import torch
import os
import copy
import tqdm
from torch.nn import MSELoss
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, RMSEMetric
from torch.utils.tensorboard import SummaryWriter

# import self written modules
import utils, losses


def train_val_model(model, train_loader, loss_name, optimizer, lr_scheduler,
                max_epochs, device, early_stopping_patience=None, val_loader=None,
                path_saving=None, path_tb=None):
    """Function to train a Pytorch network (offline) and score some metrics.

    Args:
        model (Pytorch model): Network architecture
        train_loader (Pytorch laoder): pytorch data loader with training input and output
        loss_name (str): name of loss function, eg 'MSE', 'dice', 'dice_and_gradient'
        optimizer (Pytorch optimizer): optimizer used to update network's weights
        lr_scheduler (Pytorch scheduler): scheduler for learning rate
        max_epochs (int): maximum number of training epochs
        device (Pytorch device): cuda device
        early_stopping_patience (int, optional): whether to stop the optimization if the loss does
                                not get better after early_stopping nr of epochs. Defaults to None.
        val_loader (Pytorch laoder): pytorch data loader with validation input and output
        path_saving (str): path to folder where models are saved. Defaults to None.
        path_tb (str): path to folder where tensorboard runs are saved. 
                Must match tensorboard -- logdir path. 
                Defaults to None in which case no writer is initialized.

    Returns:
        train_losses (list): training losses for different epochs
        val_losses (list): val losses for different epochs
        train_metrics (dict): training metrics for different epochs subdivided by forecast (0,1,...)
        val_metrics (dict): val metrics for different epochs subdivided by forecast (0,1,...)
        tot_t (float): total time needed for optimization 
    """

    # set writer for tensorboard monitoring
    if path_tb is not None:
        writer = SummaryWriter(path_tb)
        
    # initialize metrics
    dice_metric0_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric1_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric0_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric1_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    rmse_metric0_train = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_train = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric0_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_val = RMSEMetric(reduction="mean", get_not_nans=False)

    
    train_losses = [] 
    train_rmse_metrics = {k: [] for k in range(2)}
    train_dice_metrics = {k: [] for k in range(2)}
    val_losses = [] 
    val_rmse_metrics = {k: [] for k in range(2)}
    val_dice_metrics = {k: [] for k in range(2)}
    best_val_loss = 1000000  # to be sure loss decreases

    t0 = time.time()
    
    # loop over all epochs
    for epoch in range(max_epochs):
        
        model.train()
        loss_train_epoch = [] 

        # loop over all batches of data
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for train_batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{max_epochs - 1}")
                train_inputs, train_outputs = train_batch_data[0].to(device), train_batch_data[1].to(device)
                # print(f'Shape of training inputs: {train_inputs.shape}')  #  b,s_in,c,h,w  for seg
                # print(f'Shape of training outputs: {train_outputs.shape}')  # b,s_out,c,h,w  for seg
                
                # clear stored gradients
                optimizer.zero_grad()

                # forward pass
                if model.__class__.__name__ == 'SegmentationPredictionConvLSTMSTL':
                    train_predictions, train_displacement_grids, train_computed_displacements = model(train_inputs)
                else:
                    train_predictions = model(train_inputs)

                if loss_name == 'MSE':
                    train_loss = MSELoss(reduction="mean")(train_predictions, train_outputs)
                elif loss_name == 'dice':
                    train_loss = (DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_predictions[:,0,...], train_outputs[:,0,...]) + \
                                    DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_predictions[:,1,...], train_outputs[:,1,...]))/2
                elif loss_name == 'focal':
                    train_loss = (FocalLoss(include_background=True, reduction="mean", gamma=2)(train_predictions[:,0,...], train_outputs[:,0,...]) + \
                                    FocalLoss(include_background=True, reduction="mean", gamma=2)(train_predictions[:,1,...], train_outputs[:,1,...]))/2
                elif loss_name == 'dice_and_gradient':
                    train_computed_displacements_reshaped = torch.reshape(train_computed_displacements, (-1, 2,
                                                                            train_computed_displacements.shape[-3], 
                                                                            train_computed_displacements.shape[-2])) # b*s, 2, h, w
                    train_loss = losses.combined_image_dvf_loss(train_predictions, train_outputs, train_computed_displacements_reshaped)
                elif loss_name == 'dice_and_centroid':
                    train_loss = losses.combined_image_centroid_loss(train_predictions, train_outputs, device,  lambda_image=0.5, lambda_centroid=0.5)      
                else:
                    raise ValueError('Unkown loss_name specified!')
                
                # back propagate the errors and update the weights within batch    
                train_loss.backward()
                optimizer.step()
                loss_train_epoch.append(train_loss.item())
                # print(model.state_dict()['conv_enc1.0.bias'])
                # print(train_loss.item())
                
                if loss_name == 'MSE': 
                    rmse_metric0_train(train_predictions[:,0,...], train_outputs[:,0,...])
                    rmse_metric1_train(train_predictions[:,1,...], train_outputs[:,1,...])
                else:    
                    # binarize segmentation and compute dice metric
                    train_segmentations = (train_predictions > 0.5) * 1.0
                    dice_metric0_train(y_pred=train_segmentations[:,0,...], y=train_outputs[:,0,...])
                    dice_metric1_train(y_pred=train_segmentations[:,1,...], y=train_outputs[:,1,...])

            # compute loss for current epoch by averaging over batch losses and store results
            average_loss_train_epoch = np.mean(loss_train_epoch) 
            train_losses.append(average_loss_train_epoch)
            if loss_name == 'MSE':
                average_rmse_metric0_train_epoch = rmse_metric0_train.aggregate().item()
                average_rmse_metric1_train_epoch = rmse_metric1_train.aggregate().item()
                train_rmse_metrics[0].append(average_rmse_metric0_train_epoch)
                train_rmse_metrics[1].append(average_rmse_metric1_train_epoch)
            else:
                average_dice_metric0_train_epoch = dice_metric0_train.aggregate().item()
                average_dice_metric1_train_epoch = dice_metric1_train.aggregate().item()
                train_dice_metrics[0].append(average_dice_metric0_train_epoch)
                train_dice_metrics[1].append(average_dice_metric1_train_epoch)

            if lr_scheduler is not None:
                lr_scheduler.step()
                
            if path_tb is not None:
                writer.add_scalar(f"train_{loss_name}_loss", average_loss_train_epoch, epoch)
                if loss_name == 'MSE':
                    writer.add_scalar(f"train_rmse_metric0", average_rmse_metric0_train_epoch, epoch)
                    writer.add_scalar(f"train_rmse_metric1", average_rmse_metric1_train_epoch, epoch)
                else:               
                    writer.add_scalar(f"train_dice_metric0", average_dice_metric0_train_epoch, epoch)
                    writer.add_scalar(f"train_dice_metric1", average_dice_metric1_train_epoch, epoch)
            
            # validation 
            if val_loader is not None:
                loss_val_epoch = []          
                
                with torch.no_grad():
                    model.eval()
                    for val_batch_data in val_loader:
                        val_inputs, val_outputs = val_batch_data[0].to(device), val_batch_data[1].to(device)
                        # print(f'Shape of validation inputs: {val_inputs.shape}')  # (b,s_in,c,h,w)
                        # print(f'Shape of validation outputs: {val_outputs.shape}')  # (b,s_out,c,h,w)
                        
                        # forward pass
                        if model.__class__.__name__ == 'SegmentationPredictionConvLSTMSTL':
                            val_predictions, val_displacement_grids, val_computed_displacements = model(val_inputs)
                        else:
                            val_predictions = model(val_inputs)

                        if loss_name == 'MSE':
                            val_loss = MSELoss(reduction="mean")(val_predictions, val_outputs)
                        elif loss_name == 'dice':
                            val_loss = (DiceLoss(include_background=True, sigmoid=False, reduction="mean")(val_predictions[:,0,...], val_outputs[:,0,...]) + \
                                        DiceLoss(include_background=True, sigmoid=False, reduction="mean")(val_predictions[:,1,...], val_outputs[:,1,...]))/2
                        elif loss_name == 'focal':
                            val_loss = (FocalLoss(include_background=True, reduction="mean", gamma=2)(val_predictions[:,0,...], val_outputs[:,0,...]) + \
                                            FocalLoss(include_background=True, reduction="mean", gamma=2)(val_predictions[:,1,...], val_outputs[:,1,...]))/2
                        elif loss_name == 'dice_and_gradient':
                            val_computed_displacements_reshaped = torch.reshape(val_computed_displacements, (-1, 2,
                                                                                    val_computed_displacements.shape[-3], 
                                                                                    val_computed_displacements.shape[-2])) # b*s, 2, h, w
                            val_loss = losses.combined_image_dvf_loss(val_predictions, val_outputs, val_computed_displacements_reshaped)
                        elif loss_name == 'dice_and_centroid':
                            val_loss = losses.combined_image_centroid_loss(val_predictions, val_outputs, device)           
                        loss_val_epoch.append(val_loss.item())
                        
                        if loss_name == 'MSE':
                            rmse_metric0_val(val_predictions[:,0,...], val_outputs[:,0,...])
                            rmse_metric1_val(val_predictions[:,1,...], val_outputs[:,1,...]) 
                        else:                 
                            # binarize segmatation and compute dice metric
                            val_segmentations = (val_predictions > 0.5) * 1.0
                            dice_metric0_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
                            dice_metric1_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
                        
                    # compute loss for current epoch by averaging over batch losses and store results
                    average_loss_val_epoch = np.mean(loss_val_epoch) 
                    val_losses.append(average_loss_val_epoch)
                    if loss_name == 'MSE':
                        average_rmse_metric0_val_epoch = rmse_metric0_val.aggregate().item()
                        average_rmse_metric1_val_epoch = rmse_metric1_val.aggregate().item()
                        val_rmse_metrics[0].append(average_rmse_metric0_val_epoch)
                        val_rmse_metrics[1].append(average_rmse_metric1_val_epoch)       
                    else:             
                        average_dice_metric0_val_epoch = dice_metric0_val.aggregate().item()
                        average_dice_metric1_val_epoch = dice_metric1_val.aggregate().item()
                        val_dice_metrics[0].append(average_dice_metric0_val_epoch)
                        val_dice_metrics[1].append(average_dice_metric1_val_epoch)
            
                if path_tb is not None:
                    writer.add_scalar(f"val_{loss_name}_loss", average_loss_val_epoch, epoch)
                    if loss_name == 'MSE':
                        writer.add_scalar(f"val_rmse_metric0", average_rmse_metric0_val_epoch, epoch)
                        writer.add_scalar(f"val_rmse_metric1", average_rmse_metric1_val_epoch, epoch)
                    else:
                        writer.add_scalar(f"val_dice_metric0", average_dice_metric0_val_epoch, epoch)
                        writer.add_scalar(f"val_dice_metric1", average_dice_metric1_val_epoch, epoch)
                        
                                            
                # save model if validation loss improves
                if val_losses[-1] < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = val_losses[-1]
                    if path_saving is not None:
                        torch.save(model.state_dict(), os.path.join(
                            path_saving, f'best_model_epoch_{best_epoch}_{loss_name}_' + \
                            f'val_loss_{best_val_loss:.6f}.pth'))
                        print('...saved model based on new best validation loss.') 
                        
                if epoch % 1 == 0:
                    if loss_name == 'MSE':
                        print(f'Train loss: {average_loss_train_epoch} - '
                            f'Val loss: {average_loss_val_epoch} - '
                            f'Best val loss: {best_val_loss} - \n'      
                            f'Train rmse metric (250 ms forecast): {average_rmse_metric0_train_epoch} - '
                            f'Val rmse metric (250 ms forecast): {average_rmse_metric0_val_epoch} - \n'
                            f'Train rmse metric (500 ms forecast): {average_rmse_metric1_train_epoch} - '
                            f'Val rmse metric (500 ms forecast): {average_rmse_metric1_val_epoch}') 
                    else:
                        print(f'Train loss: {average_loss_train_epoch} - '
                            f'Val loss: {average_loss_val_epoch} - '
                            f'Best val loss: {best_val_loss} - \n'      
                            f'Train dice metric (250 ms forecast): {average_dice_metric0_train_epoch} - '
                            f'Val dice metric (250 ms forecast): {average_dice_metric0_val_epoch} - \n'
                            f'Train dice metric (500 ms forecast): {average_dice_metric1_train_epoch} - '
                            f'Val dice metric (500 ms forecast): {average_dice_metric1_val_epoch}') 
                                
                # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
                if early_stopping_patience is not None:
                    if (epoch - best_epoch) > early_stopping_patience:
                        print('Early stopping the optimization!')
                        break
                                            
            else:
                if epoch % 1 == 0:
                    if loss_name == 'MSE':
                        print(f'Train loss: {average_loss_train_epoch} - \n'     
                            f'Train rmse metric (250 ms forecast): {average_rmse_metric0_train_epoch} - '
                            f'Train rmse metric (500 ms forecast): {average_rmse_metric1_train_epoch}')
                    else:
                        print(f'Train loss: {average_loss_train_epoch} - \n'     
                            f'Train dice metric (250 ms forecast): {average_dice_metric0_train_epoch} - '
                            f'Train dice metric (500 ms forecast): {average_dice_metric1_train_epoch}')                   
                        
    # close tensorboard writer 
    if path_tb is not None:          
        writer.close()
      
    t1 = time.time()
    tot_t = round((t1 - t0) / 60)
    print(f'\n------------ Total time needed for optimization: {tot_t} min --------- ') 
    
    return train_losses, val_losses, train_dice_metrics, val_dice_metrics, \
            train_rmse_metrics, val_rmse_metrics, tot_t

    

def train_val_model_online(model_initial, data_loader,
                            input_data, loss_name, lr, l2,
                            wdw_size_i, wdw_size_o,
                            device, min_train_data_length=80,
                            centroid_norm='wdw_norm', online_epochs=10):
    """Train a Pytorch network on first window of data and then train and
    validate it continuosly (i.e. online) on the remaining sliding windows. 
    By setting online_epochs to None it is possible to solely validate on cine video basis.

    Args:
        model_initial (Pytorch model): Network architecture
        data_loader (Pytorch laoder): pytorch data loader with training input and output
        loss_name (str): name of loss function, eg 'MSE', 'dice', 'dice_and_gradient' for onlien optimization
        lr (int): learning rate for online Adam optimizer 
        l2 (int): weight decay for online Adam optimzer 
        wdw_size_i (int): length of input sequences
        wdw_size_o (int): length of output sequences
        device (Pytorch device): cuda device
        min_train_data_length (str): nr of data points of data points used for onlien training.
                                        80 datapoints correspond to 20 s of motion data.
        centroid_norm (str): normalization scheme for centroid motion curve.
        online_epochs (int): nr of online training epochs

    Returns:
        metrics_videos (dict with lists): dictionary with metrics for different forecasted time points (0=250 ms; 1=500ms) for different cine videos
        tot_t_online_videos (list): total time needed for online training for different cine videos
    """
     
    # initialize metrics
    dice_metric0_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric1_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd_metric0_max_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_max_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric0_95_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_95_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric0_50_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_50_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    rmse_metric0_SI_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_SI_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric0_AP_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_AP_val = RMSEMetric(reduction="mean", get_not_nans=False)
    
    # define dictionary to store metric for different cine videos
    val_dice_metric_videos = {k: [] for k in range(2)}
    val_hd_max_metric_videos = {k: [] for k in range(2)}
    val_hd_95_metric_videos = {k: [] for k in range(2)}
    val_hd_50_metric_videos = {k: [] for k in range(2)}
    val_rmse_SI_metric_videos = {k: [] for k in range(2)}
    val_rmse_AP_metric_videos = {k: [] for k in range(2)}
    tot_t_online_videos = []
    
    for batch_data in data_loader: 
        tot_times_online = []
        
        # get series of input and output sequences for one cine video
        inputs, outputs = batch_data[0], batch_data[1]
        # print(f'Shape of input batch: {inputs.shape}')  #  b,s_in,c,h,w  for seg
        # print(f'Shape of output batch: {outputs.shape}')  # b,s_out,c,h,w  for seg
    
        # check if current batch has enough data windows
        length_video = wdw_size_i + 1 * inputs.shape[0] - 1
        if length_video <= min_train_data_length + wdw_size_o:
            print(f'Current video was excluded as length too short: {length_video} time points.')
            continue
        
        # set model back to initial model for every cine video
        model = copy.deepcopy(model_initial)
        # print(f'Model parameters initial: {list(model.parameters())[-1]}')
        
        # set Adam optimizer for possible online training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        
        # loop over all sequences of current video (i.e. over the batch dimension)
        for wdw_nr in range(len(inputs) - (min_train_data_length - wdw_size_i + 1) - wdw_size_o):
            # get indices for set of windows with length 
            # min_train_data_length for online training
            wdw_nr_train_start = wdw_nr 
            wdw_nr_train_stop = wdw_nr + (min_train_data_length - wdw_size_i) 
            # get index of currently available validation input window
            # and of ground truth output window (of course not possible in real-time scenario,
            # only used to compute validation metric) 
            wdw_nr_val = wdw_nr_train_stop + wdw_size_o
            
            # get validation input and output window
            val_inputs, val_outputs = inputs[wdw_nr_val, None, ...].to(device), outputs[wdw_nr_val, None, ...].to(device)
            # print(f'val_input.shape: {val_inputs.shape}')  # torch.size([1, wdw_size_i, ...])
            # print(f'val_output.shape: {val_outputs.shape}')  # torch.size([1, wdw_size_o, ...])
            
            if input_data == 'centroids' and centroid_norm == 'wdw_norm':
                max_amplitude_SI, min_amplitude_SI = torch.max(val_inputs[:,:,0]).item(), torch.min(val_inputs[:,:,0]).item()
                max_amplitude_AP, min_amplitude_AP = torch.max(val_inputs[:,:,1]).item(), torch.min(val_inputs[:,:,1]).item()
                
                # normalize current window
                t0_norm = time.time()
                val_inputs[:,:,0] = utils.normalize(val_inputs[:,:,0], 
                                        {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                t1_norm = time.time()
                # print(f'Time needed for online normalization of wdw: {round((t1_norm - t0_norm) * 1000)} ms')
                val_outputs[:,:,0] = utils.normalize(val_outputs[:,:,0], 
                                        {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                val_inputs[:,:,1] = utils.normalize(val_inputs[:,:,1], 
                                        {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True)
                val_outputs[:,:,1] = utils.normalize(val_outputs[:,:,1], 
                                        {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                
            # set model to eval model for validation  
            model.eval()

            # forward pass
            if model.__class__.__name__ == 'SegmentationPredictionConvLSTMSTL':
                t0_forward = time.time()
                val_predictions, _, _ = model(val_inputs) 
                t1_forward = time.time()
            else:
                t0_forward = time.time()
                val_predictions = model(val_inputs) 
                t1_forward = time.time()                
            # print(f'Time needed for forward pass: {round((t1_forward - t0_forward) * 1000)} ms')
            # print(f'...sending prediction to MLC for wdw_nr {wdw_nr}')

            # shift last input image by difference between predicted centroids and centroids of last image
            if input_data == 'centroids': 
                # get amplitudes and segmenations
                val_inputs_seg = batch_data[2][wdw_nr_val, None, ...]
                val_outputs_seg = batch_data[3][wdw_nr_val, None, ...]
                if centroid_norm == 'video_norm':
                    max_amplitude_AP, min_amplitude_AP = batch_data[4], batch_data[5] 
                    max_amplitude_SI, min_amplitude_SI = batch_data[6], batch_data[7]
                
                # shift last input segmentation
                t0_shift = time.time()
                val_predictions_seg = utils.shift_by_centroid_diff(predictions=val_predictions, 
                                                                    inputs=val_inputs, inputs_seg=val_inputs_seg,
                                                                    min_amplitude_SI=min_amplitude_SI, max_amplitude_SI=max_amplitude_SI, 
                                                                    min_amplitude_AP=min_amplitude_AP, max_amplitude_AP=max_amplitude_AP)
                t1_shift = time.time()
                # print(f'Time needed for seg shift: {round((t1_shift - t0_shift) * 1000)} ms')
                
                val_outputs = val_outputs_seg
                # binarize segmentation
                val_segmentations = (val_predictions_seg > 0.5) * 1.0
            else:
                # binarize segmentation
                val_segmentations = (val_predictions > 0.5) * 1.0
                
            # get center of mass of seg to compute rmse
            centroids_output0, centroids_segmentations0, \
            centroids_output1, centroids_segmentations1 = utils.get_centroids_segmentations(val_outputs, val_segmentations)
            
            # compute metrics for current iteration (cumulatively)    
            dice_metric0_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
            dice_metric1_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
            hd_metric0_max_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
            hd_metric1_max_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
            hd_metric0_95_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
            hd_metric1_95_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
            hd_metric0_50_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
            hd_metric1_50_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
            rmse_metric0_SI_val(centroids_segmentations0[None,0,None] , centroids_output0[None,0,None])
            rmse_metric1_SI_val(centroids_segmentations1[None,0,None], centroids_output1[None,0,None])
            rmse_metric0_AP_val(centroids_segmentations0[None,1,None], centroids_output0[None,1,None])
            rmse_metric1_AP_val(centroids_segmentations1[None,1,None], centroids_output1[None,1,None])


            # training (happens 'after' validation for iterative model as
            # optimization takes some time)
            if online_epochs is not None:
                # get input and output set of windows for online training
                train_inputs = inputs[wdw_nr_train_start:wdw_nr_train_stop + 1, ...].to(device)
                train_outputs = outputs[wdw_nr_train_start:wdw_nr_train_stop + 1, ...].to(device)
                # print(f'train_inputs.shape: {train_inputs.shape}') # torch.size([nr wdws to reach min_train_data_length, wdw_size_i, ...])
                # print(f'train_outputs.shape: {train_outputs.shape}')  # torch.size([nr wdws to reach min_train_data_length, wdw_size_o, ...]) 
 
                if input_data == 'centroids' and centroid_norm == 'wdw_norm':
                    max_amplitude_SI, min_amplitude_SI = torch.max(train_inputs[:,:,0]).item(), torch.min(train_inputs[:,:,0]).item()
                    max_amplitude_AP, min_amplitude_AP = torch.max(train_inputs[:,:,1]).item(), torch.min(train_inputs[:,:,1]).item()
                    
                    # normalize current set of windows
                    t0_norm_online = time.time()
                    train_inputs[:,:,0] = utils.normalize(train_inputs[:,:,0], 
                                            {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                            'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                    t1_norm_online = time.time()
                    # print(f'Time needed for online normalization of set of wdws: {round((t1_norm_online - t0_norm_online) * 1000)} ms')
                    train_outputs[:,:,0] = utils.normalize(train_outputs[:,:,0], 
                                            {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                            'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                    train_inputs[:,:,1] = utils.normalize(train_inputs[:,:,1], 
                                            {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                            'desired': {'lower': -1, 'upper': 1}}, to_tensor=True)
                    train_outputs[:,:,1] = utils.normalize(train_outputs[:,:,1], 
                                            {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                            'desired': {'lower': -1, 'upper': 1}}, to_tensor=True)                
                       
                loss_train_online_epochs = []      
                t0_online = time.time()
                # print('Online training...')
                # loop over all epochs of online training
                for epoch in range(online_epochs):
                    model.train()
                
                    # clear stored gradients
                    optimizer.zero_grad()

                    # forward pass
                    if model.__class__.__name__ == 'SegmentationPredictionConvLSTMSTL':
                        train_predictions, train_displacement_grids, train_computed_displacements = model(train_inputs)
                    else:
                        train_predictions = model(train_inputs)

                    # compute the loss for current windows
                    if loss_name == 'MSE':
                        train_loss_online = MSELoss(reduction="mean")(train_predictions, train_outputs)
                    elif loss_name == 'dice':
                        train_loss_online = (DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_predictions[:,0,...], train_outputs[:,0,...]) + \
                                                DiceLoss(include_background=True, sigmoid=False, reduction="mean")(train_predictions[:,1,...], train_outputs[:,1,...]))/2                    
                    elif loss_name == 'dice_and_gradient':
                        train_computed_displacements_reshaped = torch.reshape(train_computed_displacements, (-1, 2,
                                                                                train_computed_displacements.shape[-3], 
                                                                                train_computed_displacements.shape[-2])) # b*s, 2, h, w
                        train_loss_online = losses.combined_image_dvf_loss(train_predictions, train_outputs, train_computed_displacements_reshaped)
                    else:
                        raise ValueError('Unkown loss_name specified!')
                            
                    # backpropagate the errors and update the weights for current window
                    train_loss_online.backward()
                    optimizer.step()
                    
                    # append online training loss for current apoch and current video 
                    loss_train_online_epochs.append(train_loss_online.item())
                
                # print(f'Model parameters after optimization: {list(model.parameters())[-1]}')
                t1_online = time.time()    
                tot_t_online = round((t1_online - t0_online) * 1000)
                # print(f'Time needed for {online_epochs} online epochs: {tot_t_online} ms \n')       
                tot_times_online.append(tot_t_online)   
        
        # aggregate the final metric result using the specified reduction 
        # and append metric value for current video      
        val_dice_metric_videos[0].append(dice_metric0_val.aggregate().item())
        val_dice_metric_videos[1].append(dice_metric1_val.aggregate().item())
        val_hd_max_metric_videos[0].append(hd_metric0_max_val.aggregate().item())
        val_hd_max_metric_videos[1].append(hd_metric1_max_val.aggregate().item())
        val_hd_95_metric_videos[0].append(hd_metric0_95_val.aggregate().item())
        val_hd_95_metric_videos[1].append(hd_metric1_95_val.aggregate().item())
        val_hd_50_metric_videos[0].append(hd_metric0_50_val.aggregate().item())
        val_hd_50_metric_videos[1].append(hd_metric1_50_val.aggregate().item())
        val_rmse_SI_metric_videos[0].append(rmse_metric0_SI_val.aggregate().item())
        val_rmse_SI_metric_videos[1].append(rmse_metric1_SI_val.aggregate().item())
        val_rmse_AP_metric_videos[0].append(rmse_metric0_AP_val.aggregate().item())
        val_rmse_AP_metric_videos[1].append(rmse_metric1_AP_val.aggregate().item())
        if online_epochs is not None:
            tot_t_online_videos.append(np.mean(tot_times_online))
        
        # reset metrics for next cine video
        dice_metric0_val.reset()
        dice_metric1_val.reset()
        hd_metric0_max_val.reset()
        hd_metric1_max_val.reset()
        hd_metric0_95_val.reset()
        hd_metric1_95_val.reset()
        hd_metric0_50_val.reset()
        hd_metric1_50_val.reset()
        rmse_metric0_SI_val.reset()
        rmse_metric1_SI_val.reset()
        rmse_metric0_AP_val.reset()
        rmse_metric1_AP_val.reset()
            
        # clear gpu memory after each batch
        torch.cuda.empty_cache()
    
    if online_epochs is None:            
        return val_dice_metric_videos, val_hd_max_metric_videos, val_hd_95_metric_videos, val_hd_50_metric_videos, \
                val_rmse_SI_metric_videos, val_rmse_AP_metric_videos, \
                [], []    
    else:
        return val_dice_metric_videos,val_hd_max_metric_videos, val_hd_95_metric_videos, val_hd_50_metric_videos, \
                val_rmse_SI_metric_videos, val_rmse_AP_metric_videos, \
                loss_train_online_epochs, tot_t_online_videos    
               
                
 
def evaluate_no_predictor(data_loader, wdw_size_i, wdw_size_o,
                          min_train_data_length, device):
    """Evaluate no predictor model on a cine video basis by taking the last input segmentation and 
    comparing it with the specifed ground truth output segmentations. 
    """
    # initialize metrics
    dice_metric0_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    dice_metric1_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
    hd_metric0_max_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_max_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric0_95_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_95_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric0_50_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    hd_metric1_50_val = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=50, 
                                                    directed=False, reduction="mean", get_not_nans=False)
    rmse_metric0_SI_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_SI_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric0_AP_val = RMSEMetric(reduction="mean", get_not_nans=False)
    rmse_metric1_AP_val = RMSEMetric(reduction="mean", get_not_nans=False)
    
    val_dice_metric_videos = {k: [] for k in range(2)}
    val_hd_max_metric_videos = {k: [] for k in range(2)}
    val_hd_95_metric_videos = {k: [] for k in range(2)}
    val_hd_50_metric_videos = {k: [] for k in range(2)}
    val_rmse_SI_metric_videos = {k: [] for k in range(2)}
    val_rmse_AP_metric_videos = {k: [] for k in range(2)}
    
    for batch_data in data_loader:
        # get series of input and output sequences for one cine video
        val_inputs, val_outputs = batch_data[0].to(device), batch_data[1].to(device)
        # print(f'Shape of inputs: {inputs.shape}')  #  b,s_in,c,h,w  for seg
        # print(f'Shape of outputs: {outputs.shape}')  # b,s_out,c,h,w  for seg
        
        # check if current batch has enough data windows, analogously with onlien training
        length_video = wdw_size_i + 1 * val_inputs.shape[0] - 1
        if length_video <= min_train_data_length + wdw_size_o:
            print(f'Current video was excluded as length too short: {length_video} time points.')
            continue
    
        # get 'predicted' segmentations by taking the last input image
        # for the existing nr of outputs
        val_segmentations0 = val_inputs[:, -1, None, ...]
        val_segmentations1 = val_inputs[:, -1, None, ...]
        val_segmentations = torch.cat((val_segmentations0, val_segmentations1), dim=1)
            
        # get center of mass of seg to compute rmse
        centroids_output0, centroids_segmentations0, \
        centroids_output1, centroids_segmentations1 = utils.get_centroids_segmentations(val_outputs, val_segmentations)
            
        # compute metrics for current iteration (cumulatively)    
        dice_metric0_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
        dice_metric1_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
        hd_metric0_max_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
        hd_metric1_max_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
        hd_metric0_95_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
        hd_metric1_95_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
        hd_metric0_50_val(y_pred=val_segmentations[:,0,...], y=val_outputs[:,0,...])
        hd_metric1_50_val(y_pred=val_segmentations[:,1,...], y=val_outputs[:,1,...])
        rmse_metric0_SI_val(centroids_segmentations0[None,0,None] , centroids_output0[None,0,None])
        rmse_metric1_SI_val(centroids_segmentations1[None,0,None], centroids_output1[None,0,None])
        rmse_metric0_AP_val(centroids_segmentations0[None,1,None], centroids_output0[None,1,None])
        rmse_metric1_AP_val(centroids_segmentations1[None,1,None], centroids_output1[None,1,None])

        # aggregate the final metric result using the specified reduction 
        # and append metric value for current video      
        val_dice_metric_videos[0].append(dice_metric0_val.aggregate().item())
        val_dice_metric_videos[1].append(dice_metric1_val.aggregate().item())
        val_hd_max_metric_videos[0].append(hd_metric0_max_val.aggregate().item())
        val_hd_max_metric_videos[1].append(hd_metric1_max_val.aggregate().item())
        val_hd_95_metric_videos[0].append(hd_metric0_95_val.aggregate().item())
        val_hd_95_metric_videos[1].append(hd_metric1_95_val.aggregate().item())
        val_hd_50_metric_videos[0].append(hd_metric0_50_val.aggregate().item())
        val_hd_50_metric_videos[1].append(hd_metric1_50_val.aggregate().item())
        val_rmse_SI_metric_videos[0].append(rmse_metric0_SI_val.aggregate().item())
        val_rmse_SI_metric_videos[1].append(rmse_metric1_SI_val.aggregate().item())
        val_rmse_AP_metric_videos[0].append(rmse_metric0_AP_val.aggregate().item())
        val_rmse_AP_metric_videos[1].append(rmse_metric1_AP_val.aggregate().item())
        
        # reset metrics for next cine video
        dice_metric0_val.reset()
        dice_metric1_val.reset()
        hd_metric0_max_val.reset()
        hd_metric1_max_val.reset()
        hd_metric0_95_val.reset()
        hd_metric1_95_val.reset()
        hd_metric0_50_val.reset()
        hd_metric1_50_val.reset()
        rmse_metric0_SI_val.reset()
        rmse_metric1_SI_val.reset()
        rmse_metric0_AP_val.reset()
        rmse_metric1_AP_val.reset()
    
    return val_dice_metric_videos, val_hd_max_metric_videos, val_hd_95_metric_videos, val_hd_50_metric_videos, \
            val_rmse_SI_metric_videos, val_rmse_AP_metric_videos, \
            [], []    

   