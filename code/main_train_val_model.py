"""
Main script to train and validate model in offline fashion
"""
# %%
# IMPORT MODULES

import torch.utils.tensorboard

# self written modules & parameters from config
import config
if config.code != 'predict':
    raise ValueError('Attention: code must be specified to be "predict" in config.py!')
from config import device
from config import path_training_data, path_validation_data 
from config import net
from config import wdw_size_i, wdw_size_o, centroid_norm
from config import lr, max_epochs, batch_size, l2
from config import loss_name, early_stopping_patience, input_data
from config import direction, breathhold_inclusion
from auxiliary import plotting, architectures, utils
from auxiliary import train_val_test, data_preparation

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
torch.cuda.empty_cache()
from monai.transforms import Compose, ToTensor, RandAffine

# %%
# CREATE FOLDER FOR RESULTS
 
path_tb = os.path.join(config.path_project_results, 'tensorboard', config.start_time_string + '_' + net)     
os.makedirs(path_tb, exist_ok=True)     
path_saving = os.path.join(config.path_project_results, net, 'optimization', config.start_time_string) 
os.makedirs(path_saving, exist_ok=True)

# %%
# DEFINE TRANSFORMS AND GET DATA

# transforms
if input_data == 'centroids':
    train_transforms = Compose([ToTensor(dtype=torch.float32)])  
    val_transforms = Compose([ToTensor(dtype=torch.float32)]) 

else:
    if config.augmentations is False:
        train_transforms = Compose([ToTensor(dtype=torch.float32)],)  
    else:
        train_transforms = Compose([RandAffine(prob=0.5, rotate_range=np.pi/3,
                                            shear_range=0.5,
                                            scale_range=0.2,
                                            mode='nearest',
                                            padding_mode='border'), 
                                    ToTensor(dtype=torch.float32)])
    val_transforms = Compose([ToTensor(dtype=torch.float32)]) 
    
# training data
train_set = data_preparation.OfflineSequentialDataset(path_data=path_training_data, 
                                input_data=input_data, direction=direction, 
                                breathhold_inclusion=breathhold_inclusion, 
                                wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o, 
                                step_size=1, centroid_norm=centroid_norm,
                                transforms=train_transforms)
print(f'Number of training samples: {train_set.__len__()}')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=False, num_workers=4,
                                            drop_last=False)
  
# validation data
val_set = data_preparation.OfflineSequentialDataset(path_data=path_validation_data, 
                                input_data=input_data, direction=direction, 
                                breathhold_inclusion=breathhold_inclusion, 
                                wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o, 
                                step_size=1, centroid_norm=centroid_norm,
                                transforms=val_transforms)
print(f'Number of validation samples: {val_set.__len__()}')
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=4,
                                            drop_last=False)

#%%
# PLOT EXAMPLARY DATA
example_inputs, example_outputs = next(iter(train_loader))
# example_inputs, example_outputs = next(iter(val_loader))
if input_data == 'centroids':
    print(f'Shape of input data:  {np.shape(example_inputs)}')  # (64, 16, 2) --> b,s,c
    print(f'Shape of output data:  {np.shape(example_outputs)}')  # (64, 2, 2) --> b,s,c
    plt.figure(figsize=(8,8))
    plt.title('Inputs')
    plt.plot(example_inputs[0,:,0], 'o-')
    plt.figure(figsize=(8,8))
    plt.title('Outputs')
    plt.plot(example_outputs[0,:,0], 'o-')
elif input_data == 'segmentations_and_frames':
    print(f'Shape of input data:  {np.shape(example_inputs)}')  #  (64, 16, 2, 512, 512) --> b,s,c,h,w
    print(f'Shape of output data:  {np.shape(example_outputs)}')  #  (64, 2, 2, 512, 512) --> b,s,c,h,w
    plt.figure(figsize=(8,8))
    plt.title('Inputs - seg')
    plt.imshow(example_inputs[0,-1,0,:,:], cmap='gray')
    plt.figure(figsize=(8,8))
    plt.title('Inputs - frame')
    plt.imshow(example_inputs[0,-1,1,:,:], cmap='gray')
    plt.figure(figsize=(8,8))
    plt.title('Outputs - seg')
    plt.imshow(example_outputs[0,0,0,:,:], cmap='gray')   
else:
    print(f'Shape of input data:  {np.shape(example_inputs)}')  #  (64, 16, 1, 512, 512) --> b,s,c,h,w
    print(f'Shape of output data:  {np.shape(example_outputs)}')  #  (64, 2, 1, 512, 512) --> b,s,c,h,w
    plt.figure(figsize=(8,8))
    plt.title('Inputs')
    plt.imshow(example_inputs[0,-1,0,:,:], cmap='gray')
    plt.figure(figsize=(8,8))
    plt.title('Outputs')
    plt.imshow(example_outputs[0,0,0,:,:], cmap='gray')  
# print(np.unique(example_inputs[0,-1,...]))  # [0. 1.] for segs

# %%
# MODEL BUILDING AND OPTIMIZATION

if net == 'LSTM_SI_AP':
    model = architectures.CentroidPredictionLSTM(input_features=config.input_features, 
                                hidden_features=config.hidden_features, 
                                output_features=config.output_features,
                                num_layers=config.num_layers, 
                                seq_len_in=wdw_size_i, seq_len_out=wdw_size_o, 
                                device=device, dropout=config.dropout, bi=config.bi)
    # set string for results folder
    net_params = f'net={net}-hidden_size={config.hidden_features}-num_layers={config.num_layers}' + \
                    f'-input_size={config.input_features}-batch_size={batch_size}-lr={lr}-epochs={max_epochs}' + \
                    f'-l2={l2}-dropout={config.dropout}-bi={config.bi}-loss_name={loss_name}'
elif net == 'ConvLSTM':
    model = architectures.SegmentationPredictionConvLSTM(input_shape=example_inputs.shape, 
                                                         seq_len_out=wdw_size_o, 
                                                         device=device, 
                                                         dec_input_init_zeros=config.dec_input_init_zeros)
    # set string for results folder
    net_params = f'net={net}-dec_input_init_zeros={config.dec_input_init_zeros}' + \
                    f'-batch_size={batch_size}-lr={lr}-epochs={max_epochs}' + \
                    f'-l2={l2}-loss_name={loss_name}' + \
                    f'-augmentations={config.augmentations}'         
elif net == 'ConvLSTM_STL':
    model = architectures.SegmentationPredictionConvLSTMSTL(input_shape=example_inputs.shape, 
                                                         seq_len_out=wdw_size_o, 
                                                         device=device, 
                                                         dec_input_init_zeros=config.dec_input_init_zeros,
                                                         max_displacement=config.max_displacement,
                                                         input_data=input_data)
    # set string for results folder
    net_params = f'net={net}-dec_input_init_zeros={config.dec_input_init_zeros}' + \
                    f'-batch_size={batch_size}-lr={lr}-epochs={max_epochs}' + \
                    f'-l2={l2}-max_dispacement={config.max_displacement}-loss_name={loss_name}'  + \
                    f'-augmentations={config.augmentations}'


# move model to device and get nr of parameters
model.to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {pytorch_total_params}\n')

# set optimizer etc
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
lr_scheduler = None
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# optimize the model
train_losses, val_losses, train_dice_metrics, val_dice_metrics,\
train_rmse_metrics, val_rmse_metrics, \
tot_t = train_val_test.train_val_model(model=model, train_loader=train_loader, 
                                    loss_name=loss_name, 
                                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    max_epochs=max_epochs, device=device, 
                                    early_stopping_patience=early_stopping_patience, 
                                    val_loader=val_loader,
                                    path_saving=path_saving, path_tb=path_tb)          

# %%
# SAVE STATS AND PLOTS

other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
                f'-step_size={1}-direction={direction}' + \
                f'-breathhold_inclusion={breathhold_inclusion}-{centroid_norm}' + \
                f'-input_data={input_data}'
                
# save some stats
utils.save_stats_train_val(path_saving, net_params, other_params, 
                            loss_name, train_losses, val_losses, 
                            train_dice_metrics, val_dice_metrics, 
                            train_rmse_metrics, val_rmse_metrics, tot_t) 

# plot and save resulting losses
plotting.losses_plot(train_losses=train_losses, val_losses=val_losses, 
                        loss_name=loss_name,
                        display=True, last_epochs=None, 
                        path_saving=os.path.join(path_saving, 'losses.png'))

# plot ground truth vs predicted for example data
example_inputs, example_outputs = example_inputs.to(device), example_outputs.to(device)
if net == 'LSTM_SI_AP':
    example_predictions = model(example_inputs)
    plotting.predicted_wdw_plot(example_inputs[:,:,0], example_outputs[:,:,0], example_predictions[:,:,0], wdw_nr=-1, last_pred=False,
                        display=True, path_saving=os.path.join(path_saving, f'predicted_SI_wdw_{-1}.png'))
    plotting.predicted_wdw_plot(example_inputs[:,:,1], example_outputs[:,:,1], example_predictions[:,:,1], wdw_nr=-1, last_pred=False,
                        display=True, path_saving=os.path.join(path_saving, f'predicted_AP_wdw_{-1}.png'))
elif net == 'ConvLSTM':
    example_predictions = model(example_inputs)
    plotting.in_out_pred_frames_plot_vertical(example_inputs, example_outputs, example_predictions, 
                                                nr_frames=wdw_size_o, display=True, 
                                                path_saving=os.path.join(path_saving, 'in_out_pred_predictions.png'))
    example_segmentations = (example_predictions > 0.5) * 1.0
    plotting.in_out_pred_frames_plot_vertical(example_inputs, example_outputs, example_segmentations, 
                                                nr_frames=wdw_size_o, display=True,
                                                path_saving=os.path.join(path_saving, 'in_out_pred_segmentations.png'))
elif net == 'ConvLSTMSTL':
    example_predictions, example_displacement_grids, example_computed_displacements = model(example_inputs)
    plotting.dvf_frame_overlay_plot(computed_displacements=example_computed_displacements, predictions=example_predictions, 
                           pat_nr=0, frame_nr=1, path_saving=os.path.join(path_saving, 'dvf_predictions_0_to_1_overlay.png'),
                           display=False, to_cpu=True)
    plotting.in_out_pred_frames_plot_vertical(example_inputs, example_outputs, example_predictions, 
                                                nr_frames=wdw_size_o, display=True, 
                                                path_saving=os.path.join(path_saving, 'in_out_pred_predictions.png'))
    example_segmentations = (example_predictions > 0.5) * 1.0
    plotting.in_out_pred_frames_plot_vertical(example_inputs, example_outputs, example_segmentations, 
                                                nr_frames=wdw_size_o, display=True, 
                                                path_saving=os.path.join(path_saving, 'in_out_pred_segmentations.png'))
# %%
