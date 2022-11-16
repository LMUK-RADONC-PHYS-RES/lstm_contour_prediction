"""
Created on May 25 2021

@author: Elia Lombardo

Main script to test model and if needed continuosly train model on single traces for centroid position prediction
"""
# %%
# IMPORT MODULES

# self written modules & parameters from config
import config

from config import device, online_training
if online_training is False:
    raise ValueError("Attention: this script is supposed to be run with config.online_training=True ! ")
from config import path_cases
# path_cases = path_cases[17:] # small test
from config import net, input_data
from config import wdw_size_i, wdw_size_o
from config import lr, l2, centroid_norm
from config import loss_name, path_trained_model
from config import direction, breathhold_inclusion, cohort
from config import min_train_data_length, online_epochs
from auxiliary import plotting, architectures, utils
from auxiliary import train_val_test, data_preparation

# %%
import os
import torch
torch.cuda.empty_cache()
import numpy as np
import matplotlib.pyplot as plt

# %%
# CREATE FOLDER FOR RESULTS
 
path_tb = os.path.join(config.path_project_results, 'tensorboard', config.start_time_string + '_' + net)     
os.makedirs(path_tb, exist_ok=True)     
path_saving = os.path.join(config.path_project_results, net, 'inference', config.start_time_string) 
os.makedirs(path_saving, exist_ok=True)

# %%
# GET DATA

# online training/validation/testing data
data_set = data_preparation.OnlineSequentialDataset(path_data=path_cases, 
                                input_data=input_data, direction=direction, 
                                breathhold_inclusion=breathhold_inclusion, 
                                wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o, 
                                step_size=1, centroid_norm=centroid_norm)
print(f'Number of data samples: {data_set.__len__()}')
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,
                                            shuffle=False, num_workers=0, 
                                            drop_last=False,
                                            collate_fn=data_preparation.collate_correct_dim)
  
#%%
# PLOT EXAMPLE DATA
iterator = iter(data_loader)
if input_data == 'centroids':
    for i in range(4):
        example_inputs, example_outputs, example_inputs_seg, example_outputs_seg, \
            max_amplitude_AP, min_amplitude_AP, max_amplitude_SI, min_amplitude_SI = next(iterator)
    print(f'Shape of input data:  {np.shape(example_inputs)}')  # (133, 32, 2) --> b,s,c
    print(f'Shape of output data:  {np.shape(example_outputs)}')  # (133, 2, 2) --> b,s,c
    if np.shape(example_inputs)[0] > 0:
        plt.figure()
        plt.title('Inputs')
        plt.plot(example_inputs[0,:,0], 'o-')
        plt.figure()
        plt.title('Outputs')
        plt.plot(example_outputs[0,:,0], 'o-')
        print(f'Shape of input seg data:  {np.shape(example_inputs_seg)}')  #  (133, 32, 1, 256, 256) --> b,s,c,h,w
        print(f'Shape of output seg data:  {np.shape(example_outputs_seg)}')  #  (133, 2, 1, 256, 256) --> b,s,c,h,w
        plt.figure()
        plt.title('Inputs seg')
        plt.imshow(example_inputs_seg[0,-1,0,:,:], cmap='gray')
        plt.figure()
        plt.title('Outputs seg')
        plt.imshow(example_outputs_seg[0,0,0,:,:], cmap='gray')  
else:
    for i in range(4):
        example_inputs, example_outputs = next(iterator)
    print(f'Shape of input data:  {np.shape(example_inputs)}')  #  (133, 32, 1, 256, 256) --> b,s,c,h,w
    print(f'Shape of output data:  {np.shape(example_outputs)}')  #  (133, 2, 1, 256, 256) --> b,s,c,h,w
    if np.shape(example_inputs)[0] > 0:
        plt.figure()
        plt.title('Inputs')
        plt.imshow(example_inputs[0,-1,0,:,:], cmap='gray')
        plt.figure()
        plt.title('Outputs')
        plt.imshow(example_outputs[0,0,0,:,:], cmap='gray')  
        # print(np.unique(example_inputs))  # [0. 1.]       
print('\n')

# %%
# INITIALIZE MODELS AND LOAD PREVIOUSLY TRAINED OFFLINE MODEL

if net == 'LSTM_SI_AP':
    model = architectures.CentroidPredictionLSTM(input_features=config.input_features, 
                                                    hidden_features=config.hidden_features, 
                                                    output_features=config.output_features,
                                                    num_layers=config.num_layers, 
                                                    seq_len_out=wdw_size_o,
                                                    dropout=0, bi=config.bi,
                                                    device=device)
    net_params = f'net={net}-hidden_size={config.hidden_features}-num_layers={config.num_layers}' + \
                    f'-input_size={config.input_features}-lr={lr}' + \
                    f'-l2={l2}-dropout={config.dropout}-loss_name={loss_name}-bi={config.bi}' + \
                    f'-trained_model={path_trained_model}' 
elif net == 'ConvLSTM':
    model = architectures.SegmentationPredictionConvLSTM(input_shape=example_inputs.shape, 
                                                        seq_len_out=wdw_size_o, 
                                                        device=device, 
                                                        dec_input_init_zeros=config.dec_input_init_zeros)
    net_params = f'net={net}-dec_input_init_zeros={config.dec_input_init_zeros}' + \
                    f'-lr={lr}-l2={l2}-loss_name={loss_name}-trained_model={path_trained_model}'  
elif net == 'ConvLSTM_STL':
    model = architectures.SegmentationPredictionConvLSTMSTL(input_shape=example_inputs.shape, 
                                                        seq_len_out=wdw_size_o, 
                                                        device=device, 
                                                        dec_input_init_zeros=config.dec_input_init_zeros,
                                                        max_displacement=config.max_displacement,
                                                        input_data=input_data)
    net_params = f'net={net}-dec_input_init_zeros={config.dec_input_init_zeros}' + \
                    f'-lr={lr}-l2={l2}-loss_name={loss_name}-max_displacement={config.max_displacement}-trained_model={path_trained_model}' 
                    

if path_trained_model is not None:
    # load trained model
    model_files = [] 
    losses = [] 
    # loop over all subfolders and files of one pre training
    for dir_name, subdir_list, file_list in os.walk(path_trained_model):
        for file in file_list:
            if file[-4:] == '.pth':
                model_files.append(file)
                # append all the MSE values
                losses.append(float(file[-12:-4]))
                    
    losses = np.array(losses)
    model_files = np.array(model_files)

    # find best model by looking at the smallest loss 
    best_model = model_files[np.argmin(losses)]
    path_best_model = os.path.join(path_trained_model, best_model)
    print(f'Path to best offline model: {path_best_model} \n')

    # load model weights
    model.load_state_dict(torch.load(path_best_model))  
    
model.to(device)
print(model)
print('\n')
# %%
# RE-TRAIN MODELS ONLINE

# train and validate or only validate the model
dice_metric_videos, hd_max_metric_videos, hd_95_metric_videos, \
hd_50_metric_videos, rmse_SI_metric_videos, rmse_AP_metric_videos, \
loss_train_online_epochs, tot_t_online_videos = train_val_test.train_val_model_online(model, data_loader=data_loader,
                                                                                        input_data=input_data, loss_name=loss_name, 
                                                                                        wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o,
                                                                                        lr=lr, l2=l2, device=device,
                                                                                        min_train_data_length=min_train_data_length,
                                                                                        centroid_norm=centroid_norm,
                                                                                        online_epochs=online_epochs)    

# %%
# SAVE STATS AND PLOTS
other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
            f'-step_size={1}-breathhold_inclusion={breathhold_inclusion}-cohort={cohort}' + \
            f'-set={config.set}-online_epochs={config.online_epochs}-input_data={input_data}'      
                
# save some stats   
utils.save_stats_train_val_online(path_saving, net_params, other_params,
                                dice_metric_videos, hd_max_metric_videos, 
                                hd_95_metric_videos, hd_50_metric_videos,
                                rmse_SI_metric_videos, rmse_AP_metric_videos,
                                tot_t_online_videos, set=config.set)

# plot and save resulting online losses
if online_epochs is not None:
    plotting.losses_plot(train_losses=loss_train_online_epochs,
                            loss_name=loss_name,
                            display=True, last_epochs=None, 
                            path_saving=os.path.join(path_saving, 'online_losses_last_wdw.png'))


# plot ground truth vs predicted for example data (make sure there is some data first)
if np.shape(example_inputs)[0] > 1:
    example_inputs, example_outputs = example_inputs[0, None].to(device), example_outputs[0, None].to(device)
    if net == 'LSTM_SI_AP':
        # normalize centroid curves with corresponding amplitudes
        if centroid_norm == 'wdw_norm':
            max_amplitude_SI, min_amplitude_SI = torch.max(example_inputs[:,:,0]).item(), torch.min(example_inputs[:,:,0]).item()
            max_amplitude_AP, min_amplitude_AP = torch.max(example_inputs[:,:,1]).item(), torch.min(example_inputs[:,:,1]).item()
            
        example_inputs[:,:,0] = utils.normalize(example_inputs[:,:,0], 
                                        {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
        example_inputs[:,:,1] = utils.normalize(example_inputs[:,:,1], 
                                        {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
        example_outputs[:,:,0] = utils.normalize(example_outputs[:,:,0], 
                                        {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
        example_outputs[:,:,1] = utils.normalize(example_outputs[:,:,1], 
                                        {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
                        
        example_predictions = model(example_inputs)
        plotting.predicted_wdw_plot(example_inputs[:,:,0], example_outputs[:,:,0], example_predictions[:,:,0], wdw_nr=-1, last_pred=False,
                            display=True, path_saving=os.path.join(path_saving, f'predicted_SI_wdw_{-1}.png'))
        plotting.predicted_wdw_plot(example_inputs[:,:,1], example_outputs[:,:,1], example_predictions[:,:,1], wdw_nr=-1, last_pred=False,
                            display=True, path_saving=os.path.join(path_saving, f'predicted_AP_wdw_{-1}.png'))
        
        example_predictions_seg = utils.shift_by_centroid_diff(predictions=example_predictions, 
                                                            inputs=example_inputs, inputs_seg=example_inputs_seg,
                                                            min_amplitude_SI=min_amplitude_SI, max_amplitude_SI=max_amplitude_SI, 
                                                            min_amplitude_AP=min_amplitude_AP, max_amplitude_AP=max_amplitude_AP)
        plotting.in_out_pred_frames_plot_vertical(example_inputs_seg, example_outputs_seg, example_predictions_seg, 
                                                    nr_frames=wdw_size_o, display=True, 
                                                    path_saving=os.path.join(path_saving, 'in_out_pred_predictions.png'))
        example_segmentations = (example_predictions_seg > 0.5) * 1.0
        plotting.in_out_pred_frames_plot_vertical(example_inputs_seg, example_outputs_seg, example_segmentations, 
                                                    nr_frames=wdw_size_o, display=True,
                                                    path_saving=os.path.join(path_saving, 'in_out_pred_segmentations.png'))
        
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
