"""
Configuration file for main scripts
"""

#%%
import os
import matplotlib
import torch
import time
import sys

# set path to project folder 
# (to be adapted to your project download location or
# to path inside docker container where project folder is mounted)
path_project = '/home' 

# path to data folder
path_project_data = os.path.join(path_project, 'data')

# path to code folder
path_project_code = os.path.join(path_project, 'code')

# path to results folder
path_project_results = os.path.join(path_project, 'results')

# add project auxiliary folder to your Python path to be able to import self written modules from anywhere
sys.path.append(os.path.join(path_project_code, 'auxiliary'))

# importing self written module from auxiliary folder 
from auxiliary import utils, data_preparation

#%%
# SET GENERAL SETTINGS 

# GPU settings
gpu_usage = True
dev_nr = 0  # 0,1 
if gpu_usage:
    if torch.cuda.is_available():  
        device = torch.device(f'cuda:{dev_nr}') 
        # set device nr to standard GPU
        torch.cuda.set_device(dev_nr)   
    else:  
        device = torch.device('cpu') 
else:
    device = torch.device('cpu')
    
# print some environment info
print_info_env = True
if print_info_env:
    print('--------------------------------------------------')
    print('Python version:', sys.version)
    print('PyTorch version:', torch.__version__)
    print('CUDNN version:', torch.backends.cudnn.version())
    print('Available devices: ', torch.cuda.device_count())
    if gpu_usage:
        print('Current cuda device: ', torch.cuda.current_device())
        print('Name of device used: ', torch.cuda.get_device_name(dev_nr))
    else:
        print('Running on CPU.')
    print('--------------------------------------------------')
    
# whether to run without display
no_display = False
if no_display:
    # to run on cluster with no display --> needs to be deactivated to display images
    matplotlib.use('agg')

#%%
# SET PARAMETERS FOR SCRIPTS

# time at which run is started, used to save results
start_time_string = time.strftime("%Y-%m-%d-%H:%M:%S")  

# whether to run data preprocessing or optimization/inference main script
code = 'predict'  # 'preprocess' or 'predict'

if code == 'preprocess':
    # relative path to folder with original cases to be preprocessed
    # cohort_original = '2022_05_23_test2'
    # cohort_original = '2022_05_25_test_post_upgrade'
    # cohort_original = '2021_06_16_respiratory_patients_LMU_ogv'
    cohort_original = '2021_10_25_free_breathing_patients_co60_Gemelli_ogv'
    # relative path to folder name were preprocessed cases will be copied
    # cohort_preprocessed = '2022_05_23_test2'
    # cohort_preprocessed = '2022_05_25_test_post_upgrade'
    # cohort_preprocessed = '2021_06_16_respiratory_patients_LMU_ogv' 
    cohort_preprocessed = '2021_10_25_free_breathing_patients_co60_Gemelli_ogv'
    # short run with only a few frames for filling process
    short_run = False  
    # first watershed for all frames without further processing --> which might or might not work
    full_ws_threshold = 0.1 
    # watershed for the frames which failed for the first attempt
    ws_threshold = 0.02
    # length of slinding window for outlier detection and moving average filter
    wdw_size = 3
    # number of frames which can be unfilled/with small successive baseline DSC before case is excluded
    frame_nr_exclusion_threshold = 8
    # threshold for IQR of motion considered small in mm
    iqr_motion_threshold = 0
    # threshold for IQR of motion considered small in mm
    successive_dsc_threshold = 0.4
    # if not None, output spacing of resampled frames relative to identity
    out_spacing = (1.0, 2.7, 2.7)
    # whether to display the plots, works only when code is run in IPython
    display_plots = False  
    # whether to save the plots
    save_plots = True 
    # version of cine video (pre smart vision upgrade or post upgrade)
    version = 'pre-upgrade' # pre-upgrade, post-upgrade
    
    
elif code == 'predict': 
    # input data type
    input_data = 'segmentations'  # centroids, segmentations, frames, segmentations_and_frames
    # motion direction of centroid to predict
    direction = 'SI_AP'  # 'SI' 'SI_AP'
    # centroid normalization scheme
    centroid_norm = 'wdw_norm'
    # whether to input motion with or without breathholds
    breathhold_inclusion = False
    # total duration of set of sliding windows for online training
    min_train_data_length = 80  # 80 corresponds to 20 seconds 
    # length of input and output sequences
    wdw_size_i = 32  # 8, 16, 24, 32 --> 2, 4, 6, 8 seconds
    wdw_size_o = 2  # 1, 2, 3 --> 250, 500, 750 ms
 
    # print some  info
    print_info = True
    if print_info:
        print(f'Input data type: {input_data}')
        print(f'Motion direction: {direction}')
        print(f'Breath-hold data inclusion: {breathhold_inclusion}')
        print(f'Input window size: {wdw_size_i}')
        print(f'Output window size: {wdw_size_o}')        

    # ------------- network and optimization parameters ----------------
    net = 'ConvLSTM'  # no_predictor, LSTM_SI, LSTM_SI_AP, ConvLSTM, ConvLSTM_STL
    print(f'Selected model: {net}')
    print('\n') 

    if net == 'LSTM_SI':
        input_features = 1
        hidden_features = 15
        output_features = 1
        num_layers = 5
        dropout = 0.0      
        l2 = 0
        bi = False
        loss_name = 'MSE'  # MSE
        lr = 1e-4
        batch_size = 3   
    elif net == 'LSTM_SI_AP':
        input_features = 2
        hidden_features = 15
        output_features = 2
        num_layers = 5
        dropout = 0.0      
        # l2 = 0   # offline
        l2 = 1e-6   # online
        bi = False
        loss_name = 'MSE'  # MSE
        # lr = 5e-4  # offline
        lr = 1e-6 # online
        batch_size = 128  
    elif net == 'ConvLSTM':
        dec_input_init_zeros = False
        loss_name = 'dice'  # dice, focal, dice_and_centroid
        l2 = 0   
        lr = 1e-4
        augmentations = True
        batch_size = 32
    elif net == 'ConvLSTM_STL':
        max_displacement = 0.3
        dec_input_init_zeros = False
        loss_name = 'dice'  # dice, dice_and_gradient, focal, dice_and_centroid
        l2 = 0 
        lr = 1e-4
        augmentations = True
        batch_size = 3
    elif net == 'no_predictor':
        # no parameters
        pass
    else:
        raise Exception('Unknown net specified!')          

    
    # if False, train on a dataset and then validate on a separate dataset. --> main_train_val_model.py
    # If True, continuosly re-train and/or validate on data of a specific case --> main_online_train_val_model.py
    online_training = False   
    
    if online_training is False: 
        # max_epochs = 500*5  # (500 LSTM with BH, without BH x 5)
        max_epochs = 50*5  # (50 ConvLSTM with BH, without BH x 5)
        # max_epochs = 5*5  # (5 ConvLSTM-STL with BH, without BH x 5) 
        # early_stopping_patience = 200*5  # 200
        early_stopping_patience = 20*5  # 20
        # early_stopping_patience = 2*5  # 2
        
        # path to folder with offline training/validation data
        path_training_data = os.path.join(path_project_data, 'preprocessed', 'offline_training_data')
        path_validation_data = os.path.join(path_project_data, 'preprocessed', 'offline_validation_data')
        
    else:
        # path to offline trained model to be used for online training
        # path_trained_model = None  # online validation/testing without pre trained model
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-11-12:28:43')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-11-05:42:07')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-18-15:36:56')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-19-09:50:50')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-19-09:51:50')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-19-13:40:53')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-22-15:17:05')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-24-14:20:01')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-21-12:26:07')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-25-11:23:38')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-25-09:56:45')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-25-15:27:19')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-26-11:48:24')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-26-11:48:24')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-30-13:47:35')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-28-06:33:23')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-07-31-08:13:24')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-07-11:44:57')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-07-11:48:01')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-04-14:11:39')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-07-11:26:32')
        path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-10-11:00:00')
        # path_trained_model = os.path.join(path_project_results, net, 'optimization' , '2022-08-10-11:48:40')
        
        # nr of online training epochs. If None, no online training performed 
        # -> validation of offline model on a cine video basis
        online_epochs = None  # None, 30
        
        # cohort folder name 
        cohort = '2021_06_16_respiratory_patients_LMU_ogv' 
        # cohort = '2021_10_25_free_breathing_patients_co60_Gemelli_ogv'
        
        # path to folder with case data for online training
        path_data = os.path.join(path_project_data, 'preprocessed', cohort)
        
        # data set used for inference in online training script.
        # 'train', 'val' or 'test' 
        # can be used for sanity checks or real validation/testing.
        # example: you have an offline trained model but want to check its perfomance 
        # again on the validation set but on a case basis --> select set = 'val'
        set = 'val' 
        print(f'Dataset used: {set} - {cohort}')
        
        # select trainig, validation and testing patients according to data split decided for 
        # offline data set in data_preparation.py
        if set == 'train':
            path_cases = utils.subdir_paths(path_data)[:data_preparation.train_cases]
        elif set == 'val':
            path_cases = utils.subdir_paths(path_data)[data_preparation.train_cases:data_preparation.train_val_cases]
        elif (set == 'test') and ('LMU' in cohort):
            path_cases = utils.subdir_paths(path_data)[data_preparation.train_val_cases:]
        elif (set == 'test') and ('Gemelli' in cohort):
            breathhold_inclusion = True
            path_cases = utils.subdir_paths(path_data)
        else:
            raise Exception('Unknown set specified!')   
              
else:
    raise Exception('Unknonw code specified to be run!')
                
            
#%%
