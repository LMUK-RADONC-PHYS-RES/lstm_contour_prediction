"""
Created on May 25 2021

@author: Elia Lombardo

Main script to test model and if needed continuosly train model on single traces for centroid position prediction
"""
# %%
# IMPORT MODULES

# self written modules & parameters from config
import config

from config import device
from config import path_cases
# path_cases = path_cases[10:] # small test
from config import net
if net != 'no_predictor':
    raise ValueError('Attention: select "no_predictor" as net in config.py to run this script!')
from config import wdw_size_i, wdw_size_o, min_train_data_length
from config import direction, breathhold_inclusion, cohort
from auxiliary import utils
from auxiliary import train_val_test, data_preparation

# %%
import os
import torch
torch.cuda.empty_cache()
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import Compose, ToTensor

# %%
# CREATE FOLDER FOR RESULTS
   
path_saving = os.path.join(config.path_project_results, net, 'inference', config.start_time_string) 
os.makedirs(path_saving, exist_ok=True)

# %%
# DEFINE TRANSFORMS AND GET DATA

# transforms: 1st composition is for centroids, 2nd for segmentations
data_transforms = (Compose([ToTensor(dtype=torch.float32)]),
                    Compose([ToTensor(dtype=torch.float32)]))

# for no predictor use segmentations directly
input_data = 'segmentations'
# online training/validation/testing data
data_set = data_preparation.OnlineSequentialDataset(path_data=path_cases, 
                                input_data=input_data, direction=direction, 
                                breathhold_inclusion=breathhold_inclusion, 
                                wdw_size_i=wdw_size_i, wdw_size_o=wdw_size_o, 
                                step_size=1)
print(f'Number of data samples: {data_set.__len__()}')
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,
                                            shuffle=False, num_workers=0, 
                                            drop_last=False,
                                            collate_fn=data_preparation.collate_correct_dim)
  
#%%
# PLOT EXAMPLE DATA
iterator = iter(data_loader)
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

#%%
# MODEL EVALUATION

# evaluate no predictor
dice_metric_videos, hd_max_metric_videos, hd_95_metric_videos, hd_50_metric_videos, \
rmse_SI_metric_videos, rmse_AP_metric_videos, _, _ = train_val_test.evaluate_no_predictor(data_loader,
                                                                                            wdw_size_i, wdw_size_o,
                                                                                            min_train_data_length,
                                                                                            device=device)

# %%
# SAVE STATS 

net_params = 'using last input segmentation as output'
other_params = f'device={device}-wdw_size_i={wdw_size_i}-wdw_size_o={wdw_size_o}' + \
            f'-step_size={1}-breathhold_inclusion={breathhold_inclusion}' + \
            f'-cohort={cohort}-set={config.set}'
    
                
# save some stats   
utils.save_stats_train_val_online(path_saving, net_params, other_params,
                                dice_metric_videos, hd_max_metric_videos, 
                                hd_95_metric_videos, hd_50_metric_videos,
                                rmse_SI_metric_videos, rmse_AP_metric_videos,
                                set=config.set)