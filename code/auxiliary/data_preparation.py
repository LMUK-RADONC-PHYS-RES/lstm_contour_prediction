"""
Functions for data loading
"""
# %%
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from monai.transforms import Compose, NormalizeIntensity, ToTensor


# import self written modules
import utils

# %%


def sliding_wdws(data, wdw_size_i, wdw_size_o, step_size=1):
    """ Given a sequence of input data, subdivide it in input and output windows.
    Args:
        data: list with input data
        wdw_size_i: length of generated window to be used as input
        wdw_size_o: length of generated window to be used as output
        step_size: number of data points the window rolls at each step
    """
    x = []
    y = []

    # loop over the full sequence 
    for i in range(len(data) - wdw_size_i - wdw_size_o - 1):
        # select input and output windows 
        _x = data[step_size * i:(step_size * i + wdw_size_i)]
        _y = data[step_size * i + wdw_size_i:step_size * i + wdw_size_i + wdw_size_o]

        # keep the windows only if both input and output have expected size
        if len(_x) == wdw_size_i and len(_y) == wdw_size_o:
            x.append(_x)
            y.append(_y)

    return x, y


def sliding_wdws_vectorized(data, wdw_size_i=8, wdw_size_o=2, step_size=1):
    """ Given a sequence of input data, subdivide it in input and output windows (vectorized operations).
    Adapted from: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.
    Args:
        data: list with input data
        wdw_size_i: length of generated window to be used as input
        wdw_size_o: length of generated window to be used as output
        step_size: number of data points the window rolls at each step
    """
    start = 0
    stop = data.shape[0]

    # find indices of all possible input windows using vectorized operations
    idx_windows_i = (start + 
        # create list with indices for first window, e.g. [0,1,2,3,4,5]
        np.expand_dims(np.arange(wdw_size_i), 0) +
        # create a column vector [0, step, 2*step, ...] to be added to first window list
        np.expand_dims(np.arange(stop - wdw_size_i - wdw_size_o + 1, step=step_size), 0).T)

    # find indices of output windows by taking for every row of the index_window_i matrix 
    # the last window_size_o nr of elements and add on top the window_size_o
    idx_windows_o = idx_windows_i[:, -wdw_size_o:] + wdw_size_o
    # print(f'shape of idx_window_i: {np.shape(idx_windows_i)}') # (nr_wdws, wdw_size_i)
    # print(data[idx_windows_i]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    # return input and ouput data windows
    return data[idx_windows_i], data[idx_windows_o]


def get_wdws(data_snippets, wdw_size_i=8, wdw_size_o=2, step_size=1):
    """ Obtain a list with all sliding input and output windows from list with all data snippets. """
    input_wdw_list = []
    output_wdw_list = []

    for snippet in (data_snippets):
        input_wdws, output_wdws = sliding_wdws_vectorized(data=snippet, wdw_size_i=wdw_size_i, 
                                                          wdw_size_o=wdw_size_o, step_size=step_size)
        # print(output_wdws)

        # check if nr of input and output wdws is the same
        if len(input_wdws) != len(output_wdws):
            raise Exception("Attention! Nr of input and output windows unequal.")
        else:
            # append all windows for a given snippet in the list which will contain the windows for all snippets
            input_wdw_list.append(input_wdws)
            output_wdw_list.append(output_wdws)

    return input_wdw_list, output_wdw_list


def replace_outliers(array, m_out=10.):
    """ Replace outliers in data window using the median of the distances from the median.
    Args:
        array: np.array with input data window
        m_out: multiple of the median of the distances from the median above which to replace the corrresponding value (outlier)
    """
    # compute distances from median
    d = np.abs(array - np.median(array))
    # print(d)

    # compute median of distances from median
    mdev = np.median(d)
    # print(mdev)

    # scale the distances with mdev
    s = d / (mdev if mdev else 1.)
    # print(s)
    
    # return data where outliers are replaced with median
    # print(np.where(s > m))
    array[np.where(s > m_out)] = np.median(array)

    return array 


def outlier_replacement_vectorized(array, seq_size=6, m_out=7):
    """ Given a sequence of data, subdivide it in windows (vecotrized operation) and then slide over them without overlapping to replace outliers without creating overlapping data.
    Adapted from: https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5.
    Args:
        array: input data
        seq_size: size of sliding window and at the same time step size during sliding
        m: multiple of the median of the distances from the median above which to replace the corrresponding value
    """
    start = 0
    stop = len(array)
    step_size = seq_size  # need to be the same to avoid replicating data

    # find indices of all possible windows using vectorized operations
    idx_windows = (start + 
        np.expand_dims(np.arange(seq_size), 0) +
        # Create a rightmost vector as [0, step, 2*step, ...].
        np.expand_dims(np.arange(stop - seq_size + 1, step=step_size), 0).T)

    # print(array[idx_windows]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    array_replaced = []
    # loop over all windows
    for seq in array[idx_windows]:
        # replace outliers for current window
        seq_replaced = replace_outliers(seq=seq, m_out=m_out)
        # extend current list by new (outlier replaced) window
        array_replaced.extend(seq_replaced)
    
    return array_replaced


def rolling_outlier_replacement(array, wdw_size=3, threshold_out=0.1):
    """ Replace outlieres based on deviation from median amplitude within sliding window using pandas rolling window.
    Adapted from: https://stackoverflow.com/questions/62692771/outlier-detection-based-on-the-moving-mean-in-python
    Args:
        array: list with input data as numpy array
        wdw_size: size of rolling wdw, with step size being automatically = 1 (pandas)     
        threshold_out: threshold amplitude above which to replace the corrresponding 
                        value (outlier) with the mdian within current window
    """
    # get normalized data to be used to find indices of data to be replaced -->
    # needed as threshold works best on all data if the amplitudes are comparable (i.e. normalized)
    array_norm = utils.normalize(array, {'actual': {'lower': np.min(array), 'upper': np.max(array)}, 
                                        'desired': {'lower': -1, 'upper': 1}})
    
    # put data into pandas dataframe
    df = pd.DataFrame({'data': array})
    df_norm = pd.DataFrame({'data': array_norm})

    # calculate rolling median for current element based on previous wdw_size nr of elements, 
    # else output NaN
    df['rolling_medians'] = df['data'].rolling(window=wdw_size, center=True).median()
    df_norm['rolling_medians'] = df_norm['data'].rolling(window=wdw_size, center=True).median()
    # print(f"df rolling medians: {df['rolling_medians']}")

    # calculate difference on normalized data
    df_norm['diff'] = df_norm['data'] - df_norm['rolling_medians']

    # find indices of values to be replaced with median value of window for normalized data
    diff = np.abs(df_norm['diff'].to_numpy())
    # print(diff)
    replace_flag = np.where(diff > threshold_out)
    # print(f'replace flag: {replace_flag}')

    replaced_array = np.copy(array)
    # return data where outliers are replaced with median of window
    replaced_array[replace_flag] = df['rolling_medians'].to_numpy()[replace_flag]

    return replaced_array  


def get_snippets(data, pauses_start, bhs, bhs_start, min_length_snippet=8):
    """ Separate  full data sequence into snippets (sub-sequences) according to image pauses and breathholds.
    Args:
        data: input data sequence where sequence dimension must be axis=0
        pauses_start: list with 1 for image pause start
        bhs: list with 1 for breath-holds
        bhs_start: list with 1 for breath-hold start
        min_length_snippet: int, minimum length of snippets which are not discarded
    """
    # check if data is np.array and if not convert to it
    if isinstance(data,(np.ndarray)):
        pass
    else:
        data = np.array(data)
    # convert lists with spliiting info to np.arrays
    pauses_start = np.array(pauses_start)
    bhs = np.array(bhs)
    bhs_start = np.array(bhs_start)
        
    # 1. data separated according only to pauses (breath-holds included)
    data_snippets_with_bh = np.split(data, np.where(pauses_start == 1)[0] + 1)


    # 2. data separated according to pauses and breath-holds start (breath-holds excluded)
    data_snippets_pause_bh = np.split(data, np.where((pauses_start == 1) | (bhs_start == 1))[0] + 1)

    # split also breath hold information according to image puases / bh start
    snippets_breathholds = np.split(bhs, np.where((pauses_start == 1) | (bhs_start == 1))[0] + 1)
    # print(np.shape(snippets_breathholds))

    # delete frames which were flagged as breathhold by taking into account snippet structure
    data_snippets_without_bh = []
    for snippet_nr in range(len(data_snippets_pause_bh)):
        current_data_snippet = data_snippets_pause_bh[snippet_nr]
        current_bhs_snippet = snippets_breathholds[snippet_nr]
        # delete breahholds
        current_snippet = np.delete(current_data_snippet, np.where(current_bhs_snippet == 1), axis=0)
        if len(current_snippet) > min_length_snippet:
            data_snippets_without_bh.append(current_snippet)

    return data_snippets_with_bh, data_snippets_without_bh


def generate_offline_data_folder(path_data_in, path_data_out, direction='SI_AP', 
                                    breathhold_inclusion=True, wdw_size_i=32, wdw_size_o=2, 
                                    step_size=1, centroid_norm='wdw_norm'):
        """Generate input and output windows to be saved to disk for offline training and validation.
        The advantage of this approach consists in the fact that a Pytorch Dataset can be defined 
        () to load data with a batch size bigger than 1 while using multi-processing in the DataLoader. 

        Args:
            path_data_in (str): path where data is stored on a case basis
            path_data_out (str): path where data will be stored on a input/output windows basis
            direction (str, optional): direction of center of mass motion. Defaults to 'SI_AP'.
            breathhold_inclusion (bool, optional): whether to include the breathhold data or not. Defaults to True.
            wdw_size_i (int, optional): length of the input data sequence. Defaults to 32.
            wdw_size_o (int, optional): length of the output data sequence. Defaults to 2.
            step_size (int, optional): number of data point the windows roll at each step. Defaults to 1.
            centroid_norm (str, optional): how to normalize the centroid curves. Normalize based on max and min
                                            of entire video when setting 'video_norm' and based on single windows
                                            when setting 'wdw_norm'.
        """
        counter_wdw = 0
        
        # create folder where to store input and output windows of data as .npy
        path_centroids_out = os.path.join(path_data_out, 'centroids',
                                 f'breathhold_inclusion_{breathhold_inclusion}_' + 
                                 f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' +
                                 f'step_size_{step_size}_{centroid_norm}')
        os.makedirs(os.path.join(path_centroids_out, 'input_windows'), exist_ok=True)
        os.makedirs(os.path.join(path_centroids_out, 'output_windows'), exist_ok=True)
        path_segmentations_out = os.path.join(path_data_out, 'segmentations',
                                 f'breathhold_inclusion_{breathhold_inclusion}_' + 
                                 f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_step_size_{step_size}')
        os.makedirs(os.path.join(path_segmentations_out, 'input_windows'), exist_ok=True)
        os.makedirs(os.path.join(path_segmentations_out, 'output_windows'), exist_ok=True)
        path_frames_out = os.path.join(path_data_out, 'frames',
                                 f'breathhold_inclusion_{breathhold_inclusion}_' + 
                                 f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_step_size_{step_size}')
        os.makedirs(os.path.join(path_frames_out, 'input_windows'), exist_ok=True)
        os.makedirs(os.path.join(path_frames_out, 'output_windows'), exist_ok=True)
        
        
        # loop over all cases of preprocessed data
        for path_case in path_data_in:
            # check if folder with centroids and info is empty
            if len(os.listdir(os.path.join(path_case, 'centroids'))) == 0:
                # go to next case
                continue
            else:
                # loop over all info files of one case
                for _, _, file_list in os.walk(os.path.join(path_case, 'centroids')):
                    for file_name_centroid_info in file_list:
                        # get paths to data
                        path_centroid_info = os.path.join(path_case, 'centroids', file_name_centroid_info)
                        path_seg = os.path.join(path_case, 'segmentations', file_name_centroid_info[:-24] + 'segmentations.npy')
                        path_frame = os.path.join(path_case, 'frames', file_name_centroid_info[:-24] + 'frames.npy')

                        # load dataframe with info for current video
                        print(f'Loading data for: {path_centroid_info[:-5]}')
                        df = pd.read_excel(path_centroid_info)
                        # get info for splitting into snippets
                        pauses_start=df['Imaging paused start']
                        bhs=df['Breath-holds']
                        bhs_start=df['Breath-holds start']
                        
                        # get sequence of centroid positions
                        if direction == 'SI':
                            data = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                            if centroid_norm == 'video_norm':
                                # normalize data using video's min max to range -1 to +1
                                data = utils.normalize(data, 
                                                {'actual': {'lower': np.min(data), 'upper': np.max(data)}, 
                                                'desired': {'lower': -1, 'upper': 1}})    
                            # add channel dim 
                            data_centroids = data[:, None]            
                        elif direction == 'SI_AP':
                            data_SI = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                            if centroid_norm == 'video_norm':
                                # normalize data using video's min max to range -1 to +1
                                data_SI = utils.normalize(data_SI, 
                                                {'actual': {'lower': np.min(data_SI), 'upper': np.max(data_SI)}, 
                                                'desired': {'lower': -1, 'upper': 1}})
                            data_AP = np.array(df['Target COM post-ant (after smoothing) [mm]'].values)
                            if centroid_norm == 'video_norm':
                                # normalize data using video's min max to range -1 to +1
                                data_AP = utils.normalize(data_AP, 
                                                {'actual': {'lower': np.min(data_AP), 'upper': np.max(data_AP)}, 
                                                'desired': {'lower': -1, 'upper': 1}})
                            data_centroids = np.concatenate((data_SI[:, None], data_AP[:, None]), axis=1)
                            
                        else:
                            raise Exception('Unexpected direction specified!')

                        # get sequence of binary segmentations for current video
                        data_segmentations = np.load(path_seg)
                        # print(f'Shape of segmentations after loading: {data.shape}')  # (1453, 512, 512)                           
                        
                        # get sequence of cine frames for current video
                        data_frames = np.load(path_frame)
                        # print(f'Shape of frames after loading: {data.shape}')  # (1453, 512, 512)
                        

                        # separate data into list of snippets with shape=(nr_snippets,) 
                        # according to image pauses and bhs
                        snippets_with_bh_centroids, snippets_without_bh_centroids = get_snippets(data=data_centroids, 
                                                                            pauses_start=pauses_start, 
                                                                            bhs=bhs, 
                                                                            bhs_start=bhs_start)
                        snippets_with_bh_segmentations, snippets_without_bh_segmentations = get_snippets(data=data_segmentations, 
                                                                            pauses_start=pauses_start, 
                                                                            bhs=bhs, 
                                                                            bhs_start=bhs_start)
                        snippets_with_bh_frames, snippets_without_bh_frames = get_snippets(data=data_frames, 
                                                                            pauses_start=pauses_start, 
                                                                            bhs=bhs, 
                                                                            bhs_start=bhs_start)
                                                                           
                        # include bhs in motion curves
                        if breathhold_inclusion:
                            # get data input and ouput windows
                            x_centroids, y_centroids = get_wdws(snippets_with_bh_centroids, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)
                            x_segmentations, y_segmentations = get_wdws(snippets_with_bh_segmentations, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)
                            x_frames, y_frames = get_wdws(snippets_with_bh_frames, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)
                        # exclude bhs from motion curves
                        else:
                            # get data input and ouput windows
                            x_centroids, y_centroids = get_wdws(snippets_without_bh_centroids, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)
                            x_segmentations, y_segmentations = get_wdws(snippets_without_bh_segmentations, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)
                            x_frames, y_frames = get_wdws(snippets_without_bh_frames, 
                                                wdw_size_i=wdw_size_i, 
                                                wdw_size_o=wdw_size_o, 
                                                step_size=step_size)

                        # concatenate all snippets in the data list to an array 
                        # with shape=(nr_wdws, wdw_size, (...)) and automatically drop empty items
                        x_centroids, y_centroids = np.concatenate(x_centroids, axis=0), np.concatenate(y_centroids, axis=0)
                        x_segmentations, y_segmentations = np.concatenate(x_segmentations, axis=0), np.concatenate(y_segmentations, axis=0)
                        x_frames, y_frames = np.concatenate(x_frames, axis=0), np.concatenate(y_frames, axis=0)

                # save each input and output window to disk separately
                for wdw_nr in range(x_centroids.shape[0]):

                    if centroid_norm == 'wdw_norm':
                        # normalize on an input sequence basis
                        # SI
                        min_x_SI = np.min(x_centroids[wdw_nr,:,0])
                        max_x_SI = np.max(x_centroids[wdw_nr,:,0])                        
                        x_centroids[wdw_nr,:,0] = utils.normalize(x_centroids[wdw_nr,:,0], 
                                        {'actual': {'lower': min_x_SI, 'upper': max_x_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}) 
                        y_centroids[wdw_nr,:,0] = utils.normalize(y_centroids[wdw_nr,:,0], 
                                        {'actual': {'lower': min_x_SI, 'upper': max_x_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}) 
                        # AP
                        min_x_AP = np.min(x_centroids[wdw_nr,:,1])
                        max_x_AP = np.max(x_centroids[wdw_nr,:,1])                              
                        x_centroids[wdw_nr,:,1] = utils.normalize(x_centroids[wdw_nr,:,1], 
                                        {'actual': {'lower': min_x_AP, 'upper': max_x_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}) 
                        y_centroids[wdw_nr,:,1] = utils.normalize(y_centroids[wdw_nr,:,1], 
                                        {'actual': {'lower': min_x_AP, 'upper': max_x_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}) 
                    np.save(os.path.join(path_centroids_out, 'input_windows', f'input_window_{counter_wdw:08d}.npy'), x_centroids[wdw_nr])
                    np.save(os.path.join(path_centroids_out, 'output_windows', f'output_window_{counter_wdw:08d}.npy'), y_centroids[wdw_nr])

                    np.save(os.path.join(path_segmentations_out, 'input_windows', f'input_window_{counter_wdw:08d}.npy'), x_segmentations[wdw_nr])
                    np.save(os.path.join(path_segmentations_out, 'output_windows', f'output_window_{counter_wdw:08d}.npy'), y_segmentations[wdw_nr])

                    np.save(os.path.join(path_frames_out, 'input_windows', f'input_window_{counter_wdw:08d}.npy'), x_frames[wdw_nr])
                    np.save(os.path.join(path_frames_out, 'output_windows', f'output_window_{counter_wdw:08d}.npy'), y_frames[wdw_nr])
                                        
                    counter_wdw += 1
                    
                    
class OfflineSequentialDataset(torch.utils.data.Dataset):
    """Pytorch Dataset to load sequential input data. During initialization of this class only the
    paths to all the input and output windows (pre-requisite: run generate_offline_data_folder() 
    function first this generate this data) are loaded while the actual data is
    loaded with the __getitem__ function (optimal in terms of memory consumption). 
    This Dataset can be used with Pytorch Dataloaders using any batchsize
    and multi-processing for offline training of LSTM models.

    Args:
        path_data (str): path to folder with case subfolders to input data.
        input_data (str, optional): input data variant: 'centroids', 'segmentations', 'frames'.
        direction (str, optional): direction of center of mass motion: 'SI', 'SI_AP'. 
        breathhold_inclusion (bool, optional): whether to include breathholds in the data or not. 
        wdw_size_i (int, optional): length of data sequence to be used as input. 
        wdw_size_o (int, optional): length of data sequence to be used as output. 
        step_size (int, optional): number of data points the window rolls at each step.
        centroid_norm (str, optional): normalization scheme for centroid motion curves.
        transforms (Pytorch transforms, optional): transforms which are applied to the data.
    """

    def __init__(self, path_data, input_data='centroids', direction='SI_AP', 
                       breathhold_inclusion=True, wdw_size_i=32, wdw_size_o=2, 
                       step_size=1, centroid_norm='wdw_norm', transforms=None):
        
        self.input_data = input_data
        self.direction = direction
        self.wdw_size_i = wdw_size_i
        self.wdw_size_o =  wdw_size_o
        self.transforms = transforms
        self.zero_centering_transform = Compose([NormalizeIntensity()])

        if input_data == 'centroids':
            self.path_to_wdw_i = os.path.join(path_data, input_data, 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}_{centroid_norm}',
                                                'input_windows')
            self.path_to_wdw_o = os.path.join(path_data, input_data, 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}_{centroid_norm}',
                                                'output_windows')
        elif input_data == 'segmentations_and_frames':
            self.path_to_wdw_i = os.path.join(path_data, 'segmentations', 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}',
                                                'input_windows')
            self.path_to_wdw_o = os.path.join(path_data, 'segmentations', 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}',
                                                'output_windows')           
            self.path_to_frame_wdw_i = os.path.join(path_data, 'frames', 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}',
                                                'input_windows')
        else:
            self.path_to_wdw_i = os.path.join(path_data, input_data, 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}',
                                                'input_windows')
            self.path_to_wdw_o = os.path.join(path_data, input_data, 
                                                f'breathhold_inclusion_{breathhold_inclusion}_' +
                                                f'wdw_size_i_{wdw_size_i}_wdw_size_o_{wdw_size_o}_' + 
                                                f'step_size_{step_size}',
                                                'output_windows')
            
        
        # initialize list to store paths to input and output windows
        self.paths_wdw_i = []
        self.paths_wdw_o = []
        
        # loop over all input window files
        for filename in sorted(os.listdir(os.path.join(self.path_to_wdw_i))):
            # append paths 
            self.paths_wdw_i.append(os.path.join(self.path_to_wdw_i, filename))

        # loop over all output window files
        for filename in sorted(os.listdir(os.path.join(self.path_to_wdw_o))):
            # append paths 
            self.paths_wdw_o.append(os.path.join(self.path_to_wdw_o, filename))
            
        # additionally load input frames
        if self.input_data == 'segmentations_and_frames':
            self.paths_frame_wdw_i = []
            # loop over all input window files
            for filename in sorted(os.listdir(os.path.join(self.path_to_frame_wdw_i))):
                # append paths 
                self.paths_frame_wdw_i.append(os.path.join(self.path_to_frame_wdw_i, filename))
            

    def __getitem__(self, idx):
        """Generates pair of input and output windows"""
        
        # get input and output window pair with shape (wdw_size, ...)
        # print(f'Loading follwoing window: {self.paths_wdw_i[idx]}')
        # print(f'Loading follwoing window: {self.paths_wdw_o[idx]}')
        input_wdw, output_wdw = np.load(self.paths_wdw_i[idx]), np.load(self.paths_wdw_o[idx])
        
        if self.input_data == 'segmentations_and_frames':
            input_frame_wdw = np.load(self.paths_frame_wdw_i[idx])
        
        # apply data augmentation on the fly
        if self.transforms:
            # to be able to apply the same transform on both input and output window we
            # concatenate the sequence dimension such that both input and output windows
            # are considered a single channel dimension
            concat_wdw = np.concatenate((input_wdw, output_wdw), axis=0)
            if self.input_data == 'segmentations_and_frames':
                # explicitly apply zero centering on frames using each frames mean and std
                input_frame_wdw = self.zero_centering_transform(input_frame_wdw)
                # then concant with segmentation to apply other transforms
                concat_wdw = np.concatenate((concat_wdw, input_frame_wdw), axis=0)               
            if self.input_data == 'frames':
                # apply zero centering transforms to all frames
                concat_wdw = self.zero_centering_transform(concat_wdw)  
            # apply specifed transforms
            concat_wdw = self.transforms(concat_wdw)
            input_wdw, output_wdw = concat_wdw[:self.wdw_size_i, ...], concat_wdw[self.wdw_size_i:self.wdw_size_i+self.wdw_size_o, ...]
            if self.input_data == 'segmentations_and_frames':
                input_frame_wdw = concat_wdw[self.wdw_size_i+self.wdw_size_o:, ...]

        # select single direction of motion for centroids
        if self.input_data == 'centroids':
            if self.direction == 'SI':
                input_wdw, output_wdw = input_wdw[:, 0, None], output_wdw[:, 0, None]
            else:
                pass
        # add channel dimension for segmentations and frames
        elif self.input_data == 'segmentations' or self.input_data == 'frames':
            input_wdw, output_wdw = input_wdw[:, None, ...], output_wdw[:, None, ...]
        elif self.input_data == 'segmentations_and_frames':
            input_seg_wdw, output_wdw, input_frame_wdw = input_wdw[:, None, ...], output_wdw[:, None, ...], input_frame_wdw[:, None, ...]
            # concatenate input seg and frame to make input dual-channel
            input_wdw = np.concatenate((input_seg_wdw, input_frame_wdw), axis=1)
        else:
            raise Exception('Unexpected input_data specified!')
                
        # print(f'Shape of input_wdw prior return: {input_wdw.shape}')  # torch.Size([32, 1, 256, 256]) for seg
        # print(f'Shape of output_wdw prior return: {output_wdw.shape}')  # torch.Size([2, 1, 256, 256]) for seg
        return input_wdw, output_wdw
    

    def __len__(self):
        """Denotes the total number of input/output windows"""
        return len(self.paths_wdw_i) 


class OnlineSequentialDataset(torch.utils.data.Dataset):
    """Pytorch Dataset to load sequential input data. During initialization of this class
    only the paths to the input and output windows for different cases are loaded while the data
    itself from the different cine videos is loaded in the __getitem__ function (multi-processing possible). 
    This Dataset can be used with Pytorch Dataloaders only with batchsize=1
    for online training of LSTM models.

    Args:
        path_data (str): path to folder with case subfolders to input data.
        input_data (str, optional): input data variant: 'centroids', 'segmentations', 'frames'.
        direction (str, optional): direction of center of mass motion: 'SI', 'SI_AP'. 
        breathhold_inclusion (bool, optional): whether to include breathholds in the data or not. 
        wdw_size_i (int, optional): length of data sequence to be used as input. 
        wdw_size_o (int, optional): length of data sequence to be used as output. 
        step_size (int, optional): number of data points the window rolls at each step.
        centroid_norm (str, opt): normalization scheme applied to centroid motion curves.
        transforms (Pytorch transforms, optional): tuple with transforms which are applied to  the centroid
                                                    and segmentation data. Must be deterministic!
    """
    
    def __init__(self, path_data, input_data='centroids', direction='SI', 
                       breathhold_inclusion=True, wdw_size_i=32, wdw_size_o=2, 
                       step_size=1, centroid_norm='wdw_nr'):


        self.input_data = input_data
        self.direction = direction
        self.breathhold_inclusion = breathhold_inclusion
        self.wdw_size_i = wdw_size_i
        self.wdw_size_o = wdw_size_o
        self.step_size = step_size
        self.centroid_norm = centroid_norm
        self.transforms = Compose([ToTensor(dtype=torch.float32)])
        self.zero_centering_transform = Compose([NormalizeIntensity()])

        
        # initialize list to store paths to different input data
        self.paths_centroid_info = []
        self.paths_seg = []
        self.paths_frame = []
        
        # getting all paths to data
        for path_case in path_data:
            # print(path_case)
            # check if folder with centroids and info is empty
            if len(os.listdir(os.path.join(path_case, 'centroids'))) == 0:
                # go to next case
                continue
            else:
                # loop over all info files of one case
                for _, _, file_list in os.walk(os.path.join(path_case, 'centroids')):
                    for file_name_centroid_info in file_list:
                        # append paths 
                        self.paths_centroid_info.append(os.path.join(path_case, 'centroids', file_name_centroid_info))
                        self.paths_seg.append(os.path.join(path_case, 'segmentations', file_name_centroid_info[:-24] + 'segmentations.npy'))
                        self.paths_frame.append(os.path.join(path_case, 'frames', file_name_centroid_info[:-24] + 'frames.npy'))


    def __getitem__(self, idx):
        """Generates input and output windows for one cine video"""
        
        # load dataframe with info for current video
        print(f'Loading {self.input_data} for: {self.paths_centroid_info[idx][:-5]}')
        df = pd.read_excel(self.paths_centroid_info[idx])
        # get info for splitting into snippets
        pauses_start=df['Imaging paused start']
        bhs=df['Breath-holds']
        bhs_start=df['Breath-holds start']
        # amplitudes later needed to undo normalization (video_norm)
        max_amplitude_AP, min_amplitude_AP = df['Max/min amplitude post-ant (after smoothing) [mm]'].values[0], df['Max/min amplitude post-ant (after smoothing) [mm]'].values[1]
        max_amplitude_SI, min_amplitude_SI = df['Max/min amplitude inf-sup (after smoothing) [mm]'].values[0], df['Max/min amplitude inf-sup (after smoothing) [mm]'].values[1]
        
        # get sequence of centroid positions
        if self.input_data == 'centroids':
            if self.direction == 'SI':
                data = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                if self.centroid_norm == 'video_norm':
                    # normalize data using video's min max to range -1 to +1
                    data = utils.normalize(data, 
                                    {'actual': {'lower': np.min(data), 'upper': np.max(data)}, 
                                    'desired': {'lower': -1, 'upper': 1}})    
                # add channel dim 
                data = data[:, None]            
            elif self.direction == 'SI_AP':
                data_SI = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                if self.centroid_norm == 'video_norm':                
                    # normalize data using video's min max to range -1 to +1
                    data_SI = utils.normalize(data_SI, 
                                    {'actual': {'lower': np.min(data_SI), 'upper': np.max(data_SI)}, 
                                    'desired': {'lower': -1, 'upper': 1}})
                data_AP = np.array(df['Target COM post-ant (after smoothing) [mm]'].values)
                if self.centroid_norm == 'video_norm':
                    # normalize data using video's min max to range -1 to +1
                    data_AP = utils.normalize(data_AP, 
                                    {'actual': {'lower': np.min(data_AP), 'upper': np.max(data_AP)}, 
                                    'desired': {'lower': -1, 'upper': 1}})
                data = np.concatenate((data_SI[:, None], data_AP[:, None]), axis=1)
                
            else:
                raise Exception('Unexpected direction specified!')
            # additonally get segmentations as they will be needed to shift contourd by difference of predicted and input centroids
            data_seg = np.load(self.paths_seg[idx])
            # add channel dim
            data_seg = data_seg[:,None,...]

        # get sequence of binary segmentations for current video
        elif self.input_data == 'segmentations':
            data = np.load(self.paths_seg[idx])
            # add channel dim
            data = data[:,None,...]
            # print(f'Shape of segmentations after loading: {data.shape}')
            
        # get sequence of cine frames for current video
        elif self.input_data == 'frames':
            data = np.load(self.paths_frame[idx])
            # add channel dim
            data = data[:,None,...]
            # print(f'Shape of frames after loading: {data.shape}')
            
        elif self.input_data == 'segmentations_and_frames':
            segs = np.load(self.paths_seg[idx])
            frames = np.load(self.paths_frame[idx])
            # add channel dim and concatenate along that dim
            data = np.concatenate((segs[:,None,...], frames[:,None,...]), axis=1)
            
        else:
            raise Exception('Unknown input_data specified!')


        # separate data into list of snippets with shape=(nr_snippets,) 
        # according to image pauses and bhs
        snippets_with_bh, snippets_without_bh = get_snippets(data=data, 
                                                            pauses_start=pauses_start, 
                                                            bhs=bhs, 
                                                            bhs_start=bhs_start)
                                      
        # include bhs in motion curves
        if self.breathhold_inclusion:
            # get data input and ouput windows
            x, y = get_wdws(snippets_with_bh, 
                                wdw_size_i=self.wdw_size_i, 
                                wdw_size_o=self.wdw_size_o, 
                                step_size=self.step_size)
        # exclude bhs from motion curves
        else:
            # get data input and ouput windows
            x, y = get_wdws(snippets_without_bh, 
                                wdw_size_i=self.wdw_size_i, 
                                wdw_size_o=self.wdw_size_o, 
                                step_size=self.step_size) 

        # concatenate all snippets in the data list to an array 
        # with shape=(nr_wdws, wdw_size, (...)) and automatically drop empty items
        x, y = np.concatenate(x, axis=0), np.concatenate(y, axis=0)
        # be aware that nr_wdws will be different for each video so
        # this Dataset must be used with a DataLoader using batch_size=1
            
        if self.input_data == 'segmentations' or self.input_data == 'frames' or self.input_data == 'segmentations_and_frames':
            if self.input_data == 'segmentations_and_frames':
                # get seg an frame
                x_seg = x[:,:,0,None,...]
                x_frame = x[:,:,1,None,...]
                # apply zero centering tranform on frames
                x_frame = self.zero_centering_transform(x_frame)
                # concatenate back
                x = np.concatenate((x_seg, x_frame), axis=2)
                # get only the segmentations as output
                y = y[:,:,0,None,...]
            if self.input_data == 'frames':
                # apply zero centering tranform on frames
                x, y = self.zero_centering_transform(x), self.zero_centering_transform(y) 
                # pass       
                
            # apply ToTensor deterministic transform
            x, y = self.transforms(x), self.transforms(y)
            
            # print(f'Shape of x prior return: {x.shape}')  # torch.Size([58, 16, 1, 256, 256]) 
            # print(f'Shape of y prior return: {y.shape}')  # torch.Size([58, 2, 1, 256, 256]) 
            return x, y
        
        # get input and output sequence also for segmentation when training on centroids
        else:
            # separate data into list of snippets with shape=(nr_snippets,) 
            # according to image pauses and bhs
            snippets_with_bh_seg, snippets_without_bh_seg = get_snippets(data=data_seg, 
                                                                pauses_start=pauses_start, 
                                                                bhs=bhs, 
                                                                bhs_start=bhs_start)
                                        
            # include bhs in motion curves
            if self.breathhold_inclusion:
                # get data input and ouput windows
                x_seg, y_seg = get_wdws(snippets_with_bh_seg, 
                                    wdw_size_i=self.wdw_size_i, 
                                    wdw_size_o=self.wdw_size_o, 
                                    step_size=self.step_size)
            # exclude bhs from motion curves
            else:
                # get data input and ouput windows
                x_seg, y_seg = get_wdws(snippets_without_bh_seg, 
                                    wdw_size_i=self.wdw_size_i, 
                                    wdw_size_o=self.wdw_size_o, 
                                    step_size=self.step_size) 

            # concatenate all snippets in the data list to an array 
            # with shape=(nr_wdws, wdw_size, (...)) and automatically drop empty items
            x_seg, y_seg = np.concatenate(x_seg, axis=0), np.concatenate(y_seg, axis=0)
            # be aware that nr_wdws will be different for each video so
            # this Dataset must be used with a DataLoader using batch_size=1

            # apply ToTensor deterministic transform
            x, y = self.transforms(x), self.transforms(y)  
            x_seg, y_seg = self.transforms(x_seg), self.transforms(y_seg)     

            # print(f'Shape of x prior return: {x.shape}')  # torch.Size([58, 16, 2]) 
            # print(f'Shape of y prior return: {y.shape}')  # torch.Size([58, 2, 2])             
            return x, y, x_seg, y_seg, max_amplitude_AP, min_amplitude_AP, max_amplitude_SI, min_amplitude_SI

    def __len__(self):
        """Denotes the total number cine videos"""
        return len(self.paths_centroid_info)


def collate_correct_dim(batch):
    """Correct dimensionality of arrays resulting from Dataloader after OnlineSequentialDataset.
    Specifically, we go from (1, nr_wdws, wdw_size, ...) to (nr_wdws, wdw_size, ....) or
    in a more machine learning common terminology to (batch_size, sequence_len, ...).

    Args:
        batch (list): list of tuples containing input and target batch of data
                        for online training

    Returns:
        (tuple): reshaped inputs and outputs
    """
    # get data out of list thus dropping first dim
    x = batch[0][0]
    y = batch[0][1]
    
    if x.ndim > 3:
        return x, y
        
    else:
        # get out seg data for centroids and correct dim 
        x_seg = batch[0][2]
        y_seg = batch[0][3]     
                  
        return x, y, x_seg, y_seg, batch[0][4], batch[0][5], batch[0][6], batch[0][7]

# %%
# total number of cases in training cohort
nr_cases = 88
# percentage of train val and test cases
train_split = 0.6
val_split = 0.2
# absolute nr of train val and test cases
train_cases = round(train_split * nr_cases)
train_val_cases = round(train_cases + val_split * nr_cases)
# if __name__ == "__main__":
#     # path to training and validation cases (to be adapted to your local path to the data)
#     path_data_in_train = utils.subdir_paths('/home/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv')[:train_cases]  
#     path_data_in_val = utils.subdir_paths('/home/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv')[train_cases:train_val_cases]
#     path_data_out_train = '/home/data/preprocessed/offline_training_data'
#     path_data_out_val = '/home/data/preprocessed/offline_validation_data'
    
#     print('Offline training set:')
#     generate_offline_data_folder(path_data_in=path_data_in_train, path_data_out=path_data_out_train, 
#                                     direction='SI_AP', breathhold_inclusion=False,
#                                     wdw_size_i=16, wdw_size_o=2, step_size=1, centroid_norm='wdw_norm')
#     print('Offline validation set:')
#     generate_offline_data_folder(path_data_in=path_data_in_val, path_data_out=path_data_out_val, 
#                                     direction='SI_AP', breathhold_inclusion=False,
#                                     wdw_size_i=16, wdw_size_o=2, step_size=1, centroid_norm='wdw_norm')
    
#     print('Offline training set:')
#     generate_offline_data_folder(path_data_in=path_data_in_train, path_data_out=path_data_out_train, 
#                                     direction='SI_AP', breathhold_inclusion=False,
#                                     wdw_size_i=32, wdw_size_o=2, step_size=1, centroid_norm='wdw_norm')
#     print('Offline validation set:')
#     generate_offline_data_folder(path_data_in=path_data_in_val, path_data_out=path_data_out_val, 
#                                     direction='SI_AP', breathhold_inclusion=False,
#                                     wdw_size_i=32, wdw_size_o=2, step_size=1, centroid_norm='wdw_norm')