# %%
import numpy as np
import torch
import cv2
import SimpleITK as sitk  
import os
import scipy
from skimage.morphology import closing, square
from skimage.measure import label
import pandas as pd
import matplotlib.pyplot as plt

# import self written modules
import data_preparation


# %%


def subdir_paths(path):
    " Given a path the function returns only primary subdirectories in a sorted list. "
    return list(filter(os.path.isdir, [os.path.join(path, f) for f in sorted(os.listdir(path))]))


def normalize(values, bounds, single_value=False, to_tensor=False):
    """ Normalize values in range define by bounds.

    Args:
        values (list or array or tensor): data to be normalized, shape=(nr_data_points)
        bounds (dict): current and desired bounds, for example
        {'actual':{'lower':5,'upper':15},'desired':{'lower':-1,'upper':1}}
        single_value: to give a single value as input (and output), i.e. nr_data_points=1
        to_tensor: convert to tensor

    Returns:
        array: array with normalized values
    """

    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    else:
        pass  
    
    if single_value:
        if to_tensor:
            return torch.tensor(bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower']))             
        else:
            return bounds['desired']['lower'] + (values - bounds['actual']['lower']) * \
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / \
                    (bounds['actual']['upper'] - bounds['actual']['lower'])      
    else:  
        if to_tensor: 
            return torch.tensor(np.array([bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]))
        else:
            return np.array([bounds['desired']['lower'] + (x - bounds['actual']['lower']) *
                    (bounds['desired']['upper'] - bounds['desired']['lower']) / 
                    (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values])



def hsv_to_rgb(hsv_tuple):
    """ Convert HSV (0-180, 0-255, 0-255) to RGB (0-255, 0-255, 0-255) values.
    Adapted from: https://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion
    """
    # extract from tuple and normalize
    h = hsv_tuple[0]/180
    s = hsv_tuple[1]/255
    v = hsv_tuple[2]/255
    # follows wikipedia formula
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)
 

       
def lookupdigits(x):
    """ Digits for target in/out status. """
    lookup = [[[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 0, 0],
       [1, 0, 0, 1, 0],
       [1, 0, 0, 1, 0],
       [0, 1, 1, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0]], [[0, 0, 1, 0, 0],
       [0, 1, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]],[[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [1, 1, 1, 1, 1]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 0, 0, 1, 0],
       [0, 0, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0]], [[0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]]]
    # lookup table: %-symbol is written as 0 because by integers there is no comma, 
    # e.g. 1%, not 1.0%, so it is fair to just define % as .0
    digits = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    for i in range(len(lookup)):
        if (np.array(x)==lookup[i]).all():
            return digits[i]
    print('no number')
    return -1


def framedigits(x):
    """ Frame number digits. """
    lookup = [[[0, 1, 1, 1, 0],   # hard-coded pixel matrix for '0' 
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 0, 0],
       [1, 0, 0, 1, 0],
       [1, 0, 0, 1, 0],
       [0, 1, 1, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0]], [[0, 0, 1, 0, 0],
       [0, 1, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]],[[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [1, 1, 1, 1, 1]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 1, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 0, 0, 1, 0],
       [0, 0, 1, 1, 0],
       [0, 1, 0, 1, 0],
       [0, 1, 0, 1, 0],
       [1, 0, 0, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0]], [[0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 0],
       [1, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]], [[0, 1, 1, 1, 0],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 1],
       [0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 1, 1, 0]]]
    digits = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0, and the letter "e" that is visible in the 2 digits frames
    for i in range(len(lookup)):
        if (np.array(x)==lookup[i]).all():
            return digits[i]
    # print('no number')
    return 0


def read_frame_nr(imgdata, nf):
    " Read out frame number by starting from the first digits and as soon as one black block is detected stop. "
    
    empty = [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]

    framenumb = []
    
    for f in range(nf):
        # print(f'Video frame number {f}/{nf}')
        # right most digit
        gray = cv2.cvtColor(imgdata[f, -69:-61,930:935],cv2.COLOR_BGR2GRAY)  # other range: imgdata[f,0:25,60:110]/(imgdata[f,10:18,67:81]
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        blur = cv2.GaussianBlur(gray,(1,1),1)
        bw = closing(imgdata[f,-69:-61,930:935,0]>thresh,square(1))
        label_image=label(bw)
        # print(f'label_image: {framedigits(label_image)}')
        
        #second digit from the right
        gray=cv2.cvtColor(imgdata[f,-69:-61,924:929],cv2.COLOR_BGR2GRAY)
        ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        blur = cv2.GaussianBlur(gray,(1,1),1)
        bw1=closing(imgdata[f,-69:-61,924:929,0]>thresh,square(1))
        label_image1=label(bw1)
        if (np.array(label_image1)==empty).all():
            framenumb.append(framedigits(label_image))
        else:
            gray=cv2.cvtColor(imgdata[f,-69:-61,918:923],cv2.COLOR_BGR2GRAY)
            ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            blur = cv2.GaussianBlur(gray,(1,1),1)
            bw2=closing(imgdata[f,-69:-61,918:923,0]>thresh,square(1))
            label_image2=label(bw2)
            # everytime when the empty is detected, calculate all the digits on the right side of the "empty digit"
            if (np.array(label_image2)==empty).all(): 
                framenumb.append(framedigits(label_image1)*10+framedigits(label_image))
            else:
                #plt.imshow(imgdata[f,-69:-61,912:917])
                #plt.show()
                gray=cv2.cvtColor(imgdata[f,-69:-61,912:917],cv2.COLOR_BGR2GRAY)
                ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                blur = cv2.GaussianBlur(gray,(1,1),1)
                bw3=closing(imgdata[f,-69:-61,912:917,0]>thresh,square(1))
                label_image3=label(bw3)
                if (np.array(label_image2)==empty).all():
                    framenumb.append(framedigits(label_image2)*100+framedigits(label_image1)*10+framedigits(label_image))
                else:
                    framenumb.append(framedigits(label_image3)*1000+framedigits(label_image2)*100+framedigits(label_image1)*10+framedigits(label_image)) #no "empty digits" at all since 4 digits, so just compute the 4digits
                    # print(f'Detected image number {framenumb[f]}')     
                    
    return framenumb


def find_pauses(nf, imgdata):
    """Find the frames with the expression "Image Paused" on the right upper corner."""
    
    imagepauses=np.zeros(nf)
    
    for f in range(nf):
        if np.sum(imgdata[f,10:20,850:])>100000:
            print(f'Image pause found for frame: {f}/{nf}')
            imagepauses[f]=1
            
    return imagepauses


def get_beam_status(nf, imgdata, version='pre-upgrade'):
    "Get the beam status On/Off from the original video."
    
    status=[]
    
    if version == 'pre-upgrade':
        # check if box at the bottom left corner is green=off or yellow=on
        for f in range(nf):
            if imgdata[f,-14:-6,6:14][0,0,0]>200 and imgdata[f,-14:-6,6:14][7,7,0]>200 and imgdata[f,-14:-6,6:14][0,7,0]>200 and imgdata[f,-14:-6,6:14][7,0,0]>200:
                status.append("on")
            elif imgdata[f,-14:-6,6:14][0,0,0]<200 and imgdata[f,-14:-6,6:14][7,7,0]<200 and imgdata[f,-14:-6,6:14][0,7,0]<200 and imgdata[f,-14:-6,6:14][7,0,0]<200:
                status.append("off")
            else:
                print('Attention: something went wrong!')
                
    elif version == 'post-upgrade':
        # check if at the position where the 2nf F of OFF would be, the green channel is large=off or small=on
        for f in range(nf):
            #plt.imshow(imgdata[f,-14:-6,6:14])
            #plt.show()
            if np.sum(imgdata[f,-13:,72:80][:,:,1]) < 3000:
                status.append("on")
            if np.sum(imgdata[f,-13:,72:80][:,:,1]) > 3000:
                status.append("off")
                # chekc if pixel group next to last F is empty, if not something's wrong
            if np.sum(imgdata[f,-13:,81:89][:,:,1]) > 1000:    
                print('Attention: pixel group next to last F does not seem empty!')
                
    else:
        raise Exception('Unknown version specified!')
       
    return status
            
   

def breathhold_detection_vectorized(array1, array2=None, wdw_size=20, amp_threshold1=0.05, amp_threshold2=0.005):
    """ Given a sequence of data, subdivide it in windows and then slide over them to find breath-holds.
    Args:
        array1: input sequence (e.g. inf-sup)
        array2: second input sequence (e.g. post-ant motion)
        wdw_size: size of sliding window 
        amp_threshold: normalized amplitude threshold below which to consider the corresponding window as a breath-hold
    """
    
    # get normalized data to be used to find indices of breathholds -->
    # needed as threshold works best on all data if the amplitudes are comparable (i.e. normalized)
    array1_norm = normalize(array1, {'actual': {'lower': np.min(array1), 'upper': np.max(array1)}, 
                                        'desired': {'lower': -1, 'upper': 1}})
    if array2 is not None:
        array2_norm = normalize(array2, {'actual': {'lower': np.min(array2), 'upper': np.max(array2)}, 
                                            'desired': {'lower': -1, 'upper': 1}}) 
         
    start = 0
    stop = len(array1)
    if array2 is not None:
        if len(array1) != len(array2):
            raise Exception('Attention! Length of array1 and array2 different.')
    step_size = 1 

    # find indices of all possible windows using vectorized operations
    idx_windows = (start + 
        np.expand_dims(np.arange(wdw_size), 0) +
        # Create a rightmost vector as [0, step, 2*step, ...].
        np.expand_dims(np.arange(stop - wdw_size + 1, step=step_size), 0).T)

    #print(array[idx_windows]) # e.g. [[0.8,0.9,0.92,0.9],[0.8,0.74,0.42,0.44]]

    breathholds=np.zeros(len(array1))
    # loop over all windows
    for window in idx_windows:
        # compute distances from median for normalized curve
        d1 = np.abs(array1_norm[window] - np.median(array1_norm[window]))
        #print(d)

        # compute median of distances from median
        mdev1 = np.median(d1)
        #print(mdev)

        # consider sequence breathhold if median of the distances is below a normalized amplitude of amp_threshold
        if mdev1 < amp_threshold1:
            breathholds[window] = 1

        # do the same for the second dimension and consider breathhold if there is constancy in either one of the two dims
        if array2 is not None:
            # compute distances from median for normalized curve
            d2 = np.abs(array2_norm[window] - np.median(array2_norm[window]))

            # compute median of distances from median
            mdev2 = np.median(d2)

            # consider sequence breathhold if median of the distances is below a normalized amplitude of amp_threshold
            if mdev2 < amp_threshold2:
                breathholds[window] = 1           

    return breathholds

def get_total_duration(path_data, breathhold_inclusion,
                       wdw_size_i, wdw_size_o, step_size, fps):
    """Get total duration of input sequences derived from a dataset of excel lists with centroid positions.

    Args:
        path_data (str): path to folder containing patients
        breathhold_inclusion (bool): whether to include breathholds
        wdw_size_i (int): length of input sequence
        wdw_size_o (int): length of output sequence
        step_size (int): step of rolling window
        fps (int): frames per second of cine videos
    """
    
    x_durations = []
    # loop over all cases
    for path_case in path_data:
        # check if folder with centroids and info is empty
        if len(os.listdir(os.path.join(path_case, 'centroids'))) == 0:
            # go to next case
            continue
        else:
            # loop over all info files of one case
            for _, _, file_list in os.walk(os.path.join(path_case, 'centroids')):
                for file_name_centroid_info in file_list:
                    # append paths 
                    path_centroid_info = os.path.join(path_case, 'centroids', file_name_centroid_info)

                    # load dataframe with info for current video
                    print(f'Loading data for: {path_centroid_info[:-5]}')
                    df = pd.read_excel(path_centroid_info)
                    # get info for splitting into snippets
                    pauses_start=df['Imaging paused start']
                    bhs=df['Breath-holds']
                    bhs_start=df['Breath-holds start']
                    
                    # get sequence of centroid positions
                    data = np.array(df['Target COM inf-sup (after smoothing) [mm]'].values)
                    # add channel dim 
                    data = data[:, None]            

                    # separate data into list of snippets with shape=(nr_snippets,) 
                    # according to image pauses and bhs
                    snippets_with_bh, snippets_without_bh = data_preparation.get_snippets(data=data, 
                                                                        pauses_start=pauses_start, 
                                                                        bhs=bhs, 
                                                                        bhs_start=bhs_start)
                                                
                    # include bhs in motion curves
                    if breathhold_inclusion:
                        # get data input and ouput windows
                        x, y = data_preparation.get_wdws(snippets_with_bh, 
                                            wdw_size_i=wdw_size_i, 
                                            wdw_size_o=wdw_size_o, 
                                            step_size=step_size)
                    # exclude bhs from motion curves
                    else:
                        # get data input and ouput windows
                        x, y = data_preparation.get_wdws(snippets_without_bh, 
                                            wdw_size_i=wdw_size_i, 
                                            wdw_size_o=wdw_size_o, 
                                            step_size=step_size) 

                    # concatenate all snippets in the data list to an array 
                    # with shape=(nr_wdws, wdw_size, (...)) and automatically drop empty items
                    x, y = np.concatenate(x, axis=0), np.concatenate(y, axis=0)
                    # be aware that nr_wdws will be different for each video so
                    # this Dataset must be used with a DataLoader using batch_size=1
                    
                    # append data to list containing for each case the input and output windows 
                    print(f'Nr frames current video: {x.shape[0]}')
                    x_durations.append(x.shape[0])
            
    # concatenate the data from all cases into an array of shape (total_nr_wdws, wdw_size, ...)
    total_duration = np.sum(x_durations)  / fps  
    print(f'{round(total_duration, 2)} [s]; \
            {round(total_duration/60, 2)} [min]; \
            {round(total_duration/3600, 2)} [h]')
        
    
def save_stats_train_val(path_saving, net_params, other_params, loss_name, 
                            train_losses, val_losses, 
                            train_dice_metrics, val_dice_metrics,
                            train_rmse_metrics, val_rmse_metrics,
                            tot_t=0):
    """Save statistics generated during offline network training and validation to txt file.

    Args:
        path_saving (string): path to results folder
        net_params (string): network parameters used during optimization
        other_params (string): other parameters used 
        loss_name (string): name of loss function used
        train_losses (list): training losses for different epochs
        val_losses (list): validation losses for different epochs
        tot_t (int): total time needed for optimization in 'unit'
    """

    with open(os.path.join(path_saving, 'stats.txt'), 'a') as file:
        file.write(f'Network parameters used: \n {net_params} \n')
        file.write(f'Other parameters used: \n {other_params} \n')
        file.write(f'Best train {loss_name} loss = {np.min(train_losses)} \n')
        file.write(f'Best val {loss_name} loss = {np.min(val_losses)} \n')
        if len(train_dice_metrics[0]) > 0:
            file.write(f'Best train dice metric (250 ms) = {np.max(train_dice_metrics[0])} \n')
            file.write(f'Best train dice metric (500 ms) = {np.max(train_dice_metrics[1])} \n')
            file.write(f'Best val dice metric (250 ms) = {np.max(val_dice_metrics[0])} \n')
            file.write(f'Best val dice metric (500 ms) = {np.max(val_dice_metrics[1])} \n')      
        if len(train_rmse_metrics[0]) > 0:          
            file.write(f'Best train rmse metric (250 ms) = {np.min(train_rmse_metrics[0])} \n')
            file.write(f'Best train rmse metric (500 ms) = {np.min(train_rmse_metrics[1])} \n')
            file.write(f'Best val rmse metric (250 ms) = {np.min(val_rmse_metrics[0])} \n')
            file.write(f'Best val rmse metric (500 ms) = {np.min(val_rmse_metrics[1])} \n')  
        file.write('\n')
        file.write(f'------ Total time needed for optimization: {tot_t} min ------- ') 
    

def save_stats_train_val_online(path_saving, net_params, other_params,
                                dice_metric_videos, hd_max_metric_videos,
                                hd_95_metric_videos, hd_50_metric_videos, 
                                rmse_SI_metric_videos, rmse_AP_metric_videos,
                                tot_t_online_videos=[], set='val'):
    """Save statistics generated during online network training (optional) and validation/testing to txt file.

    Args:
        path_saving (string): path to results folder
        net_params (string): network parameters used during optimization
        other_params (string): other parameters used 
        metric_videos (dict): dictionary with list containing given metric for each 
                                cine video for 250 ms [0] and 500 ms [1] forecasts
        tot_times_online (list): list with online training times                           
        set (str): either 'train', 'val' or 'test', gives info on which data set was 
                    actually used during model inference
    """

    with open(os.path.join(path_saving, f'stats_online_{set}.txt'), 'a') as file:
        file.write(f'Network parameters used: \n {net_params} \n')
        file.write(f'Other parameters used: \n {other_params} \n')
        if len(dice_metric_videos[0]) > 0:
            file.write(f'Average {set} dsc metric (250 ms) = {np.mean(dice_metric_videos[0])} \n')
            file.write(f'STD {set} dsc metric (250 ms) = {np.std(dice_metric_videos[0])} \n')
            file.write(f'Average {set} dsc metric (500 ms) = {np.mean(dice_metric_videos[1])} \n')
            file.write(f'STD {set} dsc metric (500 ms) = {np.std(dice_metric_videos[1])} \n')   
            file.write(f'\n')     
        if len(hd_max_metric_videos[0]) > 0:          
            file.write(f'Average {set} hd max metric (250 ms) = {np.mean(hd_max_metric_videos[0])} \n')
            file.write(f'STD {set} hd max metric (250 ms) = {np.std(hd_max_metric_videos[0])} \n')
            file.write(f'Average {set} hd max metric (500 ms) = {np.mean(hd_max_metric_videos[1])} \n')
            file.write(f'STD {set} hd max metric (500 ms) = {np.std(hd_max_metric_videos[1])} \n')   
            file.write(f'\n')     
        if len(hd_95_metric_videos[0]) > 0:          
            file.write(f'Average {set} hd 95th metric (250 ms) = {np.mean(hd_95_metric_videos[0])} \n')
            file.write(f'STD {set} hd 95th metric (250 ms) = {np.std(hd_95_metric_videos[0])} \n')
            file.write(f'Average {set} hd 95th metric (500 ms) = {np.mean(hd_95_metric_videos[1])} \n')
            file.write(f'STD {set} hd 95th metric (500 ms) = {np.std(hd_95_metric_videos[1])} \n') 
            file.write(f'\n')      
        if len(hd_50_metric_videos[0]) > 0:          
            file.write(f'Average {set} hd 50th metric (250 ms) = {np.mean(hd_50_metric_videos[0])} \n')
            file.write(f'STD {set} hd 50th metric (250 ms) = {np.std(hd_50_metric_videos[0])} \n')
            file.write(f'Average {set} hd 50th metric (500 ms) = {np.mean(hd_50_metric_videos[1])} \n')
            file.write(f'STD {set} hd 50th metric (500 ms) = {np.std(hd_50_metric_videos[1])} \n')  
            file.write(f'\n')     
        if len(rmse_SI_metric_videos[0]) > 0:          
            file.write(f'Average {set} rmse SI metric (250 ms) = {np.mean(rmse_SI_metric_videos[0])} \n')
            file.write(f'STD {set} rmse SI metric (250 ms) = {np.std(rmse_SI_metric_videos[0])} \n')
            file.write(f'Average {set} rmse SI metric (500 ms) = {np.mean(rmse_SI_metric_videos[1])} \n')
            file.write(f'STD {set} rmse SI metric (500 ms) = {np.std(rmse_SI_metric_videos[1])} \n')   
            file.write(f'\n')     
        if len(rmse_AP_metric_videos[0]) > 0:          
            file.write(f'Average {set} rmse AP metric (250 ms) = {np.mean(rmse_AP_metric_videos[0])} \n')
            file.write(f'STD {set} rmse AP metric (250 ms) = {np.std(rmse_AP_metric_videos[0])} \n')
            file.write(f'Average {set} rmse AP metric (500 ms) = {np.mean(rmse_AP_metric_videos[1])} \n')
            file.write(f'STD {set} rmse AP metric (500 ms) = {np.std(rmse_AP_metric_videos[1])} \n')  
            file.write(f'\n')     
        file.write('\n')
        file.write(f'------ Average time needed for online optimization: {np.mean(tot_t_online_videos)} ms ------- ') 
        

    # save metrics
    np.savetxt(os.path.join(path_saving, 'dice_250.txt'), dice_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'dice_500.txt'), dice_metric_videos[1])  
    np.savetxt(os.path.join(path_saving, 'hdmax_250.txt'), hd_max_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'hdmax_500.txt'), hd_max_metric_videos[1])    
    np.savetxt(os.path.join(path_saving, 'hd95_250.txt'), hd_95_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'hd95_500.txt'), hd_95_metric_videos[1])        
    np.savetxt(os.path.join(path_saving, 'hd50_250.txt'), hd_50_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'hd50_500.txt'), hd_50_metric_videos[1])    
    np.savetxt(os.path.join(path_saving, 'rmseSI_250.txt'), rmse_SI_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'rmseSI_500.txt'), rmse_SI_metric_videos[1])
    np.savetxt(os.path.join(path_saving, 'rmseAP_250.txt'), rmse_AP_metric_videos[0])
    np.savetxt(os.path.join(path_saving, 'rmseAP_500.txt'), rmse_AP_metric_videos[1])
    
    


def get_info_preprocessed_cohort(path_cohort, path_saving, save=False):
    """Get information from the excel sheets of a preprocessed data set.

    Args:
        path_cohort (str): path to preprocessed cohort
        path_saving (str): path to results folder to contain plots etc
        save (bool, optional): whether to save plots or not. Defaults to False.
    """
    
    snippet_durations_with_bh = [] 
    snippet_durations_without_bh = []
    nr_xlsx = 0
    
    # create folder for results
    os.makedirs(path_saving, exist_ok=True)
        
    # loop over all paths of cases 
    for path_case in subdir_paths(path_cohort):
        
        # eg 'case03'
        current_case = os.path.basename(path_case)
        print(f'\n-------- Current case: {current_case} -----------\n')  

        # loop over all files of one case
        for _dir_name, _subdir_list, file_list in os.walk(path_case):
            for file_name in file_list:
                
                # check if file is xlsx, else go to next file
                if file_name.endswith('.xlsx'):
                    nr_xlsx += 1
                    
                    # read in excel file for one video
                    df = pd.read_excel(os.path.join(path_case, file_name),
                                       engine='openpyxl')
                    
                    #print(df)
                    # append average snippets duration for each video
                    snippet_durations_without_bh.append(df.loc[0, 'Mean snippet duration without/with BH [s]'])
                    snippet_durations_with_bh.append(df.loc[1, 'Mean snippet duration without/with BH [s]'])
                    
    #  get video and case averaged mean snippets duration 
    mean_snippet_without_bh = np.mean(np.array(snippet_durations_without_bh))              
    mean_snippet_with_bh = np.mean(np.array(snippet_durations_with_bh)) 
             
    #  plot histograms
    plt.hist(snippet_durations_without_bh, bins=50, label=f'Mean: {round(mean_snippet_without_bh, 2)} s')   
    plt.legend(loc="upper right")
    plt.ylabel('Occurrence')
    plt.xlabel('Average snippet duration without BHs')
    if save:
        plt.savefig(os.path.join(path_saving, f'hist_snippet_duration_without_BHs.png'), 
                    bbox_inches="tight")
    plt.close()

    plt.hist(snippet_durations_with_bh, bins=50, label=f'Mean: {round(mean_snippet_with_bh, 2)} s')   
    plt.legend(loc="upper right")
    plt.ylabel('Occurrence')
    plt.xlabel('Average snippet duration with BHs')
    if save:
        plt.savefig(os.path.join(path_saving, f'hist_snippet_duration_with_BHs.png'), 
                    bbox_inches="tight")
    plt.close()  
    
    
    with open(os.path.join(path_saving, 'results_info.txt'), 'a') as file:
        file.write(f'Number of excel sheets found: {nr_xlsx} \n')
        
    print('...results saved.')
    

def get_duty_cycle_efficiency(status, nr_frames_total, fps, 
                              file_name, path_results=None, 
                              overall=False):
    """Get the duty cycle efficiency defined as the number of “beam-on” frames 
    divided by the total number of MR ciné frames acquired during that fraction.

    Args:
        status (list): on or off status
        framenumb (list): frame numbers
        nf (int): total nr of frames in video
        fps (float): frames per second of video
        file_name (str): name of fraction
        path_results (str, optional): path where to store txt with results. Defaults to None.
        overall (bool, optional): if True, the text in the txt will be chnaged
    """
    # duty cycle efficiency analysis
    nr_beam_on_frames = 0
    
    for el in status:
        if el == 'on':
            nr_beam_on_frames += 1
    print(f'Number of beam on frames: {nr_beam_on_frames}')
           
    duty_cycle_eff = nr_beam_on_frames/nr_frames_total
    print(f'Duty cycle efficiency: {duty_cycle_eff}')

    if path_results is not None:
        with open(os.path.join(path_results, 'duty_efficiency_stats.txt'), 'a') as file:
            if overall is False:
                file.write(f'Treatment delivery time / Beam on time / Duty cycle efficiency for ' +
                        file_name + 
                        f': {round(nr_frames_total/(fps*60),3)} mins / {round(nr_beam_on_frames/(fps*60),3)} mins / {nr_beam_on_frames}/{nr_frames_total} = {round(duty_cycle_eff*100, 3)} % \n')
            else:
                file.write(f'Overall duty cycle efficiency' +
                        f': {nr_beam_on_frames}/{nr_frames_total} = {round(duty_cycle_eff*100, 3)} % \n')


def resample_image(in_array, out_spacing=(1.0, 1.0, 1.0), 
                   interpolator=sitk.sitkNearestNeighbor, default_value=0):
    """Resample numpy array to specified spacing using SITK.

    Args:
        in_array (numpy array): Image in mupy array format with dimensions (d,h,w).
        out_spacing (tuple, optional): Spacing of resampled image with dimensions (d,h,w) 
            compared to spacing of original image which is automatically set to (1.0, 1.0, 1.0). 
            For example  when setting out_spacing=(1.0, 1.0, 0.5) the image will  
            be upsampled by a factor of 2 in height.
        interpolator (SITK interpolator, optional): For example sitk.sitkLinear, sitk.sitkNearestNeighbor.
        default_value (int, optional): Pixel value used when a transformed pixel is outside of the image.

    Returns:
        out_array: Resampled image as numpy array with dimensions (d,h,w).
    """

    # get sitk image from numpy array
    in_image = sitk.GetImageFromArray(in_array)

    # get input and ouput spacing and size
    in_spacing = in_image.GetSpacing()  # (1.0, 1.0, 1.0)
    out_spacing = out_spacing[::-1] # change to sitk indexing (w,h,d)
    in_size = in_image.GetSize()  # (w,h,d) as SITK indexing is inverted compared to numpy
    out_size = (int(in_size[0] * (in_spacing[0] / out_spacing[0])),
                int(in_size[1] * (in_spacing[1] / out_spacing[1])),
                int(in_size[2] * (in_spacing[2] / out_spacing[2])))

    # set resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputDirection(in_image.GetDirection())
    resampler.SetOutputOrigin(in_image.GetOrigin())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    
    # perform interpolation with parameters set above
    out_image = resampler.Execute(in_image) 

    out_array = sitk.GetArrayFromImage(out_image)
    # print(out_array.shape) # (d,h,w) as GetArrayFromImage restores numpy indexing

    return out_array   


def shift_by_centroid_diff(predictions, inputs, inputs_seg,
                           min_amplitude_SI, max_amplitude_SI, min_amplitude_AP, max_amplitude_AP):
    """Shift last input segmentation by difference between predicted centorids and centroids of last input segmentation.

    Args:
        predictions (array or tensor): predicted centroids in SI and AP direction
        inputs (array or tensor): input centroids in SI and AP direction
        inputs_seg (array): input binary segmentations
        min_amplitude_SI (float): minimum amplitude in SI to undo normalization of centroids
        max_amplitude_SI (float): maximum amplitude in SI to undo normalization of centroids
        min_amplitude_AP (float): minimum amplitude in AP to undo normalization of centroids
        max_amplitude_AP (float): maximum amplitude in AP to undo normalization of centroids

    Returns:
        tensor: predicted, i.e. shifted binary segmentation
    """
    
    predictions_SI_mm = normalize(predictions[0,:,0], {'actual': {'lower': -1, 'upper': 1}, 
                            'desired': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}})
    predictions_AP_mm = normalize(predictions[0,:,1], {'actual': {'lower': -1, 'upper': 1}, 
                            'desired': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}})
    last_input_SI_mm = normalize(inputs[0,-1,0], {'actual': {'lower': -1, 'upper': 1}, 
                            'desired': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}}, single_value=True)
    last_input_AP_mm = normalize(inputs[0,-1,1], {'actual': {'lower': -1, 'upper': 1}, 
                            'desired': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}}, single_value=True)
    
    # difference in centroids positions for 250 ms forecast
    delta_centroids0_SI = -(predictions_SI_mm[0] - last_input_SI_mm)  # positive means superior for centroids but inferior for segmentations
    delta_centroids0_AP = predictions_AP_mm[0] - last_input_AP_mm  # positive means anterior direction for both
    # difference in centroids positions for 500 ms forecast
    delta_centroids1_SI = -(predictions_SI_mm[1] - last_input_SI_mm)
    delta_centroids1_AP = predictions_AP_mm[1] - last_input_AP_mm    
    
    # take last input segmentation and shift it by delta centroids  
    predictions_seg0 = scipy.ndimage.shift(inputs_seg[0, -1, 0, :, :], 
                                                shift=[delta_centroids0_SI, delta_centroids0_AP],
                                                order=3, mode='nearest')
    predictions_seg1 = scipy.ndimage.shift(inputs_seg[0, -1, 0, :, :], 
                                                shift=[delta_centroids1_SI, delta_centroids1_AP],
                                                order=3, mode='nearest')
    # concatenate two forecasts and restore b,s,c,h,w, dimensionality
    predictions_seg = torch.tensor(np.concatenate((predictions_seg0[None, None, None, ...], predictions_seg1[None, None, None, ...]), 
                                                        axis=1))
    return predictions_seg


def get_centroids_segmentations(outputs, predictions, to_tensor=True):
    """Get center of mass for output segmentations and predicted segmentations for 250 ms and 500 ms forecast.

    Args:
        outputs (array or tensor): ground truth segmentations with shape (1, 2, 1, ...)
        predictions (array or tensor): predicted segmentations with shape (1, 2, 1, ...)

    Returns:
        tuple with floats: centroids positions in SI and AP.
    """
    
    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    else:
        pass  
    
    centroids_output0 = scipy.ndimage.measurements.center_of_mass(outputs[0,0,0,...])
    centroids_predictions0 = scipy.ndimage.measurements.center_of_mass(predictions[0,0,0,...])
    centroids_output1 = scipy.ndimage.measurements.center_of_mass(outputs[0,1,0,...])
    centroids_predictions1 = scipy.ndimage.measurements.center_of_mass(predictions[0,1,0,...])
    
    if to_tensor:
        centroids_output0, centroids_predictions0 = torch.tensor(centroids_output0, dtype=torch.float32), torch.tensor(centroids_predictions0, dtype=torch.float32)
        centroids_output1, centroids_predictions1 = torch.tensor(centroids_output1, dtype=torch.float32), torch.tensor(centroids_predictions1, dtype=torch.float32)
    
    return centroids_output0, centroids_predictions0, centroids_output1, centroids_predictions1


def get_different_model_predictions(example_inputs_original, example_outputs_original,
                                    example_inputs_seg_original, example_outputs_seg_original,
                                    example_outputs_frame_original,
                                    model_lstm_si_ap, model_convlstm, model_convlstm_stl,
                                    device=None, batches=[63, 64, 65], center=128, crop_halves=128//3,
                                    net=None):
    """Get predicted output segmentations for different models for later plotting. """

    ground_truth_centroids = []
    ground_truth_segs = []
    ground_truth_frames = []
    lstm_si_ap_coms = []
    convlstm_coms = []
    convlstm_stl_coms = []
    no_predictor_coms = []
    predicted_coms = []
    lstm_si_ap_segs = []
    convlstm_segs = []
    convlstm_stl_segs = []
    no_predictor_segs = []
    predicted_segs = []
                                        
    for batch in batches:
        print(f'--------- batch {batch-batches[0]}/{len(batches)} ------------')
        # plot ground truth vs predicted for example data 
        example_inputs, example_outputs = example_inputs_original[batch, None].to(device), example_outputs_original[batch, None].to(device)
        example_inputs_seg, example_outputs_seg = example_inputs_seg_original[batch, None].to(device), example_outputs_seg_original[batch, None].to(device)
        example_outputs_frame = example_outputs_frame_original[batch, None].to(device)
        
        # ground truth frame
        ground_truth_frames.append(example_outputs_frame[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy())
        
        # ground truth seg
        ground_truth_segs.append((example_outputs_seg[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)

        # ground truth centroids in SI
        ground_truth_centroids.append(example_outputs[0,-1,0])
        
        # normalize centroid curves with corresponding amplitudes (wdw_norm)
        max_amplitude_SI, min_amplitude_SI = torch.max(example_inputs[:,:,0]).item(), torch.min(example_inputs[:,:,0]).item()
        max_amplitude_AP, min_amplitude_AP = torch.max(example_inputs[:,:,1]).item(), torch.min(example_inputs[:,:,1]).item()
            
        example_inputs_0 = normalize(example_inputs[:,:,0], 
                                        {'actual': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
        example_inputs_1 = normalize(example_inputs[:,:,1], 
                                        {'actual': {'lower': min_amplitude_AP, 'upper': max_amplitude_AP}, 
                                        'desired': {'lower': -1, 'upper': 1}}, to_tensor=True) 
        example_inputs_norm = torch.cat((example_inputs_0[:,:,None], example_inputs_1[:,:,None]), dim=-1)
        
        # get conversion between pixel number and poistion in mm from center of boundary contour
        pixel_to_position_SI =  example_inputs[0,-1,0] + scipy.ndimage.measurements.center_of_mass(example_inputs_seg[0,-1,0,...].detach().cpu().numpy())[0]


        if net is None:
            # LSTM SI AP      
            example_predictions_lstm_si_ap = model_lstm_si_ap(example_inputs_norm)
            lstm_si_ap_coms.append(normalize(example_predictions_lstm_si_ap[0,-1,0], {'actual': {'lower': -1, 'upper': 1}, 
                                    'desired': {'lower': min_amplitude_SI, 'upper': max_amplitude_SI}}, single_value=True)) 
            example_predictions_seg_lstm_si_ap = shift_by_centroid_diff(predictions=example_predictions_lstm_si_ap, 
                                                                inputs=example_inputs_norm, inputs_seg=example_inputs_seg,
                                                                min_amplitude_SI=min_amplitude_SI, max_amplitude_SI=max_amplitude_SI, 
                                                                min_amplitude_AP=min_amplitude_AP, max_amplitude_AP=max_amplitude_AP)
            lstm_si_ap_segs.append((example_predictions_seg_lstm_si_ap[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)

            # ConvLSTM
            example_predictions_convlstm = model_convlstm(example_inputs_seg)
            convlstm_segs.append((example_predictions_convlstm[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)
            convlstm_coms.append(pixel_to_position_SI - scipy.ndimage.measurements.center_of_mass((example_predictions_convlstm[0,-1,0,...].detach().cpu().numpy() > 0.5) * 1.0)[0])
                
            # ConvLSTMSTL
            example_predictions_convlstm_stl, _, _ = model_convlstm_stl(example_inputs_seg)
            convlstm_stl_segs.append((example_predictions_convlstm_stl[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)
            convlstm_stl_coms.append(pixel_to_position_SI - scipy.ndimage.measurements.center_of_mass((example_predictions_convlstm_stl[0,-1,0,...].detach().cpu().numpy() > 0.5) * 1.0)[0])

            # no predictor
            no_predictor_segs.append((example_inputs_seg[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)
            no_predictor_coms.append(example_inputs[0,-1,0])
        
        
        elif net == 'LSTM-shift':
            example_predictions_lstm_si_ap = model_lstm_si_ap(example_inputs_norm)
            example_predictions_seg_lstm_si_ap = shift_by_centroid_diff(predictions=example_predictions_lstm_si_ap, 
                                                                inputs=example_inputs_norm, inputs_seg=example_inputs_seg,
                                                                min_amplitude_SI=min_amplitude_SI, max_amplitude_SI=max_amplitude_SI, 
                                                                min_amplitude_AP=min_amplitude_AP, max_amplitude_AP=max_amplitude_AP)
            predicted_segs.append((example_predictions_seg_lstm_si_ap[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)
        
        elif net == 'ConvLSTM-STL':
            example_predictions_convlstm_stl, _, _ = model_convlstm_stl(example_inputs_seg)
            predicted_segs.append((example_predictions_convlstm_stl[0,-1,0,center-crop_halves:center+crop_halves,center-crop_halves:center+crop_halves].detach().cpu().numpy() > 0.5) * 1.0)
   
        else:
            raise ValueError('Unexpected net name!')
        
    if net is None:
        return ground_truth_centroids, ground_truth_segs, ground_truth_frames, lstm_si_ap_coms, convlstm_coms, convlstm_stl_coms, no_predictor_coms, \
                                                                                lstm_si_ap_segs, convlstm_segs, convlstm_stl_segs, no_predictor_segs
    else:
        return ground_truth_centroids, ground_truth_segs, ground_truth_frames, predicted_segs
       
    
    
# %%

# if __name__ == "__main__":
#     path_data_train = subdir_paths('/home/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv')[:data_preparation.train_cases]
#     path_data_val = subdir_paths('/home/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv')[data_preparation.train_cases:data_preparation.train_val_cases]
#     path_data_test_LMU = subdir_paths('/home/data/preprocessed/2021_06_16_respiratory_patients_LMU_ogv')[data_preparation.train_val_cases:]
#     path_data_test_Gemelli = subdir_paths('/home/data/preprocessed/2021_10_25_free_breathing_patients_co60_Gemelli_ogv')[:]

#     get_total_duration(path_data=path_data_train, breathhold_inclusion=True,
#                         wdw_size_i=32, wdw_size_o=2, step_size=1, fps=4)
#     get_total_duration(path_data=path_data_val, breathhold_inclusion=True,
#                         wdw_size_i=32, wdw_size_o=2, step_size=1, fps=4)
#     get_total_duration(path_data=path_data_test_LMU, breathhold_inclusion=True,
#                         wdw_size_i=32, wdw_size_o=2, step_size=1, fps=4)
#     get_total_duration(path_data=path_data_test_Gemelli, breathhold_inclusion=True,
#                         wdw_size_i=32, wdw_size_o=2, step_size=1, fps=4)