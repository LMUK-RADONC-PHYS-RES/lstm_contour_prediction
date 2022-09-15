# %%

import matplotlib.pyplot as plt
import os
import numpy as np
import torch

# %%

def in_out_pred_frames_plot_vertical(inputs, outputs, predictions, 
                                        nr_frames=2, path_saving=None, 
                                        display=False):
 
    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(inputs):
        inputs = inputs.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type')   


    # visualizing results
    fig, axs = plt.subplots(nr_frames, 3, figsize=(16, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace =.1, wspace=.01)
    # flatten out axes
    # axs = axs.ravel()

    for i in range(nr_frames):
        axs[i, 0].imshow(inputs[0,-nr_frames+i,0,:,:], cmap='gray')
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[0, 0].set_title(f'Ground truth input')

        axs[i, 1].imshow(outputs[0,i,0,:,:], cmap='gray')
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[0, 1].set_title(f'Ground truth output')

        axs[i, 2].imshow(predictions[0,i,0,:,:], cmap='gray')
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        axs[0, 2].set_title(f'Predicted output')

    if path_saving is not None:
        plt.savefig(path_saving)
    if display:
        plt.show()   
    plt.close()
    

def dvf_frame_overlay_plot(computed_displacements, predictions, 
                           pat_nr=0, frame_nr=1, path_saving=None, 
                           display=False, to_cpu=False):

    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
        computed_displacements = computed_displacements.detach().cpu().numpy()
    elif isinstance(predictions, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type')   
    
    # get dvfs in two directions
    u = computed_displacements[pat_nr, frame_nr, :, :, 0]
    v = computed_displacements[pat_nr, frame_nr, :, :, 1]
    
    # overlay of displacement and images
    fig,ax = plt.subplots(figsize=(14,14))
    pa = ax.quiver(u,-v) # apparently due to axes display the 2nd dimension for the displacement has to be multiplied with -1
    pb = ax.imshow(predictions[pat_nr,frame_nr-1,0,:,:], cmap='gray', alpha=0.2)
    ax.set_title(f'Inverse DVF from prediction {frame_nr-1} to {frame_nr}')
    
    if path_saving is not None:
        plt.savefig(path_saving)
    if display:
        plt.show()   
    plt.close()


def in_out_pred_frames_plot(inputs, outputs, predictions1, 
                            predictions2=None, predictions3=None, 
                            nr_frames=10, nr_models=1, 
                            model_name1=None, model_name2=None, model_name3=None,
                            path_saving=None, display=False):

    # visualizing results
    fig, axs = plt.subplots(nr_models+2, nr_frames, figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    # flatten out axes
    axs = axs.ravel()

    for i in range(nr_frames):
        axs[i].imshow(inputs[0,i,0,:,:].detach().cpu().numpy())
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[nr_frames//2].set_title(f'Ground truth input')

        axs[i+1*nr_frames].imshow(outputs[0,i,0,:,:].detach().cpu().numpy())
        axs[i+1*nr_frames].set_xticks([])
        axs[i+1*nr_frames].set_yticks([])
        axs[nr_frames + nr_frames//2].set_title(f'Ground truth output')

        axs[i+2*nr_frames].imshow(predictions1[0,i,0,:,:].detach().cpu().numpy())
        axs[i+2*nr_frames].set_xticks([])
        axs[i+2*nr_frames].set_yticks([])
        axs[2*nr_frames + nr_frames//2].set_title(f'Predicted output - {model_name1}')

        if predictions2 is not None:
            axs[i+3*nr_frames].imshow(predictions2[0,i,0,:,:].detach().cpu().numpy())
            axs[i+3*nr_frames].set_xticks([])
            axs[i+3*nr_frames].set_yticks([])
            axs[3*nr_frames + nr_frames//2].set_title(f'Predicted output - {model_name2}')

        if predictions3 is not None:
            axs[i+4*nr_frames].imshow(predictions3[0,i,0,:,:].detach().cpu().numpy())
            axs[i+4*nr_frames].set_xticks([])
            axs[i+4*nr_frames].set_yticks([])
            axs[4*nr_frames + nr_frames//2].set_title(f'Predicted output - {model_name3}')


    if path_saving is not None:
        plt.savefig(path_saving)

    if display:
        plt.show()   
        
    plt.close()
    

def box_plot(x, y, fn, stats=True, 
             median_x=None, iqr_x=None, median_y=None, iqr_y=None, 
             display=True, save=False, path_saving=None, variant=None):    
    """
    Get boxplots for x and y motion.
    https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
    """
    
    boxdata = [x, y]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(boxdata, patch_artist=True)
    colors = ['blue', 'green']
    for patch, color in zip(bp['boxes'], colors): 
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='red', linewidth=4)
    for flier in bp['fliers']:
        flier.set(marker='D', color='black', alpha=0.5)
    ax.set_title("Target motion", fontsize=25)
    plt.ylabel("Shift in mm")
    ax.set_xticklabels(['Post- Ant motion', 'Inf-Sup motion'])
    
    if stats:
        plt.text(1.0, 0.75, f'Post-ant median and IQR: {round(median_x, 2)}, {round(iqr_x, 2)}', 
                 fontdict={'color': 'darkred',
                'weight': 'normal',
                'size': 16, }, transform=ax.transAxes)
        plt.text(1.0, 0.25, f'Inf-sup median and IQR: {round(median_y, 2)}, {round(iqr_y, 2)}', 
                 fontdict={'color': 'darkred',
                'weight': 'normal',
                'size': 16, }, transform=ax.transAxes)
        
    plt.grid(True)
    plt.subplots_adjust(right=0.35) 
      
    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_boxplot_' + variant + '.png'),
                    bbox_inches="tight")
    if display:
        # plt.show(bp)   
        plt.show()   
    plt.close()
    
     
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
 
                
def violin_plot(x, y, fn,
             display=True, save=False, path_saving=None, variant=None):
    """
    Get violinplot.
    https://matplotlib.org/3.1.1/gallery/statistics/customized_violin.html
    https://eustomaqua.github.io/2020/2020-03-24-Matplotlib-Tutorial-Gallery/
    """
        
    boxdata = [sorted(x), sorted(y)] 
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    vp = ax.violinplot(boxdata, showmeans=False, showmedians=True, showextrema=False)
    for pc in vp['bodies']:
        # pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    quartile1, medians, quartile3 = np.percentile(boxdata, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3) 
        for sorted_array, q1, q3 in zip(boxdata, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=10)
    ax.vlines(inds, whiskers_min, whiskers_max, color='green', linestyle='-', lw=3)

    # set style for the axes
    labels = ['Post-Ant', 'Int-Sup']
    for ax in [ax]:
        set_axis_style(ax, labels)

    plt.title(" Target motion", fontsize=25)
    plt.ylabel("Shift in mm")
    plt.grid(True)
    
    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_violinplot_' + variant + '.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
        

# Scatter histogram https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/scatter_hist.html
def scatter_hist(x, y, fn, 
                 display=True, save=False, path_saving=None, variant=None):
        
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03  # changing the distance between the histograms and the scatter plot


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    # fig = plt.figure(figsize=(8, 8))

    ax = plt.axes(rect_scatter)
    ax.set_xlabel('Post-Ant [mm]')
    ax.set_ylabel('Inf-Sup [mm]')
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    binwidth = 0.15
    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax/binwidth) + 1) * binwidth
    lim = 6
    ax.set_xlim()  # adjust this to the points
    ax.set_ylim()

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_scatterhist' + variant + '.png'),
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
 
 
def scatter_hist_3d(x, y, fn, tm, 
                  display=True, save=False, path_saving=None, variant=None):
    """ # Scatter histogram https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter """
    
    plt.figure(figsize=(10, 7))
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03  # changing the distance between the histograms and the scatter plot


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax = plt.axes(rect_scatter)
    ax.set_xlabel('Post-Ant [mm] \n ' + fn)
    ax.set_ylabel('Inf-Sup [mm]')
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    
    
    # nolabels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    # the scatter plot
    cm = plt.cm.get_cmap('viridis')
    sc = ax.scatter(x, y, c=tm, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label('Time [s]', rotation=270, labelpad=22)

    binwidth = 0.15
    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax/binwidth) + 1) * binwidth
    lim = 6
    
   
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='#D7191C')
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_scatterhist3d_' + variant + '.png'),
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
        
        
def random_frame_filling(nf, target, original_target, fn,
                         display=True, save=False, path_saving=None):
    """ Plot a few random frames to check if they were filled. """

    random_frames = np.array([[nf // 5, nf // 3], [nf // 2, nf // 1.3]], dtype=np.int)
    # print(random_frames)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 16), sharex=True, sharey=True)
    for col in range(2):
        for row in range(2):
            ax[col][row].imshow(target[random_frames[col][row]] + original_target[random_frames[col][row]])
            ax[col][row].set_title(f'Frame {random_frames[col][row]}/{nf}', fontsize=18)

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_random_frame_filling.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
    plt.close()


def motion_plot(tm, cxm, cym, fn, display=True, save=False, path_saving=None):
    """ Plot motion in millimters. """

    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))
    axs[0].plot(tm, cxm, 'ko', linestyle='-')
    axs[0].set_ylabel('Post-ant motion [mm]')
    axs[1].plot(tm, cym, 'ko', linestyle='-')
    axs[1].set_ylabel('Inf-sup motion [mm]')  
    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(tm[0] - 1, tm[-1] + 1)
    axs[0].grid(True)
    axs[1].grid(True)

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_motion_in_mm.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
            
         
def motion_smoothing_comparison(tm, cx, cx_or, cx_f_or, cy, cy_or, cy_f_or,
                                fn, fps, display=True, save=False, path_saving=None):
    """ Plot original, outlier replaced and filterd motion curves in same subplot to allow for a comparison."""
    
    fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))

    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black', label='original')
    axs[0].plot(tm, cx_or, 'ko', linestyle='--', color='red', label='replaced')
    axs[0].plot(tm, cx_f_or, 'ko', linestyle='--', color='blue', label='replaced and filtered')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black', label='original')
    axs[1].plot(tm, cy_or, 'ko', linestyle='--', color='red', label='replaced')
    axs[1].plot(tm, cy_f_or, 'ko', linestyle='--', color='blue', label='replaced and filtered')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(tm[0], tm[-1] + 1)
    
    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_smoothing_comparison.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps:
        fig, axs = plt.subplots(2, sharex=True, sharey=False, figsize=(15, 10))

        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[0].plot(tm[:int(100 * fps)], cx_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='red', label='replaced')
        axs[0].plot(tm[:int(100 * fps)], cx_f_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='blue', label='replaced and filtered')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[1].plot(tm[:int(100 * fps)], cy_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='red', label='replaced')
        axs[1].plot(tm[:int(100 * fps)], cy_f_or[:int(100 * fps)], 'ko', linestyle='--', 
                    color='blue', label='replaced and filtered')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylabel('Inf-sup motion [mm]') 

        axs[1].set_xlabel('Time [s]')

        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_smoothing_comparison_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    plt.close()
                
            
def motion_with_info(tm, cx, cx_or, cx_f_or, cy, cy_or, cy_f_or,
                    status, breathholds, framenumb_all, imagepauses, fn, 
                    fps, display=True, save=False, path_saving=None):
    
    """" Plot motion curves plus beam status, image pauses and breathhold information. """
    
    # defined globally for all figures
    plt.rcParams['axes.grid'] = True
    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25)
    plt.rc('axes', labelsize=25)  # to change the size of the letters

    fig, axs = plt.subplots(5, sharex=True, sharey=False, figsize=(25, 20), 
                            gridspec_kw={'height_ratios': [2, 2, 1, 1, 1]})
    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black', label='original')
    axs[0].plot(tm, cx_or, 'ko', linestyle='-', color='red', label='replaced')
    axs[0].plot(tm, cx_f_or, 'ko', linestyle='-', color='blue', label='replaced and filtered')
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black', label='original')
    axs[1].plot(tm, cy_or, 'ko', linestyle='-', color='red', label='replaced')
    axs[1].plot(tm, cy_f_or, 'ko', linestyle='-', color='blue', label='replaced and filtered')
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlim(tm[0], tm[-1] + 1)

    for i in range(len(cy_f_or)):
        # beam status
        if status[i] == 'on':
            axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
            axs[2].set_ylabel('Beam status On')
            plt.setp(axs[2].get_yticklabels(), visible=False)
        
        # breath-holds
        if breathholds[i] == 1:
            axs[3].axvline(x=tm[i], ymin=0, ymax=1, color='r', linewidth=4)
            axs[3].set_ylabel('Breath-holds')
            plt.setp(axs[3].get_yticklabels(), visible=False)

    if np.sum(imagepauses) > 0:
        for i in range(len(framenumb_all)):
            # image pauses
            if imagepauses[i] == 1:
                axs[4].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                axs[4].set_ylabel('Imaging paused')
                axs[4].set_xlabel('Time [s]')
                plt.setp(axs[4].get_yticklabels(), visible=False)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps: 
        # defined globally for all figures
        plt.rcParams['axes.grid'] = True
        plt.rc('xtick', labelsize=25) 
        plt.rc('ytick', labelsize=25)
        plt.rc('axes', labelsize=25)  # to change the size of the letters

        fig, axs = plt.subplots(5, sharex=True, sharey=False, figsize=(25, 20),
                                gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5, 0.5]})
        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', color='black', label='original')
        axs[0].plot(tm[:int(100 * fps)], cx_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='red', label='replaced and filtered')
        axs[0].plot(tm[:int(100 * fps)], cx_f_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='blue', label='replaced and filtered')
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black', label='original')
        axs[1].plot(tm[:int(100 * fps)], cy_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='red', label='replaced and filtered')
        axs[1].plot(tm[:int(100 * fps)], cy_f_or[:int(100 * fps)], 'ko', linestyle='-', 
                    color='blue', label='replaced and filtered')
        axs[1].set_ylabel('Inf-sup motion [mm]')  


        for i in range(int(100 * fps)):
            if status[i] == 'on':
                axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
                axs[2].set_ylabel('Beam status On')
                plt.setp(axs[2].get_yticklabels(), visible=False)

        # breath-holds
            if breathholds[i] == 1:
                axs[3].axvline(x=tm[i], ymin=0, ymax=1, color='r', linewidth=4)
                axs[3].set_ylabel('Breath-holds')
                plt.setp(axs[3].get_yticklabels(), visible=False)

            if np.sum(imagepauses) > 0:
                # image pauses
                if imagepauses[i] == 1:
                    axs[4].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                    axs[4].set_ylabel('Imaging paused')
                    axs[4].set_xlabel('Time [s]')
                    plt.setp(axs[4].get_yticklabels(), visible=False)


        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    
    plt.close()


def motion_with_status_pause_info(tm, cx, cy,
                    status, framenumb_all, imagepauses, fn, 
                    fps, display=True, save=False, path_saving=None):
    
    """" Plot motion curves plus beam status, image pauses and breathhold information. """
    
    # defined globally for all figures
    plt.rcParams['axes.grid'] = True
    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25)
    plt.rc('axes', labelsize=25)  # to change the size of the letters

    if np.sum(imagepauses) > 0:
        fig, axs = plt.subplots(4, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1,1]})
    else:
        fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1]})

    axs[0].plot(tm, cx, 'ko', linestyle='-', color='black')
    axs[0].set_ylabel('Post-ant motion [mm]')

    axs[1].plot(tm, cy, 'ko', linestyle='-', color='black')
    axs[1].set_ylabel('Inf-sup motion [mm]')  

    axs[1].set_xlim(tm[0], tm[-1] + 1)

    for i in range(len(cy)):
        # beam status
        if status[i] == 'on':
            axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=4)
            axs[2].set_ylabel('Beam status On')
            if len(imagepauses) == 0:
                axs[2].set_xlabel('Time [s]')
            plt.setp(axs[2].get_yticklabels(), visible=False)
    
    if np.sum(imagepauses) > 0:
        for i in range(len(framenumb_all)):
            # image pauses
            if imagepauses[i] == 1:
                axs[3].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=4)
                axs[3].set_ylabel('Imaging paused')
                axs[3].set_xlabel('Time [s]')
                plt.setp(axs[3].get_yticklabels(), visible=False)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info.png'), 
                    bbox_inches="tight")
    if display:
        plt.show()
        
    # if sequence particularly long, plot only first 100 seconds
    if len(tm) > 100 * fps: 
        # defined globally for all figures
        plt.rcParams['axes.grid'] = True
        plt.rc('xtick', labelsize=25) 
        plt.rc('ytick', labelsize=25)
        plt.rc('axes', labelsize=25)  # to change the size of the letters

        if len(imagepauses) > 0:
            fig, axs = plt.subplots(4, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1,1]})
        else:
            fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(25,20),gridspec_kw={'height_ratios': [2,2,1]})
            
            
        axs[0].plot(tm[:int(100 * fps)], cx[:int(100 * fps)], 'ko', linestyle='-', color='black')
        axs[0].set_ylabel('Post-ant motion [mm]')

        axs[1].plot(tm[:int(100 * fps)], cy[:int(100 * fps)], 'ko', linestyle='-', 
                    color='black')
        axs[1].set_ylabel('Inf-sup motion [mm]')  


        for i in range(int(100 * fps)):
            if status[i] == 'on':
                axs[2].axvline(x=tm[i], ymin=0, ymax=1, color='g', linewidth=6)
                axs[2].set_ylabel('Beam status On')
                if len(imagepauses) == 0:
                    axs[2].set_xlabel('Time [s]')
                plt.setp(axs[2].get_yticklabels(), visible=False)

            if len(imagepauses) > 0:
                # image pauses
                if imagepauses[i] == 1:
                    axs[3].axvline(x=framenumb_all[i] / fps, ymin=0, ymax=1, color='r', linewidth=6)
                    axs[3].set_ylabel('Imaging paused')
                    axs[3].set_xlabel('Time [s]')
                    plt.setp(axs[3].get_yticklabels(), visible=False)


        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path_saving, fn[:-4] + '_eff_motion_info_100s.png'), 
                        bbox_inches="tight")
        if display:
            plt.show()
    
    plt.close()

        

def losses_plot(train_losses=None, val_losses=None, loss_name=None,
                display=False, last_epochs=50, 
                path_saving=None):
    
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    
    if train_losses is not None:
        plt.plot(train_losses, 'o-', label="Training loss")
    if val_losses is not None:
        plt.plot(val_losses, 'o-', label="Validation loss")
    
    plt.ylabel(f"{loss_name}")
    plt.xlabel("Epoch number")
    plt.legend()    
    
    if path_saving is not None:
        plt.savefig(path_saving, bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
    
    if last_epochs is not None:
        plt.figure(figsize=(10, 7))
    
        plt.plot(train_losses[-last_epochs:], 'o-', label="Training loss")
        if val_losses is not None:
            plt.plot(val_losses[-last_epochs:], 'o-', label="Validation loss")
        
        plt.ylabel(f"{loss_name}")
        plt.xlabel("Epoch number")
        plt.legend() 
        
        if path_saving is not None:
            plt.savefig(path_saving, bbox_inches="tight")
        if display:
            plt.show()
        plt.close() 


def losses_plot_detailed(train_losses=None, val_losses=None, 
                        loss_name=None,
                        log=False,
                        display=False, 
                        save=False, path_saving=None, 
                        info_loss=''):
 
    plt.rcParams.update({'font.size': 22})   
    plt.figure(figsize=(10, 7))
    
    if log:
        # set logarithmic axis
        axs = plt.axes(yscale='log')
    else:
        axs = plt.axes()        
    
    if train_losses is not None:
        axs.plot(train_losses, '-', label="Training loss")
    if val_losses is not None:
        axs.plot(val_losses, '-', label="Validation loss")
    
    axs.set_ylabel(f"Normalized {loss_name}")
    # axs.set_ylim(min(train_losses), max(train_losses));
    axs.set_xlabel("Epoch number")
    # axs.set_xlim(-1, len(train_losses) + 1);
    
    # set legend and grid
    axs.legend()    
    axs.grid(linestyle='dashed')
    
    # set minor ticks on and ticks on both sides
    # axs.xaxis.set_major_locator(plt.MultipleLocator(1))
    # axs.yaxis.set_major_locator(plt.MultipleLocator(1))
    axs.minorticks_on()
    axs.xaxis.set_minor_locator(plt.MaxNLocator())
    axs.tick_params(labeltop=False, labelright=False)
    axs.tick_params(which='both', top=True, right=True)
    
    if log is False:
        # set scientific notation for y axis
        axs.ticklabel_format(axis='y', 
                             style='sci',
                             scilimits=(0, 0))

    if save:
        plt.savefig(os.path.join(path_saving, info_loss + 'losses.png'), bbox_inches="tight")
    if display:
        plt.show()
    plt.close()
    
           
        
def predicted_wdw_plot(x, y, y_pred, wdw_nr=-1, last_pred=True,
                       display=True, path_saving=None):
    """ Plot ground truth vs predict time series window.

    Args:
        x (Pytorch tensor or np.array): ground truth input windows, shape = [batch_size, wdw_size_o] 
        y (Pytorch tensor or np.array): ground truth output windows, shape = [batch_size, wdw_size_o] 
        y_pred (Pytorch tensor): predicted output windows, shape = [batch_size, wdw_size_o] 
        wdw_nr (int, optional): window nr in list with windows to be plotted. Defaults to -1.
        last_pred (bool, optional): whether to plot only the last prediction. Defaults to True.
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
    
    # take last wdw, changing shape to (wdw_size_i,)
    x = x[wdw_nr, ...]  
    y = y[wdw_nr, ...]
    y_pred = y_pred[wdw_nr, ...]    

    # convert tensor to numpy arrays on cpu 
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise Exception('Attention: unknown type')
    
    # create time axis
    t = np.arange(len(x) + len(y))
    
    # print(np.shape(x))  # (8,)
    # print(np.shape(y))  # (1,)
    # print(np.shape(y_pred))  # (1,)
  
    plt.figure(figsize=(10, 7))  
    plt.plot(t[:len(x)], x, 'o-', color='black', label="True input")
    plt.plot(t[len(x):len(x) + len(y)], y, 'o-', color='blue', label="True output")
    
    if last_pred:
        plt.plot(t[len(x) + len(y) - 1], y_pred[-1], 'o-', color='red', label="Predicted output")
    else:
        plt.plot(t[len(x):len(x) + len(y)], y_pred, 'o-', color='red', label="Predicted output")

    
    plt.ylabel("Relative amplitude")
    plt.xlabel("Time step")
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid()
    
    plt.title(f"Ground truth and predicted sequences")
    
    if path_saving is not None:
        plt.savefig(path_saving, bbox_inches="tight")
    if display:
        plt.show() 
    plt.close()
            
        
        
def predicted_snippets_plot(y_pred, y_batch, normalization=True, 
                            first_points=None, last_points=None,
                            display=True, save=False, path_saving=None):
    """ Plot ground truth vs predict time series.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred (list of Pytorch tensors or np.array): predicted output series
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
  
    if torch.is_tensor(y_pred[0]):
        # get tensor out of list of tensors
        y_pred = torch.stack(y_pred)
        y_batch = torch.stack(y_batch)
        # get numpy arrays on CPU
        y_pred = y_pred.detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()
    elif isinstance(y_pred[0], np.ndarray):
        # get array out of list of arrays
        y_pred = np.concatenate(y_pred)
        y_batch = np.concatenate(y_batch)
    else:
        print(y_pred[0])
        raise Exception('Attention: unknown type')

    # create time axis
    t = np.arange(len(y_pred)) / 4
    

    # print(np.shape(y_pred))  # (324, 1)
    # print(np.shape(y_batch))  # (324, 1)
  
    plt.figure(figsize=(10, 7))  
    if last_points is not None:
        plt.plot(t[:last_points], y_batch[-last_points:], 'o-', color='black', label="True")
        plt.plot(t[:last_points], y_pred[-last_points:], '*-', color='blue', label="Predicted")
    elif first_points is not None:
        plt.plot(t[:first_points], y_batch[:first_points], 'o-', color='black', label="True")
        plt.plot(t[:first_points], y_pred[:first_points], '*-', color='blue', label="Predicted")
    else:
        plt.plot(t, y_batch, 'o-', color='black', label="True")
        plt.plot(t, y_pred, '*-', color='blue', label="Predicted")
        
            
    if normalization:
        plt.ylabel("Relative amplitude")
        plt.ylim([-1, 1])
    else:
        plt.ylabel("Amplitude [mm]")      
          
    plt.xlabel("Time [s]")
    plt.legend()
    
    plt.title(f"Ground truth vs. predicted snippets")
    
    if save:
        plt.savefig(os.path.join(path_saving, 
                                 f'predicted_snippets_norm{normalization}_last_points{last_points}_first_points{first_points}.png'), 
                    bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()


def predicted_snippets_comparison(y_pred_1, y_pred_2, y_batch, 
                            video_nr, 
                            normalization=True, 
                            first_points=None, last_points=None,
                            display=True, save=False, 
                            legend=True,
                            info='', path_saving=None,
                            fs=16):
    """ Plot ground truth vs predict time series for two different models.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred_1 (list of Pytorch tensors or np.array): predicted output series for LR model
        y_pred_2 (list of Pytorch tensors or np.array): predicted output series for LSTM model
        video_nr (int): snippet number picked for plotting
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        legend (bool, optional): whether to plot legend or not. Defaults to True
        info (str): additional info for name of saved plot  
        path_saving ([type], optional): path where plot is saved. Defaults to None.
        fs (int): fontsize for labels,legend etc.
    """
    # change fontsize for label, legend etc.
    plt.rcParams.update({'font.size': fs})
    
    if torch.is_tensor(y_pred_1[0]):
        # get tensor out of list of tensors
        y_pred_1 = torch.stack(y_pred_1)
        # get numpy arrays on CPU
        y_pred_1 = y_pred_1.detach().cpu().numpy()
    elif isinstance(y_pred_1[0], np.ndarray):
        # get array out of list of arrays
        y_pred_1 = np.concatenate(y_pred_1)
    else:
        print(y_pred_1[0])
        raise Exception('Attention: unknown type for y_pred_1')
    
    if torch.is_tensor(y_pred_2[0]):
        # get tensor out of list of tensors
        y_pred_2 = torch.stack(y_pred_2)
        # get numpy arrays on CPU
        y_pred_2 = y_pred_2.detach().cpu().numpy()
    elif isinstance(y_pred_2[0], np.ndarray):
        # get array out of list of arrays
        y_pred_2 = np.concatenate(y_pred_2)
    else:
        print(y_pred_2[0])
        raise Exception('Attention: unknown type for y_pred_2')
    
    
    if torch.is_tensor(y_batch[0]):
        # get tensor out of list of tensors
        y_batch = torch.stack(y_batch)
        # get numpy arrays on CPU
        y_batch = y_batch.detach().cpu().numpy()
    elif isinstance(y_batch[0], np.ndarray):
        # get array out of list of arrays
        y_batch = np.concatenate(y_batch)
    else:
        print(y_batch[0])
        raise Exception('Attention: unknown type for y_batch')
    
    # create time axis
    t = np.arange(len(y_pred_1)) / 4
    
    # print(np.shape(y_pred_1))  # (324, 1)
    # print(np.shape(y_batch))  # (324, 1)
  
    plt.figure(figsize=(10, 7))  
    if last_points is not None:
        plt.plot(t[:last_points], y_batch[-last_points:], 'o-', color='black', label="True")
        plt.plot(t[:last_points], y_pred_1[-last_points:], '*--', color='blue', label="LR")
        plt.plot(t[:last_points], y_pred_2[-last_points:], 'd--', color='red', label="LSTM")
    elif first_points is not None:
        plt.plot(t[:first_points], y_batch[:first_points], 'o-', color='black', label="True")
        plt.plot(t[:first_points], y_pred_1[:first_points], '*--', color='blue', label="LR")
        plt.plot(t[:first_points], y_pred_2[:first_points], 'd--', color='red', label="LSTM")    
    else:
        start = 0
        stop = 57
        plt.plot(t[start:stop], y_batch[start:stop], 'o-', color='black', label="True")
        plt.plot(t[start:stop], y_pred_1[start:stop], '*--', color='blue', label="LR")        
        plt.plot(t[start:stop], y_pred_2[start:stop], 'd--', color='red', label="LSTM")    
           
    if normalization:
        plt.ylabel("Relative amplitude")
        plt.ylim([-1, 1])
    else:
        plt.ylabel("Amplitude [mm]")      
          
    plt.xlabel("Time [s]")
    if legend:
        plt.legend()
    plt.grid()
    
    
    if save:
        if (first_points is None) and (last_points is None):
            plt.savefig(os.path.join(path_saving, 
                                    f'predicted_snippet{video_nr}_norm{normalization}_start{start}_stop{stop}_{info}.png'), 
                        bbox_inches="tight")
        else:            
            plt.savefig(os.path.join(path_saving, 
                                    f'predicted_snippet{video_nr}_norm{normalization}_last_points{last_points}_first_points{first_points}_{info}.png'), 
                        bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()


def diff_violinplot(y_pred_1, y_pred_2, y_batch,
                            display=True, save=False, 
                            info='', path_saving=None):
    """ Plot ground truth vs predict time series.

    Args:
        y_batch (list of Pytorch tensors or or np.array): ground truth output series
        y_pred (list of Pytorch tensors or np.array): predicted output series
        first_points(int, optional): whether to plot only specified nr of first time points
        last_points(int, optional): whether to plot only specified nr of last time points
        display (bool, optional): whether to dislpay plot (works with Ipython). Defaults to True.
        save (bool, optional): whether to save plot. Defaults to False.
        info (str): additional info for name of saved plot          
        path_saving ([type], optional): path where plot is saved. Defaults to None.
    """
    diff_true_1 = np.array([])
    diff_true_2 = np.array([])
    
    for y_pred_case_1, y_pred_case_2, y_batch_case in zip(y_pred_1, y_pred_2, y_batch):   
        if torch.is_tensor(y_pred_case_1[0]):
            # get tensor out of list of tensors
            y_pred_case_1 = torch.stack(y_pred_case_1)
            # get numpy arrays on CPU
            y_pred_case_1 = y_pred_case_1.detach().cpu().numpy()
        elif isinstance(y_pred_case_1[0], np.ndarray):
            # get array out of list of arrays
            y_pred_case_1 = np.concatenate(y_pred_case_1)[:, np.newaxis] 
        else:
            print(y_pred_case_1[0])
            raise Exception('Attention: unknown type for y_pred_case_1')
        
        if torch.is_tensor(y_pred_case_2[0]):
            # get tensor out of list of tensors
            y_pred_case_2 = torch.stack(y_pred_case_2)
            # get numpy arrays on CPU
            y_pred_case_2 = y_pred_case_2.detach().cpu().numpy()
        elif isinstance(y_pred_case_2[0], np.ndarray):
            # get array out of list of arrays
            y_pred_case_2 = np.concatenate(y_pred_case_2)[:, np.newaxis] 
        else:
            print(y_pred_case_2[0])
            raise Exception('Attention: unknown type for y_pred_case_2')
        
        if torch.is_tensor(y_batch_case[0]):
            # get tensor out of list of tensors
            y_batch_case = torch.stack(y_batch_case)
            # get numpy arrays on CPU
            y_batch_case = y_batch_case.detach().cpu().numpy()
        elif isinstance(y_batch_case[0], np.ndarray):
            # get array out of list of arrays
            y_batch_case = np.concatenate(y_batch_case)[:, np.newaxis] 
        else:
            print(y_batch_case[0])
            raise Exception('Attention: unknown type for y_batch_case')
        
        # print(f'shape y_pred_case_1: {np.shape(y_pred_case_1)}')  # (528, 1)
        # print(f'shape y_pred_case_2: {np.shape(y_pred_case_2)}')  # (528, 1)
        # print(f'shape y_batch_case: {np.shape(y_batch_case)}')  # (528, 1)

        # build difference between ground truth and predicted curves for video
        current_diff_true_1 = np.subtract(y_batch_case, y_pred_case_1)
        current_diff_true_2 = np.subtract(y_batch_case, y_pred_case_2)
        
        # append to array with all differences
        diff_true_1 = np.append(diff_true_1, current_diff_true_1)
        diff_true_2 = np.append(diff_true_2, current_diff_true_2)
    
    # print(f'shape diff_true_1: {np.shape(diff_true_1)}')  # (13359,)
    # print(f'shape diff_true_2: {np.shape(diff_true_2)}')  # (13359,)

    # do violin plot of signed difference between ground truth and predicted curves
    boxdata = [sorted(diff_true_1), sorted(diff_true_2)] 

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    vp = ax.violinplot(boxdata, showmeans=False, showmedians=True, showextrema=True)
    for pc in vp['bodies']:
        # pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    q1_1, med_1, q3_1 = np.percentile(diff_true_1, [25, 50, 75], axis=0)  
    q1_2, med_2, q3_2 = np.percentile(diff_true_2, [25, 50, 75], axis=0)  

    
    quartiles1, medians, quartiles3 = [q1_1, q1_2], [med_1, med_2], [q3_1, q3_2]   

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(inds, quartiles1, quartiles3, color='k', linestyle='-', lw=10)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='green', linestyle='-', lw=3)

    # set style for the axes
    labels = ['True - LR', 'True - LSTM']
    for ax in [ax]:
        set_axis_style(ax, labels)

    # plt.title(" Target motion", fontsize=25)
    plt.ylabel("Difference between ground truth and prediction [mm]")
    plt.grid(True)

    if save:
        plt.savefig(os.path.join(path_saving, 
                                 f'difference_violinplots_{info}.png'), 
                    bbox_inches="tight")
    if display:
        plt.show() 
        
    plt.close()
    


# %%
