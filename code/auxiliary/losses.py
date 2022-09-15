# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import monai

# %%

def gradient_loss(inputs, alpha=2):
    # from https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/5 

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        # print(f'left.shape: {left.shape} ')  # (320, 2, 64, 64)

        dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row, bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # get gradients
    dx, dy = gradient(inputs)
    # print(inputs[0, 0, :,:])
    # print(dx[0, 0, :,:])
    # print(dy[0, 0, :,:])

    # condense into one tensor and avg
    return torch.sum(dx ** alpha + dy ** alpha)


def combined_image_dvf_loss(y_pred, y_true, deformations, lambda_image=1.0, lambda_dvf=0.0):
    """Combined dice loss on contours and gradient loss on deformation vector fields output
        bz ConvLSTMSTL.

    Args:
        y_pred (tensor): predicted contours for 250 ms and 500 ms forecast.
        y_true (tensor): ground truth contours for 250 ms and 500 ms forecast.
        deformations (tensor): deformation vector fields.
        lambda_image (int, optional): weight of dice loss. Defaults to 1.
        lambda_dvf (int, optional): weight of gradient loss. Defaults to 0.
    """
    # image_loss = nn.MSELoss()(y_pred, y_true)
    # image_loss = nn.BCELoss()(y_pred, y_true)
    image_loss = (monai.losses.DiceLoss(include_background=True, sigmoid=False, reduction="mean")(y_pred[:,0,...], y_true[:,0,...]) + \
                    monai.losses.DiceLoss(include_background=True, sigmoid=False, reduction="mean")(y_pred[:,1,...], y_true[:,1,...]))/2
    # print(f'image loss: {image_loss}')
    dvf_loss = gradient_loss(deformations)
    # print(f'gradient loss: {dvf_loss}')
    
    return (lambda_image * image_loss + lambda_dvf * dvf_loss)


def compute_centroid(image, device):
    """Compute centroid in height and width for batch of 2D images in a differentiable way.

    Args:
        image (tensor): input image of shape (*dim, height, width)
    """
    
    # get indices for height dimension
    idx_height = torch.arange(256, dtype=torch.float32).to(device).detach()
    # compute the mean value weighted sum of the indices 
    # to get the centroids of one batch, ie dimension (*dim)
    centroid_height_batch_different_widths = torch.matmul(torch.transpose(image, -2,-1), idx_height)
    denominator_height = torch.sum(image, axis=(-2,-1)) + 1e-8  # add small constant to denominator to avoid nan
    centroid_height_batch = torch.sum(centroid_height_batch_different_widths, axis=-1)/denominator_height
    
    # get indices for width dimension
    idx_width = torch.arange(256, dtype=torch.float32).to(device).detach()
    # compute the mean value weighted sum of the indices 
    # to get the centroids of one batch, ie dimension (*dim)
    centroid_width_batch_different_heights = torch.matmul(image, idx_width)
    denominator_width = torch.sum(image, axis=(-2,-1)) + 1e-8  # add small constant to denominator to avoid nan
    centroid_width_batch = torch.sum(centroid_width_batch_different_heights, axis=-1)/denominator_width

    return torch.cat((centroid_height_batch, centroid_width_batch), dim=-1)


def centroid_mse_loss(y_pred, y_true, device):
    """Compute MSE loss between predicted and ground truth centroids derived from contours.

    Args:
        y_pred (tensor): predicted contours of shape b,s,c,h,w.
        y_true (tensor): ground truth contours of shape b,s,c,h,w.
    """   
    centroids_pred = compute_centroid(y_pred, device)
    # print(centroids_pred)
    centroids_true = compute_centroid(y_true, device)
    # print(centroids_true)
    
    loss = nn.MSELoss(reduction="mean")(centroids_pred, centroids_true)
    
    return loss
    
    
def combined_image_centroid_loss(y_pred, y_true, device, lambda_image=0.5, lambda_centroid=0.5):
    """Combined dice loss on contours and gradient loss on deformation vector fields output
        by ConvLSTMSTL.

    Args:
        y_pred (tensor): predicted contours for 250 ms and 500 ms forecast of shape b,s=2,c,h,w.
        y_true (tensor): ground truth contours for 250 ms and 500 ms forecast of shape b,s=2,c,h,w.
        lambda_image (int, optional): weight of dice loss. Defaults to 0.99.
        lambda_centroid (int, optional): weight of centroid loss. Defaults to 0.01.
    """
    # image_loss = nn.MSELoss()(y_pred, y_true)
    # image_loss = nn.BCELoss()(y_pred, y_true)
    image_loss = (monai.losses.DiceLoss(include_background=True, sigmoid=False, reduction="mean")(y_pred[:,0,...], y_true[:,0,...]) + \
                    monai.losses.DiceLoss(include_background=True, sigmoid=False, reduction="mean")(y_pred[:,1,...], y_true[:,1,...]))/2
    # print(f'image loss: {image_loss}')
    centroid_loss = centroid_mse_loss(y_pred, y_true, device)
    # print(f'gradient loss: {centroid_loss}')
    
    return (lambda_image * image_loss + lambda_centroid * centroid_loss)
    #return (lambda_image * image_loss)

# %%
