#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

from scipy.ndimage import minimum_filter, maximum_filter

def morph_fn(bin_img, ksize=13, operation='erode'):
    """
    对图像进行膨胀或腐蚀操作。

    参数:
    - bin_img: 二值图像 (numpy 数组)
    - ksize: 内核大小 (默认为13)
    - operation: 操作类型, 'dilate' 为膨胀, 'erode' 为腐蚀

    返回:
    - 处理后的图像
    """
    pad = (ksize - 1) // 2
    padded_img = np.pad(bin_img, pad_width=pad, mode='reflect')
    
    if operation == 'dilate':
        result_img = maximum_filter(padded_img, size=ksize, mode='constant')
    elif operation == 'erode':
        result_img = minimum_filter(padded_img, size=ksize, mode='constant')
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return result_img[pad:-pad, pad:-pad]