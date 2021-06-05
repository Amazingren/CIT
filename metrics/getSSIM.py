import os
from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import torch
from torchvision import transforms
import lpips

import numpy as np
import pandas as pd

from tqdm import tqdm
import re


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def test(generated_IMG_dir, reference_IMG_dir):
    print(generated_images_dir, reference_IMG_dir)
    print ("Loading image Pairs...")

    generated_images = []
    for img_nameG in os.listdir(generated_IMG_dir):
        imgG = imread(os.path.join(generated_IMG_dir, img_nameG))
        generated_images.append(imgG)

    reference_images = []
    for img_nameR in os.listdir(reference_IMG_dir):
        imgR = imread(os.path.join(reference_IMG_dir, img_nameR))
        reference_images.append(imgR)

    print("#####SSIM######")
    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, reference_images)
    print ("SSIM score %s" % structured_score)



if __name__ == "__main__":
    generated_images_dir = '/home/data/try-on/GMM/0309_03/result/TOM/test/try-on'
    reference_images_dir = '/home/data/try-on/GMM/0309_03/data/test/image'

    test(generated_images_dir, reference_images_dir)
