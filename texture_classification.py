# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:29:58 2017

@author: dchevvur
"""
from __future__ import print_function
import numpy as np
import PIL
from PIL import Image
from scipy import ndimage as ndi
import os
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats
 

def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

def grayscaling(img):
    width,height=img.size
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img = img.crop(
            (
                    half_the_width - 50,
                    half_the_height + 30,
                    half_the_width + 50,
                    half_the_height + 90
                    )
            )
    img = img.resize((512, 512), PIL.Image.ANTIALIAS)
    img=img.convert('L')
    return img

def main():
    print("hi")
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


    shrink = (slice(0, None, 3), slice(0, None, 3))
    image_names= list()
    images= list()
    filecount=0
    for filename in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/2017 Sogeti Hackathon/2017 Sogeti Hackathon"):
        location="C:/Deepti Chevvuri/Hackathon/DSM/2017 Sogeti Hackathon/2017 Sogeti Hackathon/"+filename
        filecount+=1
        img=Image.open(location)
        img=grayscaling(img)
        testing= img_as_float(img)[shrink]
        images.append(testing)
        image_names.append(os.path.splitext(filename)[0])
    ref_feats = np.zeros((filecount, len(kernels), 2), dtype=np.double)
    intcount=0
    for eachImage in images:
        ref_feats[intcount, :, :] = compute_feats(eachImage, kernels)
        intcount+=1



    print('testing with new image:')
    for subdir in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/2017 Sogeti Hackathon/Test"):
        for filename in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/2017 Sogeti Hackathon/Test/"+subdir):
            location="C:/Deepti Chevvuri/Hackathon/DSM/2017 Sogeti Hackathon/Test/"+subdir+"/"+filename
            img=Image.open(location)
            testing=grayscaling(img)
            testimage= img_as_float(testing)[shrink]
            print('original: '+subdir+', rotated: 0deg, match result: ', end='')
            feats = compute_feats(ndi.rotate(testimage, angle=0, reshape=False), kernels)
            print(image_names[match(feats, ref_feats)])

if __name__ == "__main__": main()