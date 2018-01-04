# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:57:30 2017

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
for filename in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/New folder/sample"):
    location="C:/Deepti Chevvuri/Hackathon/DSM/New folder/sample/"+filename
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
'''


print('testing with new image:')
for subdir in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/New folder/test"):
    for filename in os.listdir("C:/Deepti Chevvuri/Hackathon/DSM/New folder/test/"+subdir):
        location="C:/Deepti Chevvuri/Hackathon/DSM/New folder/test/"+subdir+"/"+filename
        img=Image.open(location)
        testing=grayscaling(img)
        testimage= img_as_float(testing)[shrink]
        print('original: '+subdir+', rotated: 0deg, match result: ', end='')
        feats = compute_feats(ndi.rotate(testimage, angle=0, reshape=False), kernels)
        print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
'''
# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
