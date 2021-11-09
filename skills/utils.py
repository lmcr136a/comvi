import math
import numpy as np
from PIL import Image


def convolution(Igs, G):
    Iconv = np.zeros(Igs.shape)
    ph, pw = G.shape
    pad_size = ph // 2

    upper = Igs[:1][::-1]*pad_size
    bottom = Igs[-1:][::-1]*pad_size
    padded_img = np.concatenate((upper, Igs, bottom), axis=0)

    left = padded_img[:, :1][:,::-1]*pad_size
    right = padded_img[:, -1:][:, ::-1]*pad_size
    padded_img = np.concatenate((left, padded_img, right), axis=1)

    kernel_shape = np.lib.stride_tricks.as_strided(
        padded_img, 
        shape=tuple(np.subtract(padded_img.shape, G.shape) + 1) + G.shape, 
        strides=padded_img.strides*2
        )
    Iconv = np.einsum('ij,klij->kl', G, kernel_shape)
    return Iconv


def gaussian(Igs, g_size=3, sigma=2):
    ax = np.linspace(-(g_size - 1) / 2., (g_size - 1) / 2., g_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)
    blurred_im = convolution(Igs.copy(), kernel)
    return blurred_im


def get_gaussian_pyr(img, octave_level=4, scale_level=5, g_size=3, sigma=1.6):
    pyr = [[]]*octave_level
    for i in range(octave_level):
        processing_img = img
        pyr[i] = [processing_img]
        for scale in range(scale_level-1):
            processing_img = gaussian(processing_img, g_size=g_size, sigma=sigma)
            pyr[i] += [processing_img]
            
        img = Image.fromarray(img)
        img = img.resize((img.size[0]//2, img.size[1]//2))
        img = np.asarray(img)
    return pyr