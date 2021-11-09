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


def get_DoG(img, octave_level=4, scale_level=5, g_size=3, sigma=1.6):
    sigma = 2**(1/octave_level)
    pyr = get_gaussian_pyr(img, octave_level=octave_level, scale_level=scale_level, g_size=g_size, sigma=sigma)
    DoG = []
    for octave in range(len(pyr)):
        oct_diff = []
        for i in range(1, len(pyr[octave])):
            diff = pyr[octave][i-1] - pyr[octave][i]
            oct_diff.append(diff)
        DoG.append(oct_diff)
    return np.array(DoG, dtype=object)


def compute_h(p1, p2):
    A = []
    for n in range(len(p1)):
        A.append(
            [p2[n][0], p2[n][1], 1, 0,0,0, -p1[n][0]*p2[n][0], -p1[n][0]*p2[n][1], -p1[n][0]]
        )
        A.append(
            [0,0,0, p2[n][0], p2[n][1], 1, -p1[n][1]*p2[n][0], -p1[n][1]*p2[n][1], -p1[n][1]]
            )
    A = np.array(A)
    U, s, V = np.linalg.svd(A)
    h = V.T[:, -1]
    H = np.reshape(h, (3,3))
    return H


def compute_h_norm(p1, p2):
    m1 = np.mean(p1, axis=0)
    m2 = np.mean(p2, axis=0)
    s1 = np.std(p1, axis=0)
    s2 = np.std(p2, axis=0)
    N1 = np.array([[1/s1[0], 0, -m1[0]/s1[0]],[0,1/s1[1],-m1[1]/s1[1]], [0,0,1]])
    N2 = np.array([[1/s2[0], 0, -m2[0]/s2[0]],[0,1/s2[1],-m1[1]/s2[1]], [0,0,1]])
    p1h = []
    p2h = []
    for pp1, pp2 in zip(p1, p2):
        p1h.append(np.matmul(N1, np.array([pp1[0], pp1[1], 1])))
        p2h.append(np.matmul(N2, np.array([pp2[0], pp2[1], 1])))
    
    normed_H = compute_h(p1h, p2h)
    H = np.matmul(np.linalg.inv(N1), np.matmul(normed_H, N2))
    return H


def main():
    pass


if __name__ == '__main__':
    main()
