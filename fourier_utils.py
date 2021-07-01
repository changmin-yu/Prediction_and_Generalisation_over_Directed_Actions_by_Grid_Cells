#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Changmin Yu
'''

import numpy as np

from utils import sort_complex_amplitude

def fourierMat(n):
    return np.array([[np.exp(2*np.pi*complex(0, 1)/n*j*k) for j in range(n)] for k in range(n)])

def circulantEvec(n, k):
    return np.array([np.exp(2*np.pi*complex(0, 1)*j*k/n) for j in range(n)])

def getDFT_evals(T, sort=False, sort_order='descent'):
    num_state = T.shape[0]
    fourier_mat = fourierMat(num_state)
    dft = fourier_mat @ T[0, :]
    if sort:
        return sort_complex_amplitude(dft, sort_order)
    return dft

def compute_one_dft(H, W, u, v):
    xy = np.zeros((H, W), dtype=np.complex)
    for i in range(H):
        for j in range(W):
            xy[i, j] = np.exp(complex(0, 1)*np.pi*2*(u*i/H+v*j/W))
    return xy

def compute_dft_basis(H, W):
    dft_mat = np.zeros((H, W, H, W), dtype=np.complex)
    for i in range(H):
        for j in range(W):
            dft_mat[i, j] = compute_one_dft(H, W, i, j)
    return dft_mat

def compute_dft_mag(H, W):
    dft_mag = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            dft_mag[i, j] = np.sqrt(min(i, H-i)**2 + min(j, W-j)**2)
    return dft_mag

def compute_one_dft_angs(H, W, u, v):
    xy = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            xy[i, j] = np.pi*2*(u*i/H + v*j/W)
    return xy

def compute_dft_angs(H, W):
    dft_angs = np.zeros((H, W, H, W))
    for i in range(H):
        for j in range(W):
            dft_angs[i, j] = compute_one_dft_angs(H, W, i, j)
    return np.mod(dft_angs+np.pi, 2*np.pi)-np.pi
