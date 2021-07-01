#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Changmin Yu
'''

import matplotlib.pyplot as plt
import numpy as np

from utils import gaussianPlaceField, constructOneStepTransmat2D
from fourier_utils import fourierMat, circulantEvec, getDFT_evals

def weightOrthogonal(place_field, init_pos=None):
    n = place_field.shape[0]
    f_modes = np.array([circulantEvec(n, i) for i in range(n)])
    f_coeff = getDFT_evals(place_field)
    if init_pos is None:
        init_post = np.random.choice(n)
    init_place_field = place_field[init_pos]
    weights = np.zeros((n, ), dtype=np.complex128)
    for i in range(n):
        weights[i] = init_place_field.dot(np.conj(f_modes[:, i]))
    return weights, init_pos, f_modes, f_coeff

def constructTransmat(H, W, var=None, offset=None):
    transmat = np.zeros((H*W, H*W))
    for i in range(H):
        for j in range(W):
            a = gaussianPlaceField(i, j, H, W, offset, var=var)
            a = a/np.sum(a)
            transmat[i*W+j, :] = a
    return transmat

def reconstructPlaceField(H, W, place_field, init_pos=None):
    if init_pos is None:
        init_pos_ind = np.random.choice(H*W)
    init_pos_ind = int(init_pos[0] * W + init_pos[1])
    weights, init_pos, f_modes, _ = weightOrthogonal(place_field, init_pos_ind)
    reconstructed = np.real(np.sum(f_modes * weights, axis=1)).reshape(H, W)
    return reconstructed, init_pos

def predictPlaceField(H, W, place_field, init_pos=None, trans_offset=None, timestep=1, var=None, predict_diffusion=False):
    if trans_offset is None:
        trans_offset = np.array([0., 0.])
    if init_pos is None:
        init_pos_ind = np.random.choice(H*W)
    init_pos_ind = int(init_pos[0] * W + init_pos[1])
    weights, init_pos, f_modes, _ = weightOrthogonal(place_field, init_pos_ind)
    if predict_diffusion:
        transmat = constructTransmat(H, W, var, trans_offset)
    else:
        transmat = constructOneStepTransmat2D(H, W, trans_offset)
    dft_trans = getDFT_evals(transmat)
    predicted = []
    if isinstance(timestep, np.ndarray):
        for t in timestep:
            predicted.append(np.real(np.sum(np.power(dft_trans, t) * weights * f_modes, axis=1)).reshape(H, W))
    else:
        predicted.append(np.real(np.sum(np.power(dft_trans, timestep) * weights * f_modes, axis=1)).reshape(H, W))
    return predicted

def approximateDistanceSR(H, W, target_state=None, var=None, plot=False):
    dft_mat = fourierMat(H*W)
    if target_state is None:
        target_state = np.array([np.random.choice(H), np.random.choice(W)])
    target_ind = target_state[0] * W + target_state[1]
    sym_transmat = constructTransmat(H, W, var)
    dft_sym = getDFT_evals(sym_transmat)
    dist = np.log(np.real(dft_mat.dot(np.exp(np.diag(dft_sym))).dot(np.linalg.inv(dft_mat)))[target_ind])

    if plot:
        m_grid = np.meshgrid(np.arange(W), np.arange(H))
        coords = np.stack([m_grid[0].ravel(), m_grid[1].ravel()])
        coords_x = coords[0]
        coords_y = coords[1]
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.plot_trisurf(coords_x, coords_y, -dist, linewidth=0.2, antialiased=True)
        plt.show()
    return dist

if __name__=='__main__':
    import matplotlib.pyplot as plt
    P = constructTransmat(40, 60, offset=np.array([20, 0]))
    predictions = predictPlaceField(40, 60, P, init_pos=np.array([5, 10]), \
        trans_offset=np.array([0, 5]), timestep=np.array([1, 2, 3, 4, 5]))
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        ax[i].imshow(predictions[i])