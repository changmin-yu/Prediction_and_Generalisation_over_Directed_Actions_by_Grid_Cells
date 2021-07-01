#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Changmin Yu
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from predictive_reconstruction import constructTransmat, predictPlaceField
from SR_navigation import collateNSWE_SRMeasure, adjustSR_barrier
from path_integration import straight_run_theta

if __name__=='__main__':
    P = constructTransmat(40, 60, offset=np.array([20, 0]))
    predictions = predictPlaceField(40, 60, P, init_pos=np.array([5, 10]), \
        trans_offset=np.array([0, 5]), timestep=np.array([1, 2, 3, 4, 5]))
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        ax[i].imshow(predictions[i])

    overall_measure = collateNSWE_SRMeasure(10, 10, np.array([2, 8]), np.array([2, 0]), var=2)
    plt.imshow(overall_measure)

    SR_barrier = adjustSR_barrier(10, 10)
    plt.imshow(SR_barrier)

    plt.figure()
    plt.ylim(0, 50)
    cmap = plt.cm.viridis
    l = np.array([[4, 1], [1, 4], [3, -3], [-4, -1], [-1, -4], [-3, 3]])
    l = np.mod(l+50, 50)
    for i in range(-50, 50):
        gc_spike_1, n_spikes_1, rec, pref_dirs = straight_run_theta(50, 50, np.array([1, i]), np.array([49, i+16]), 300, 1, 0, l, 0.2, 2.95)
        plt.plot([1, 49], [i, i+16], '--', alpha=0.4, color='black')
        if n_spikes_1 == 0:
            continue
        norm = matplotlib.colors.Normalize(vmin=np.min(gc_spike_1[:n_spikes_1, 2]), vmax=np.max(gc_spike_1[:n_spikes_1, 2]))
        spike_color = gc_spike_1[:n_spikes_1, 2]
        plt.scatter(gc_spike_1[1:n_spikes_1, 0], gc_spike_1[1:n_spikes_1, 1], c=cmap(norm(gc_spike_1[1:n_spikes_1, 2]), alpha=1))

