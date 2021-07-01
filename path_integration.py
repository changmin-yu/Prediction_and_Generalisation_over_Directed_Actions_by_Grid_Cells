#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Changmin Yu
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from scipy.io import loadmat

from utils import RGBToPyCmap, turbo_colormap_data, getComplexAng
from fourier_utils import compute_dft_basis

def velocityControlledOscillators(x1, x2, y1, y2, step, spatial_freq, speed, init_phase, n_hdcs, dirs, thresh, ax=None):
    w = 2*np.pi*spatial_freq
    w0 = 2*np.pi*8
    
    width, length = len(np.arange(x1, x2, step)), len(np.arange(y1, y2, step))
    
    coords = np.meshgrid(np.arange(x1, x2, step), np.arange(y1, y2, step))
    pos = np.stack([coords[0].ravel(), coords[1].ravel()], axis=-1)
    
    
    m = np.arange(n_hdcs)
    hd_pref_dir = (2*np.pi/n_hdcs) * m
    
    time = np.sqrt(np.sum(pos * pos, axis=-1))/speed
    theta_phase = w0 * time
    
    bc_amp = np.zeros((len(dirs), len(pos), 1))
    
    for i in range(len(dirs)):
        unit_vec = np.array([np.cos(hd_pref_dir[dirs[i]]), np.sin(hd_pref_dir[dirs[i]])])
        phase = init_phase[i] + theta_phase + w*(pos@unit_vec)
        amp = (2+np.cos(phase) + np.cos(theta_phase))/2
        amp = amp - thresh
        amp[amp < 0] = 0
        bc_amp[i, :, :] = amp.reshape(-1, 1)
    
    inp = np.prod(bc_amp, 0)
    inp = inp.reshape(width, length)
    
    if ax is None:
        ax = plt.gca()
    ax.imshow(inp, cmap='jet')
    return inp

def bandsPhaseFromFourier(width, length, positions, pos_sample_rate, spatial_scale, directional, theta_mod, dirs, time_const, threshold, boxcar, ax=None):
    debug = 0
    theta_freq = 8
    
    positions = positions[np.where(positions[:, 0] < 49)[0]]
    positions = positions[np.where(positions[:, 1] < 49)[0]]
    
    if positions is None:
        raise(ValueError('Position is empty'))
    
    beta = 2*np.pi/spatial_scale
    w0 = 2*np.pi*theta_freq
    pos_samples_per_cycle = np.int8(np.ceil(pos_sample_rate/theta_freq))
    time_step = 1/pos_sample_rate
    n_spikes = 0
    theta_phase = 0
    n_pos = len(positions)
    pposn = np.zeros((np.int16(np.ceil(np.max(positions[:, 1])))+1, np.int16(np.ceil(np.max(positions[:, 0])))+1))

    n_dirs = len(dirs)
    
    bc_spikes = np.zeros((n_pos, n_dirs))
    
    bc_phase_record = np.zeros((n_pos, n_dirs))
    gc_rate = np.zeros((np.int16(np.ceil(np.max(positions[:, 1])))+1, np.int16(np.ceil(np.max(positions[:, 0])))+1))
    gc_spike = np.zeros((n_pos, 5))

    D = compute_dft_basis(width, length)

    band_waves = np.array([D[dirs[i][0], dirs[i][1]].flatten() for i in range(len(dirs))])
    pref_dirs = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        wv = dirs[i]
        wv[wv > int(width/2)] = wv[wv> int(width/2)] - width
        pref_dirs[i] = np.arctan2(wv[1], wv[0])
    init_pos = np.int16(np.round(positions[0, 0]+positions[0, 1]*width))
    bc_phase = getComplexAng(band_waves[:, init_pos])
    bc_spikes[0, :] = 0
    
    if time_const > 0:
        exp_window = np.exp(-(pos_samples_per_cycle-np.arange(1, pos_samples_per_cycle+1))/time_const/pos_samples_per_cycle)
    else:
        exp_window = np.ones((pos_samples_per_cycle,))
    
    temp_sum = np.zeros((n_dirs, 1))
    theta_phases = np.zeros((n_pos))
    
    for k in range(1, n_pos):
        x = positions[k, 0]
        y = positions[k, 1]
        x_ind = np.int(np.round(x))
        y_ind = np.int(np.round(y))
        pposn[y_ind, x_ind] += 1
        
        curr_pos = np.int(np.round(x_ind+y_ind*length))
        
        step = positions[k, :] - positions[k-1, :]
        delta = step[0]*width + step[1]
        shift_angs = np.array([-2*np.pi*(dirs[i][0]*step[0]/width + dirs[i][1]*(length-step[1])/length) for i in range(len(dirs))])
        distance = np.sqrt(np.sum(np.square(step)))
        
        angle = np.mod(np.arctan2(step[1], step[0]) + 2*np.pi, 2*np.pi)
        
        bc_phase = bc_phase + shift_angs

        if directional == 1:
            bc_rate = np.cos(angle - pref_dirs)
            bc_rate = bc_rate > 0
        else:
            bc_rate = np.ones((len(dirs)))

        bc_phase = np.mod(bc_phase+np.pi, 2*np.pi)-np.pi
        bc_phase_record[k, :] = bc_phase
        
        theta_phase = np.mod(np.mean(bc_phase[bc_rate])+np.pi, 2*np.pi)-np.pi
        theta_phases[k] = theta_phase
        
        bc_close = (((bc_phase <= 3*np.pi/pos_samples_per_cycle) + 1 + (bc_phase >= -3*np.pi/pos_samples_per_cycle)) > 2)
        bc_spikes[k, :] = bc_rate * bc_close
        
        if k > pos_samples_per_cycle:
            for i in range(n_dirs):
                temp_sum[i] = exp_window.dot(bc_spikes[(k-pos_samples_per_cycle):k, i])
            
            mem_pot = np.sum(temp_sum)
            if theta_mod == 1:
                mem_pot = mem_pot * 0.5 * (1+np.cos(theta_phase))
            
            if mem_pot >= threshold:
                n_spikes += 1
                gc_rate[y_ind, x_ind] += 1
                gc_spike[n_spikes, 0] = x
                gc_spike[n_spikes, 1] = y
                gc_spike[n_spikes, 2] = theta_phase
                gc_spike[n_spikes, 3] = angle
                gc_spike[n_spikes, 4] = k
            
            if debug:
                print('{} {} {} {}'.format([np.rad2deg(bc_phase), np.rad2deg(theta_phase), bc_spikes[k, 0], mem_pot>=threshold]))
    if ax is None:
        ax = plt.gca()
    by_5_ind = np.arange(0, n_pos, 5)
    if boxcar == 0:
        spike_color = np.zeros((len(gc_spike), 1))
        cmap = plt.cm.get_cmap('turbo')
        m = cm.ScalarMappable(cmap=plt.cm.jet)
        m.set_array(np.hstack([gc_spike[:n_spikes, 2], np.pi, -np.pi]))
        norm = matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        spike_color = gc_spike[:n_spikes, 2]
        im = ax.scatter(gc_spike[:n_spikes, 0], gc_spike[:n_spikes, 1], c=cmap(norm(gc_spike[:n_spikes, 2]), alpha=5))
        ax.plot(positions[by_5_ind, 0], positions[by_5_ind, 1], c='gray', alpha=0.4)
        ax.axis('off')
        ax.set_xlim(0, length)
        ax.set_ylim(0, width)
        
    return gc_spike, n_spikes, positions, im, pref_dirs

def straight_run_theta(width, length, start, end, n_step, directional, theta_mod, dirs, time_const, threshold, ax=None):
    start_x, start_y = start
    end_x, end_y = end
    
    x_seq = np.linspace(start_x, end_x, n_step)
    y_seq = np.linspace(start_y, end_y, n_step)
    
    n_spikes = 0
    theta_phase = 0
    n_dirs = len(dirs)

    bc_spikes = np.zeros((n_step, n_dirs))
    
    bc_phase_record = np.zeros((n_step, n_dirs))
    gc_spike = np.zeros((n_step, 5))

    D = compute_dft_basis(width, length)

    band_waves = np.array([D[dirs[i][0], dirs[i][1]].flatten() for i in range(len(dirs))])
    pref_dirs = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        wv = dirs[i]
        wv[wv > int(width/2)] = wv[wv> int(width/2)] - width
        pref_dirs[i] = np.arctan2(wv[1], wv[0])
    init_pos = np.int16(np.round(start_x*width+start_y))
    bc_phase = getComplexAng(band_waves[:, init_pos])
    bc_spikes[0, :] = 0

    theta_phases = np.zeros((n_step))
    theta_phase = 0
    theta_phases[0] = theta_phase
    
    pos_samples_per_cycle = 12
    
    if time_const > 0:
        exp_window = np.exp(-(pos_samples_per_cycle-np.arange(1, pos_samples_per_cycle+1))/time_const/pos_samples_per_cycle)
    else:
        exp_window = np.ones((pos_samples_per_cycle,))
    
    temp_sum = np.zeros((n_dirs, 1))
    
    for k in range(1, n_step):
        x = x_seq[k]
        y = y_seq[k]
        step_x = x - x_seq[k-1]
        step_y = y - y_seq[k-1]
        
        delta = step_x*width+step_y
        shift_angs = np.array([-2*np.pi*(dirs[i][0]*step_x/length + dirs[i][1]*(width-step_y)/width) for i in range(len(dirs))])
        distance = np.sqrt(np.sum(np.square(np.array([step_x, step_y]))))
        angle = np.arctan2(step_y, step_x)
        bc_phase = bc_phase + shift_angs
        if directional == 1:
            bc_rate = (np.cos(angle - pref_dirs) > 0)
        else:
            bc_rate = np.ones((len(dirs)))
        bc_phase = np.mod(bc_phase+np.pi, 2*np.pi)-np.pi
        bc_phase_record[k, :] = bc_phase
        theta_phase = np.mod(np.mean(bc_phase[bc_rate])+np.pi, 2*np.pi)-np.pi
        theta_phases[k] = theta_phase
        
        bc_close = (((bc_phase <= 3*np.pi/pos_samples_per_cycle)+ 1 + (bc_phase >= -np.pi*(3/pos_samples_per_cycle))) > 2)
        bc_spikes[k, :] = bc_rate * bc_close
        
        if k > pos_samples_per_cycle:
            for i in range(n_dirs):
                temp_sum[i] = exp_window.dot(bc_spikes[(k-pos_samples_per_cycle):k, i])
            
            mem_pot = np.sum(temp_sum)
            if theta_mod == 1:
                mem_pot = mem_pot * 0.5 * (1+np.cos(theta_phase))
            
            if mem_pot >= threshold:
                n_spikes += 1
                gc_spike[n_spikes, 0] = x
                gc_spike[n_spikes, 1] = y
                gc_spike[n_spikes, 2] = theta_phase
                gc_spike[n_spikes, 3] = angle
                gc_spike[n_spikes, 4] = k
    return gc_spike, n_spikes, bc_phase_record, pref_dirs

if __name__=='__main__':
    mpl_data = RGBToPyCmap(turbo_colormap_data)
    plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])

    mpl_data_r = RGBToPyCmap(turbo_colormap_data[::-1,:])
    plt.register_cmap(name='turbo_r', data=mpl_data_r, lut=turbo_colormap_data.shape[0])

    square_path = loadmat('square_path.mat')
    l = np.array([[4, 1], [1, 4], [3, -3], [-4, -1], [-1, -4], [-3, 3]])
    l = np.mod(l+50, 50)
    gc_spike, n_spikes, pos, im, pref_dirs = bandsPhaseFromFourier(50, 50, square_path['square_path']/10, 96, 12.5, 1, 0, l, 0.2, 2.95, 0)

    plt.figure()
    plt.ylim(0, 50)
    cmap = plt.cm.viridis
    for i in range(-50, 50):
        gc_spike_1, n_spikes_1, rec, pref_dirs = straight_run_theta(50, 50, np.array([1, i]), np.array([49, i+16]), 300, 1, 0, l, 0.2, 2.95)
        plt.plot([1, 49], [i, i+16], '--', alpha=0.4, color='black')
        if n_spikes_1 == 0:
            continue
        norm = matplotlib.colors.Normalize(vmin=np.min(gc_spike_1[:n_spikes_1, 2]), vmax=np.max(gc_spike_1[:n_spikes_1, 2]))
        spike_color = gc_spike_1[:n_spikes_1, 2]
        plt.scatter(gc_spike_1[1:n_spikes_1, 0], gc_spike_1[1:n_spikes_1, 1], c=cmap(norm(gc_spike_1[1:n_spikes_1, 2]), alpha=1))

