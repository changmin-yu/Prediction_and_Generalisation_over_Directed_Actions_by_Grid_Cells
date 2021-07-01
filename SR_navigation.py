import numpy as np
from fourier_utils import fourierMat
from predictive_reconstruction import constructTransmat

def computeSRMeasure(H, W, target_state, discount=0.9, var=None, offset=None):
    if offset is None:
        offset = np.array([0., 0.])
    T = constructTransmat(H, W, var, offset)
    dft_mat = fourierMat(H*W)
    dfts = dft_mat.dot(T[0])
    measure = dft_mat.dot(np.diag(1/(1-discount*dfts))).dot(np.linalg.inv(dft_mat))
    target_ind = target_state[0] * W + target_state[1]
    probs_to_target = np.real(measure[target_ind])
    probs_to_target[target_ind] = np.inf
    return probs_to_target

def collateNSWE_SRMeasure(H, W, target, offset=None, var=None, discount=0.9):
    T_sym = constructTransmat(H, W, var)
    num_state = H*W
    dft = fourierMat(num_state)
    target_ind = target[0] * W + target[1]
    if offset is None:
        offset = np.array([0., 0.])
    offset_ind = offset[0] * W + offset[1]
    shift_ang = np.array([np.exp(complex(0, 1)*2*np.pi/num_state*offset_ind*k) for k in range(num_state)])
    shift_ang_up = np.array([np.exp(complex(0, 1)*2*np.pi/num_state*(offset_ind-W)*k) for k in range(num_state)])
    shift_ang_down = np.array([np.exp(complex(0, 1)*2*np.pi/num_state*(offset_ind+W)*k) for k in range(num_state)])
    shift_ang_left = np.array([np.exp(complex(0, 1)*2*np.pi/num_state*(offset_ind-1)*k) for k in range(num_state)])
    shift_ang_right = np.array([np.exp(complex(0, 1)*2*np.pi/num_state*(offset_ind+1)*k) for k in range(num_state)])

    evals_sym = dft.dot(T_sym[0])
    evals_orig = shift_ang * evals_sym
    evals_up = shift_ang_up * evals_sym
    evals_down = shift_ang_down * evals_sym
    evals_left = shift_ang_left * evals_sym
    evals_right = shift_ang_right * evals_sym

    dft_inv = np.linalg.inv(dft)
    grid_measure_orig = np.real(dft.dot(np.diag(1/(1-discount*evals_orig))).dot(dft_inv))[:, target_ind]
    grid_measure_up = np.real(dft.dot(np.diag(1/(1-discount*evals_up))).dot(dft_inv))[:, target_ind]
    grid_measure_down = np.real(dft.dot(np.diag(1/(1-discount*evals_down))).dot(dft_inv))[:, target_ind]
    grid_measure_left = np.real(dft.dot(np.diag(1/(1-discount*evals_left))).dot(dft_inv))[:, target_ind]
    grid_measure_right = np.real(dft.dot(np.diag(1/(1-discount*evals_right))).dot(dft_inv))[:, target_ind]

    grid_measure_collated = np.vstack([grid_measure_orig, grid_measure_up, grid_measure_down, \
        grid_measure_left, grid_measure_right])
    grid_measure_collated = np.max(grid_measure_collated, axis=0)
    grid_measure_collated[target_ind] = np.inf
    return grid_measure_collated.reshape(H, W)

def find_adjacent(states, obstacle):
    n, m = states.shape
    adj = []
    for k in range(len(obstacle)):
        i, j = obstacle[k]
        adj.append([i, j-1])
        adj.append([i, j+1])
        adj.append([i, j])
        if k == 0:
            adj.append([i-1, j])
        if k == len(obstacle)-1:
            adj.append([i+1, j])
    return np.array(adj)@np.array([1, m])

def epsilon_greedy_transmat(D, H, W, greedy_prob=0.95):
    transmat = np.zeros((H*W, H*W))
    for i in range(H):
        for j in range(W):
            adj_states = np.array([[np.mod(i+2, H), np.mod(j-1, W)], [np.mod(i+2, H), np.mod(j+1, W)], 
                                  [np.mod(i+1, W), j], [np.mod(i+3, W), j]])
            vals = np.array([D[adj[0], adj[1]] for adj in adj_states])
            greedy_action = np.argmax(vals)
            transmat[i*W+j, np.mod(adj_states[greedy_action][0]*W+adj_states[greedy_action][1], H*W)] = greedy_prob
            for adj in adj_states:
                transmat[i*W+j, np.mod(adj[0]*W+adj[1], H*W)] += (1-greedy_prob)/len(adj_states)
    return transmat

def adjustSR_barrier(H, W, obstacle=None, var=None, offset=None, discount=0.95, target_state=None):
    states = np.ones((H, W))
    if offset is None:
        offset = np.array([0., 0.])
    if obstacle is None:
        obstacle = np.array([[5, k] for k in range(1, 6)])
    obstacle_inds = obstacle@np.array([1, W])
    J = find_adjacent(states, obstacle)
    T0 = constructTransmat(H, W, var=var, offset=offset)
    SR_0 = np.linalg.inv(np.eye(H*W)-discount*T0)
    T = T0.copy()
    T[obstacle_inds, :] = 0
    T[:, obstacle_inds] = 0
    for i in range(H*W):
        if np.sum(T[i, :]) != 0:
            T[i, :] /= np.sum(T[i, :])
    L0 = np.eye(H*W) - T0
    L = np.eye(H*W) - T
    d = L[J, :] - L0[J, :]
    m0 = SR_0[:, J]
    SR_new = SR_0 - m0@np.linalg.inv(np.eye(len(J))+d@m0)@d@SR_0
    if target_state is None:
        target_state = np.array([np.random.choice(H), np.random.choice(W)])
    target_ind = target_state[0] * W + target_state[1]
    SR_new[target_ind, target_ind] = np.inf
    SR_new[obstacle_inds, target_ind] = -np.inf
    return SR_new.reshape(H, W)

def shortestPath_SR(SR_measure, H, W, init_state, target_state, external_offset=None):
    curr_state = init_state
    if len(SR_measure.shape) == 1:
        SR_measure = SR_measure.reshape(H, W)
    if external_offset is None:
        external_offset = np.array([0., 0.])
    NSEW = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) + external_offset
    state_seq = []
    count = 0
    while np.sum(np.abs(curr_state-target_state)) != 0:
        count += 1
        state_seq.append(curr_state)
        next_states = np.array([SR_measure[np.mod((curr_state + n)[0], H), np.mod((curr_state+n)[1], W)] for n in NSEW])
        curr_state = np.mod(curr_state+NSEW[np.argmax(next_states)], np.array([H, W]))
    state_seq.append(np.array([2, 8]))