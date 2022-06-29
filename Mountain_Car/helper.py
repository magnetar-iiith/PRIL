import numpy as np
from cvxopt import matrix, solvers
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
from scipy import stats

def compute_epsilon(steps, noise_multiplier, batch_size, max_steps):                                                                                  
    if noise_multiplier == 0.0:
        return float('inf')                                                                                                                           
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))                                                                               
    sampling_probability = batch_size / max_steps                                                                                                     
    rdp = compute_rdp(q=sampling_probability,                                                                                                         
                        noise_multiplier=noise_multiplier,                                                                                              
                        steps=steps,                                                                                                                    
                        orders=orders)                                                                                                                  
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

"""
k: the number of policies apart from the optimal policy
d: 10 (number of phi rewards)
V_opt_vec : [d] = [v*1, v*2, .... v*d] for optimal policy
V_vecs : [k, d] = [V1, V2, .... Vk]
where Vi = [vi_1, vi_2, .... vi_d]
value difference vectors: k such vectors
V* - V1, V* - V2, ..... V* - Vk
sum(V* - Vi) = a1 * (V*1 - Vi_1) + ...... + ad * (V*d - Vi_d)
Pi = p(sum(V* - Vi))
Optimization objective: max P1 + P2 + ... Pk
Constraint: -1 <= ai <= 1 for all i in {1,...,d}
"""

def get_alphas(k, d, V_star, V_vecs):
    S = []
    for i in range(k):
        V_temp = -1. * (V_star - V_vecs[i]) # [d]
        S.append(V_temp)
        S.append(V_temp * 2)
    # S = 2k x d
    S = np.asarray(S)
    # S_left = 2k x k
    S_left = np.identity(k)
    S_left = np.repeat(S_left, 2, 0)
    # S_diagonal = 2d x k
    S_diagonal = np.zeros(shape=(2*d, k))
    # S_down = 2d x d
    S_down = -1 * np.identity(d)
    S_down = np.repeat(S_down, 2, 0)
    S_down[::2] = S_down[::2] * -1
    # A = (2k + 2d) x (k + d)
    A_top = np.hstack([S_left, S])   
    A_bottom = np.hstack([S_diagonal, S_down])
    A = np.vstack([A_top, A_bottom])
    A = matrix(A)
    # C = (k + d) x 1
    C = np.ones(shape=(k+d, 1))
    C[k:,0] = 0.
    C = C * -1
    C = matrix(C)
    # B = (2k + 2d) x 1
    B = np.ones(shape=(2*k + 2*d,1))
    B[:2*k, 0] = 0.
    B = matrix(B)  
    sol = solvers.lp(C, A, B)
    alphas = sol['x'][k:]
    return np.array(alphas).reshape((d,))

def get_mean_values(model, num_trajs, d, std, gamma, mode="random"):
    rews = []
    positions = []
    values_pi_phi = None
    for i in range(num_trajs):
        rs, xs = model.runTestLoop(mode)
        positions.append(xs)
        rews.append(rs)
    rews = np.mean(rews)
    positions = np.mean(positions, axis=0) # length is trajectory length = 200
    phi_rews = phi_rewards(xs=xs, d=d, std=std, means=means)
    phi_rews_disc = discount_rews(phi_rews, gamma=gamma, d=d)
    values_pi_phi = get_traj_values(phi_rews_disc, d=d)
    return values_pi_phi, rews
    # values = get_estim_values(values_pi_phi, alphas)
    # return values
    # if update == 1:
    #     values_pi_phi = get_traj_values(phi_rews_disc)
    # else:
    #     # moving average
    #     values_pi_phi = (update-1) * values_pi_phi
    #     values_pi_phi = values_pi_phi + get_traj_values(phi_rews_disc)
    #     values_pi_phi = values_pi_phi / update
    #     # values_pi = get_estim_values(values_pi_phi, ALPHAS)

def phi_rewards(xs, d, std, means):
    rewards = []
    for i in range(d):
        mean = means[i]
        phi_rews = stats.norm(mean, std).pdf(xs)
        rewards.append(phi_rews)
    return rewards

def discount_rews(rs, gamma, d):
    for i in range(d): 
        disc = 1.0
        for j in range(len(rs[i])): # 200 tim
            rs[i][j] *= disc
            disc *= gamma
    return rs

def get_traj_values(rews, d):
    res_phi = []
    for i in range(d): # 10 phi rewards
        res_phi.append(np.sum(rews[i], axis=0))
    res_phi = np.array(res_phi)
    return res_phi

def get_estim_values(values, alphas):
    values = np.array(values)
    for i in range(values.shape[0]): # 33 phis
        values[i][:] *= alphas[i]
    values = np.sum(values, axis=0)
    return values

def get_norm_alphas(rew_class):
    if rew_class == 0:
        alphas = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
    elif rew_class == 1:
        alphas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif rew_class == 2:
        alphas = [0.749, 0.486, 0.246, 0.112, 0.131, 0.296, 0.55, 0.804, 0.969, 0.988]
    norms = []
    norms.append(alphas/ np.linalg.norm(alphas, ord=1))
    norms.append(alphas/ np.linalg.norm(alphas, ord=2))
    norms.append(alphas/ np.linalg.norm(alphas, ord=np.inf))
    return norms