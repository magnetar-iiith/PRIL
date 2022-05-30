import numpy as np

def get_state_rewards(env):
    rews = [0.] * env.nS
    for i in range(env.nS):
      dictn = env.P[i]
      for a in range (env.nA):
        li = dictn[a]
        for (p,s,r,d) in li:
          rews[s] += p * r
    return rews

def get_transition_prob_matrix(env):

    tns_prob = np.zeros((env.nS,env.nA,env.nS))
    for i in range(env.nS):
      dicn = env.P[i]
      for a in range(env.nA):
        li = dicn[a]
        for (p,s,r,d) in li:
          tns_prob[i][a][s] += p
    return tns_prob

def to_s(row, col, ncol):
    return row*ncol + col

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
