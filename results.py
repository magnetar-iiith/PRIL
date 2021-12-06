# script to calculate distances and test-time rewards (utility)

import os

import numpy as np

grid_sizes = ['5x5', '10x10']
MDPS = 12
iterations = 10
eps_list = ['0.1', '0.105', '0.2', '0.5', '1.0', '2', '5', '10', 'inf']

for eps in eps_list:
    for gs in grid_sizes:
        distances = []
        for i in range(MDPS):
            with open(gs + '/' + eps + '/' + str(i+1) + '_ground_r.txt', 'r') as f:
                line = f.readline()
                ground_r = [float(x) for x in line.strip().split()]
            rs = []
            for avg_iter in range(iterations):
                with open(gs + '/' + eps + '/' + str(i+1) + '_' + str(avg_iter + 1) + '_r.txt', 'r') as f:
                    line = f.readline()
                    rs.append([float(x) for x in line.strip().split()])
            ground_norms = []
            ground_norms.append(ground_r / np.linalg.norm(ground_r, ord=1))
            ground_norms.append(ground_r / np.linalg.norm(ground_r, ord=2))
            ground_norms.append(ground_r / np.linalg.norm(ground_r, ord=np.inf))
            rs_norms = []
            dists = []
            for iter in range(iterations):
                norms = []
                r = rs[iter]
                ds = []
                count = 0
                norms.append(r/ np.linalg.norm(r, ord=1))
                norms.append(r/ np.linalg.norm(r, ord=2))
                norms.append(r/ np.linalg.norm(r, ord=np.inf))
                rs_norms.append(norms)
                ds.append(np.linalg.norm(norms[0] - ground_norms[0], ord=1))
                ds.append(np.linalg.norm(norms[1] - ground_norms[1], ord=2))
                ds.append(np.linalg.norm(norms[2] - ground_norms[2], ord=np.inf))
                for j in range(len(r)):
                    if ground_r[j] * r[j] < 0:
                        count += 1
                ds.append(count)
                dists.append(ds)
            dists = np.mean(dists, axis=0)
            distances.append(dists)
        print(distances)

        with open(gs + '/' + eps + '/distances.txt', 'w') as f:
            for j in distances:
                f.write('%f ' % j[0])
                f.write('%f ' % j[1])
                f.write('%f ' % j[2])
                f.write('%f ' % j[3])
                f.write('\n')

        rs = 0
        for i in range(MDPS):
            for iter in range(iterations):
                with open(gs + '/' + eps + '/' + str(i+1) + '_' + str(avg_iter + 1) + '_r_test.txt', 'r') as f:
                    line = f.readline()
                    rs += float(line.strip().split()[0])

        rs /= MDPS * iterations

        with open(gs + '/' + eps + '/avg_reward.txt', 'w') as f:
            f.write('%f ' % rs)
            f.write('\n')