from gym.envs.toy_text import discrete
from cvxopt import matrix, solvers
import numpy as np
MAPS = {
    "4x4": ["SFFF","FHFH","FFFH","HFFG"],
    "8x8": ["SFFFFFFF","FFFOOFFF","FFFHFFFF","FFFFFHFF","FFFHFFFF","FHHFFFHF","FHFFHFHF","FFFHFFFG"],
    "5x5_a": ["SOFFF","FOFOF","FOFOF","FOFOF","FFFOG"],
    "5x5_b": ["SFFFF","FFFFF","FFFFF","FFFFF","FFFFG"],
    "5x5_c": ["SOGFF","FOFOF","FOFOF","FOFOF","FFFOF"],
    "5x5_d": ["SFFFF","FFOFO","FFFFF","FOFOF","FFFFG"],
    "5x5_e": ["FFFFO","FFFFF","FFSFF","OFOFF","GFFFF"],
    "5x5_f": ["SFFFH","FFOFF","FFFFF","FFFFO","FFFFA"],
    "5x5_g": ["SFFFA","FFOFF","FFFFF","FFFFO","FFFFH"],
    "5x5_h": ["SOFFF","FFFFF","FFFOF","FFOGF","OFFFF"],
    "5x5_i": ["SFFOF","FFFAF","FFFOF","FOFFO","FFFFH"],
    "10x10_a": ["SFFFOOFFFF","FFFFFFFFFO","FFOOFFFOFF","FFFFFFFOFF","OFFOOFFOFF","FFFOOFFFFF","FFFFFFFOFF","FOOFFOFFFO","FFOFFOFFFF","FFFFFFFFFG"],
    "10x10_b": ["SFFFFFFFFF","FFFFOFFFFF","FOOFOFOOOF","FFFFOFFFFF","FFFFFFFFFF","FOOOOOOOOF","FFFFFFFFFF","FOOFOFOOOF","FOOFOFFFFF","FFFFFFFFFG"],
    "10x10_c": ["SFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFG"],
    "10x10_d": ["SFFFFFFFFF","FFFFFFFFOF","FFOFFFFOFF","FFFOFFOFFF","FFFFOOFFFF","FFFFOOFFFF","FFFOFFOFFF","FFOFFFFOFF","FOFFFFFFFF","FFFFFFFFFG"],
    "10x10_e": ["SFFFFFFFFF","FFFFFFFFFF","FOFOFOFOFF","FOFOFOFFFF","FOFOFOOOFF","FOFOFFFFFF","FOFOOOOOFF","FOFFFFFFFF","FOOOOOOOFF","FFFFFFFFFG"],
    "10x10_f": ["SFOOOOOOOO","FFFFFFFFFO","OFFFFFFFFO","OFFOOOOFFO","OFFOOOOFFO","OFFOOOOFFO","OFFOOOOFFO","OFFFFFFFFO","OFFFFFFFFF","OOOOOOOOFG"],
    "10x10_g": ["SFFFFFFFFF","FFFFFFFFFF","FFFOOFFFFF","FFAFFFFFFF","FFAFAAOFFF","FFFFAGOFFF","FFFFOOOFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF"],
    "10x10_h": ["SFFFFFFFFF","FFFFFFFFFF","FFOFOFFFOF","FFOFOFFFOF","FFOFOFFFOF","FFOFOFFFOF","FFOFOFGFOF","FFOFOFFFOF","FFOFOFFFOF","FFFFFFFFFF"],
    "10x10_i": ["SFFFFFFFFF","FFFFFFFFFF","FFOFOFAFOF","FFOFOFAFOF","FFOFOFAFOF","FFOFOFFFOF","FFOFOFGFOF","FFOFOFFFOF","FFOFOOOOOF","FFFFFFFFFF"],
    "10x10_j": ["SFFFFFFFFF","FFOOOOOOOF","FFFOFOOFOF","FOFFFOOFOF","FOFAFFFFOF","FOFFOFOOOF","FOOFOAFFOF","FOFFOFFFAF","FOOOOOOFFF","FFFFFFFFFG"],
    "10x10_k": ["SFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","OOOOOOOOOF","FFFFFFFFFF","FOOOOOOOOO","FFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFG"],
    "10x10_l": ["SFFFFFFFFF","FFFFFFFFFF","FFFFFFFFFF","FAAOOOAAFF","FOAFFFAOFF","FFFOFOFFFF","FFFFAFFFFF","FFFFAFFFFF","OOFFFFFFFF","OOFFFFFFFG"],
}

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class FrozenLakeEnv(discrete.DiscreteEnv):
    
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, desc=None, map_name="8x8", is_slippery=True, wind=0.1, step_cost=0, start_rew=0, high_rew=1, hole_rew=None):
        if step_cost == 0:
            self.goal_rew = 1 # G
            self.hole_rew = -1 # H
            self.obstacle_rew = -1 # O 
        else:
            self.goal_rew = 10 # G
            self.hole_rew = -10 # H
            self.obstacle_rew = -10 # O

        if hole_rew is not None:
            self.hole_rew = hole_rew # H
        self.high_rew = high_rew # A
        self.start_rew = start_rew # S
        self.step_cost = step_cost # F

        assert map_name is not None
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        # self.reward_range = (0, 1)
        nA = 4
        nS = nrow * ncol
        self.nS = nS
        self.nA = nA

        if is_slippery:
          assert wind >= 0.
          assert wind <= 1.
          self.wind = wind

        ####################################
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            
            ########################
            reward = 0
            if newletter == b'G':
                reward = self.goal_rew
            elif newletter == b'H':
                reward = self.hole_rew
            elif newletter == b'O':
                reward = self.obstacle_rew
            elif newletter == b'F':
                reward = self.step_cost
            elif newletter == b'S':
                reward = self.start_rew
            elif newletter == b'A':
                reward = self.high_rew
            
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        _, rew, done = update_probability_matrix(row, col, a)
                        li.append((1.0, s, rew, done))
                    else:
                        if is_slippery:
                            r1, c1 = inc(row, col, (a-1)%4)
                            r2, c2 = inc(row, col, a)
                            r3, c3 = inc(row, col, (a+1)%4)

                            if r1 == r2 and c1 == c2:
                                li.append((1. - wind/2, *update_probability_matrix(row, col, a)))
                                li.append((wind / 2.0, *update_probability_matrix(row, col, (a + 1) % 4)))

                            elif r2 == r3 and c2 == c3:
                                li.append((wind / 2.0, *update_probability_matrix(row, col, (a - 1) % 4)))
                                li.append((1. - wind/2, *update_probability_matrix(row, col, a)))

                            else:  
                                li.append((wind / 2.0, *update_probability_matrix(row, col, (a - 1) % 4)))
                                li.append((1. - wind, *update_probability_matrix(row, col, a)))
                                li.append((wind / 2.0, *update_probability_matrix(row, col, (a + 1) % 4)))
                        else:
                            li.append((1., *update_probability_matrix(row, col, a)))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
