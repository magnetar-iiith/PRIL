import gym 
import numpy as np
import math
from scipy import stats
from gym import spaces
from gym.utils import seeding

d = 10
class MountainCarEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self, goal_velocity=0, reward_class=0):
        self.min_position = -1.2
        self.max_position = 0.6
        # self.max_speed = 0.07
        self.max_speed = 0.1
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        # self.force = 0.001
        self.force = 0.007
        self.gravity = 0.0025
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.seed()
        self.reward_class = reward_class
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def get_reward(self, reward_class, position, done):
        if reward_class == 0:
            if done: reward = 0.0
            else: reward = -1.0
        elif reward_class == 1:
            if done: reward = 1.0
            else: reward = 0.0
        else: reward = np.sin(3 * position) * 0.45 + 0.55
        return reward
    def reward_fn(self, alphas, position, std, means):
        reward = 0
        for i in range(d):
            mean = means[i]
            phi_rew = stats.norm(mean, std).pdf(position)
            reward += phi_rew * alphas[i]
        return reward
    def step(self, action, reward_type="inbuilt", alphas=None, std=None, means=None):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action, type(action),)
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        if reward_type == "inbuilt":
            reward = self.get_reward(self.reward_class, position, done)
        elif reward_type == "custom":
            reward = self.reward_fn(alphas, position, std, means) 
        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, done, {}
    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32)
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400
        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)
            clearance = 10
            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)
        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
