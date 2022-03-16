import random
import gym
from gym import spaces
import numpy as np
import time
import torch
# if not torch.cuda.is_available():
#     from gym.envs.classic_control import rendering
import copy

class RunningMan(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, initial_chances=6, dist_interval=5, max_step=50, action_penalty=20):
        # initial_chances：初始动作次数,
        # dist_interval=3：有障碍的列间间隔
        # max_step=50：游戏总长度

        # 总共两个动作
        self.action_space = spaces.Discrete(2)
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 地图的长宽为多少个格点
        self.length = 28
        self.height = 7
        # 每个格子的大小
        self.size = 40
        # 目标隔几列出现一次
        self.dist_interval = dist_interval
        # agent初始位置集合
        self.initial_space = [[0, 1]]

        self.n_actions = self.action_space.n

        # 智能体总共能动多少次
        self.initial_chances = initial_chances
        self.chances = None
        # 状态维度
        self.states_dim = 3
        # state:智能体当前的位置
        self.state = None
        # obs:输入神经网络的状态：【游戏进度,相对位置的x差值，相对位置的y差值，剩余能动字数比例】
        self.obs = None
        # 障碍物点的集合
        self.target = []
        self.viewer = None
        # 游戏最大长度
        self.max_step = max_step
        # 游戏步数
        self.counts = 0
        # render中用来显示一定范围内的长度
        self.round = 0

        # 智能体会遇到困难的次数：
        # self.challenge_time = 1+random.choice(range(initial_chances//2, initial_chances))
        self.challenge_time = self.initial_chances
        # 环境可以放置困难的list:
        self.target_pos_num = []
        self.challenge_index = None

        self.action_penalty = action_penalty

    def copy_as_env(self, old_env):
        # initial_chances：初始动作次数,
        # dist_interval=3：有障碍的列间间隔
        # max_step=50：游戏总长度

        # 总共两个动作
        self.action_space = old_env.action_space
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = old_env.observation_space
        # 地图的长宽为多少个格点
        self.length = old_env.length
        self.height = old_env.height
        # 每个格子的大小
        self.size = old_env.size
        # 目标隔几列出现一次
        self.dist_interval = old_env.dist_interval
        # agent初始位置集合
        # todo:
        # 列表深度copy
        self.initial_space = copy.deepcopy(old_env.initial_space)

        self.n_actions = old_env.action_space.n

        # 智能体总共能动多少次
        self.initial_chances = old_env.initial_chances
        self.chances = old_env.chances
        # 状态维度
        self.states_dim = old_env.states_dim
        # state:智能体当前的位置
        self.state = copy.deepcopy(old_env.state)
        # obs:输入神经网络的状态：【游戏进度,相对位置的x差值，相对位置的y差值，剩余能动字数比例】
        self.obs = copy.deepcopy(old_env.obs)
        # 障碍物点的集合
        self.target = copy.deepcopy(old_env.target)
        self.viewer = None
        # 游戏最大长度
        self.max_step = old_env.max_step
        # 游戏步数
        self.counts = old_env.counts
        # render中用来显示一定范围内的长度
        self.round = old_env.round

        # 智能体会遇到困难的次数：
        # self.challenge_time = 1+random.choice(range(initial_chances//2, initial_chances))
        self.challenge_time = old_env.challengetime
        # 环境可以放置困难的list:
        self.target_pos_num = copy.deepcopy(old_env.target_pos_num)
        self.challenge_index = copy.deepcopy(old_env.challenge_index)

        self.action_penalty = copy.deepcopy(old_env.action_penalty)



    # agent自动巡航
    def patrol(self, old_x, old_y):
        x = old_x + 1
        y = old_y
        return x, y

    # 目标刷新位置
    def target_appear(self):
        h = 1
        for index in self.challenge_index:
            self.target.append([index, h])

    def target_fall(self):
        #target里存了所有的障碍物
        old_target = self.target.copy()
        for target in old_target:
            if target[1] > 0:
                target[1] = target[1] - 1
            else:
                #如果掉到底了，就在该列再生成一个石头
                new_height = self.height
                self.target.append([target[0], new_height])
                self.target.remove(target)
        self.target.sort()
        if self.target[-1][0] == self.state[0]:
            # h = self.dist_interval + 1
            index = self.state[0]+self.dist_interval
            #print("index:"+str(index))
            if index in self.challenge_index:
                h = self.dist_interval + 1
            else:
                temp_list = list(range(0, self.dist_interval+1)) + list(range(self.dist_interval + 2, self.height))
                h = random.choice(temp_list)
            #print("target:" + str([index, h]))
            self.target.append([index, h])


    def step(self, action):
        # last_action_info={"last_action_time": -1}
        # txt_name = 'runs/' + \
        #            "chances" + str(self.initial_chances) + '_' + \
        #            "interval" + str(self.dist_interval) + '_' + \
        #            "maxstep" + str(self.max_step) + '_' + \
        #            "acpenalty" + str(self.action_penalty) + \
        #            '.txt'
        # if self.chances == 1 and action == 1:
        #     with open(txt_name, 'a') as f:
        #         f.write(str(self.state[0])+'\n')
        #     last_action_info["last_action_time"] = self.state[0]
        # print(self.target)
        # time.sleep(1)
        # self.render()
        # 判断有无剩余行动机会
        if self.chances > 0:
            # 如果有的话，就做这个动作
            real_action = action
            # 如果做的动作不是0的话（默认动作），机会就减少一个
            if real_action != 0:
                self.chances -= 1
        else:
            #否则的话，self.chances为0的话，只能做默认动作
            real_action = 0


        reward = 0
        #self.target_fall()
        done = False
        x, y = self.state
        if self.is_safe(x+1):
            # 自动向前走
            x, y = self.state
            x, y = self.patrol(x, y)
            next_state = [x, y]
            self.state = next_state
        else:
            if real_action == 1:
                # 自动向前走
                x, y = self.state
                x, y = self.patrol(x, y)
                next_state = [x, y]
                self.state = next_state
            else:
                done = True
                reward = -100

        if x > self.max_step:
            done = True
            reward = 100

        # 在这一步以前获得的奖励相当于是游戏奖励
        # 加一个动作奖励【每做一个动作，就给惩罚，惩罚的具体值用实验来确定】
        action_penalty = self.action_penalty
        if action == 1:
            reward = reward - self.action_penalty


        self.counts += 1
        # 获取新的状态观测
        self.obs, _ = self.get_obs()

        return self.obs, reward, done, {} #last_action_info

    def is_safe(self, index):
        # 如果index 在challenge_index里，是不安全的，回0
        # 否则返回1
        if index in self.challenge_index:
            return 0
        else:
            return 1

    def get_obs(self):
        done = False
        x, y = self.state
        # 如果当前这个状态不安全，马上就结束游戏
        if self.is_safe(x):
            done = False
        else:
            done = True
        # 返回的是，当前到哪了，下一个位置是不是安全，剩余次数的比例
        return np.array([x / self.max_step, self.is_safe(x+1),
                        self.chances / self.initial_chances]), done


    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    def reset(self, start_state=None):
        if start_state is None:
            self.state = random.choice(self.initial_space)
        else:
            self.state = start_state
        self.counts = 0
        self.round = 0

        x = 1
        while x < self.max_step:
            if x % self.dist_interval == 0:
                self.target_pos_num.append(x)
            x += 1
        # 智能体遇到困难的具体位置：
        random.shuffle(self.target_pos_num)
        self.challenge_index = self.target_pos_num[0:self.challenge_time]
        # print(self.challenge_index)
        self.target = []
        self.target_appear()
        self.chances = self.initial_chances
        self.obs, _ = self.get_obs()

        return self.obs

    def render(self, mode='human'):

        length = self.length
        height = self.height
        size = self.size
        # 智能体进行到哪个part了
        self.round = self.state[0] // self.length

        if self.viewer is None:
            self.viewer = rendering.Viewer(length * size, height * size, 'RunningShooter')
        # 绘制网格

        for i in range(length):
            # 竖线
            self.viewer.draw_line(
                (size, 0),
                (size, height * size),
                color=(0, 0, 0)
            ).add_attr(rendering.Transform((size * i, 0)))
            # 横线
            self.viewer.draw_line(
                (0, size), (length * size, size)).add_attr(rendering.Transform((0, size * i)))

        # 绘制障碍的位置【在智能体所在的round内】
        for target in self.target:
            # 障碍在游戏的哪一个round里
            target_round = target[0] // self.length
            target_round_x = target[0] % self.length
            target_round_y = target[1]
            # 只画智能体所在round的方块
            if target_round == self.round:
                self.drawrectangle2([target_round_x,target_round_y], color=(0, 1, 0))

        # 绘制当前state的位置(圆)
        state_round_x = self.state[0] % self.length
        state_round_y = self.state[1]
        center = (
            size + state_round_x * size - 0.5 * size,
            size + state_round_y * size - 0.5 * size)
        self.viewer.draw_circle(
            size / 2.1, 30, filled=True, color=(1, 1, 0)).add_attr(rendering.Transform(center))

        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def drawrectangle(self, point, width, height, **attrs):
        points = [point,
                  (point[0] + width, point[1]),
                  (point[0] + width, point[1] + height),
                  (point[0], point[1] + height)]
        self.viewer.draw_polygon(points, **attrs)

    def drawrectangle2(self, point, **attrs):
        size = self.size
        center = (size + point[0] * size - 0.5 * size, size + point[1] * size - 0.5 * size)
        radius = size / np.sqrt(2)
        res = 4
        points = []
        for i in range(res):
            ang = 2 * np.pi * (i - 0.5) / res
            points.append((np.cos(ang) * radius, np.sin(ang) * radius))

        self.viewer.draw_polygon(points, filled=True, **attrs).add_attr(rendering.Transform(center))


if __name__ == '__main__':
    env = RunningMan(initial_chances=8, dist_interval=1, max_step=20, action_penalty=20)
    action = 0
    maual_flag =True
    reward_list = []

    # 执行oracle策略
    for epoch in range(20):
        total_reward = 0
        obs = env.reset()
        print("obs=" + str(obs))
        # print(env.challenge_index)
        while True:
            env.render()
            # time.sleep(1)

            if maual_flag:
                k = input("action:")
                if k == "s":
                    action = 1
                else:
                    action = 0
            else:
                if abs(obs[1] * env.max_step - obs[2] * env.height) < 2e-5:
                    action = 1
                else:
                    action = 0
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            print("###############counts="+str(env.counts)+"#############")
            print("chances="+str(env.chances))
            print("reward="+str(reward))
            print("done="+str(done))
            print("action="+str(action))
            print("obs=" + str(obs))
            if done:
                reward_list.append(total_reward)

                break

    mean_reward = np.mean(reward_list)
    print(mean_reward)
    print(reward_list)
    env.close()

