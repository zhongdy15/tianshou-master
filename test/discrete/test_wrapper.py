import gym


class ActionBudgetWrapper(gym.Wrapper):
    def __init__(self, env, action_budget):
        super().__init__(env)
        self.action_budget = action_budget
        self.action_budget_remain = action_budget

    def step(self, action):
        # 燃料充足时，action不变，第0个动作不消耗燃料，其他动作消耗燃料
        # 燃料缺乏时，action为默认动作0
        if self.action_budget_remain > 0 :
            real_action = action
            if action != 0:
                self.action_budget_remain -= 1
        else:
            real_action = 0

        obs, reward, done, info = self.env.step(real_action)
        return obs, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.action_budget_remain = self.action_budget
        return observation



env = gym.make("CartPole-v1")
env = ActionBudgetWrapper(env,action_budget=4)

env.action_space.seed(42)

observation= env.reset()

for _ in range(1000):
    a = input("action:")
    if a == "a":
        action = 0
    else:
        action = 1
    #action = env.action_space.sample()
    print("action: "+str(action)+" fuel:"+str(env.action_budget_remain))
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        print("done!")
        observation= env.reset()

env.close()