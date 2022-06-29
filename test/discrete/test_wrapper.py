import gym


class ActionBudgetWrapper(gym.Wrapper):
    def __init__(self, env, action_budget,default_actionindex=0):
        super().__init__(env)
        self.action_budget = action_budget
        self.action_budget_remain = action_budget
        self.default_actionindex = default_actionindex

    def step(self, action):
        # 燃料充足时，action不变，第default_actionindex个动作不消耗燃料，其他动作消耗燃料
        # 燃料缺乏时，action为默认动作default_actionindex
        if self.action_budget_remain > 0 :
            real_action = action
            if action != self.default_actionindex:
                self.action_budget_remain -= 1
        else:
            real_action = self.default_actionindex

        obs, reward, done, info = self.env.step(real_action)
        info["fuel_remain"] = self.action_budget_remain
        return obs, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.action_budget_remain = self.action_budget
        return observation

if __name__ == '__main__':
    import time
    env = gym.make("AmidarNoFrameskip-v4")
    env = ActionBudgetWrapper(env,action_budget=4)

    env.action_space.seed(42)
    # print(env.spec.reward_threshold)
    observation= env.reset()

    for _ in range(1000):
        # a = input("action:")
        # if a == "a":
        #     action = 0
        # else:
        #     action = 1

        action = env.action_space.sample()
        print("action: "+str(action)+" fuel:"+str(env.action_budget_remain))
        observation, reward, done, info = env.step(action)
        print(info)
        env.render()


        # time.sleep(0.1)

        if done:
            print("done!")
            observation= env.reset()

    env.close()