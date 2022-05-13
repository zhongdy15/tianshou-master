import argparse
import copy
import os
import pprint
import time
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
torch.set_num_threads(16)
import sys
package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(package)
# print(sys.path)
sys.path.insert(0, package)
os.environ['CUDA_VISIBLE_DEVICES'] = '4,6,7'
# print(sys.path)
# import tianshou
# print(tianshou.utils.__path__)
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy import MaskPPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic

from tianshou.env import RunningMan
#from minessweeper import RunningMan


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='RunningShooter')#CartPole-v0 RunningShooter
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=4)
    parser.add_argument('--test-num', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # runningman special
    parser.add_argument('--initial_chances', type=int, default=8,
                        help='chances for agent to act')
    parser.add_argument('--dist_interval', type=int, default=1,
                        help='interval for target appear')
    parser.add_argument('--max_lenth', type=int, default=200,
                        help='max_lenth')
    parser.add_argument('--action_penalty', type=int, default=0,
                        help='action_penalty')
    parser.add_argument("--mask", action="store_true",
                        help="no mask default")
    parser.add_argument("--mask_factor", type=float, default=-100,
                        help="mask_factor")
    parser.add_argument('--mask_hidden_sizes', type=int, nargs='*', default=[256, 256])



    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)

    # mask_special
    # 每一个total_update_inerval中
    # 0-mask_update_start:更新inv模型
    # mask_update_start - policy_update_start:更新inv模型和mask模型
    # policy_update_start以后：保持训练得到的mask不变，仅训练策略
    parser.add_argument('--total_update_interval', type=int, default=2e10)
    parser.add_argument('--mask_update_start', type=int, default=1e10)
    parser.add_argument('--policy_update_start', type=int, default=1e10)
    parser.add_argument('--policy_learn_initial', type=int, default=250)


    args = parser.parse_known_args()[0]
    # print("!!!!device!!!"+args.device)
    # print("73")
    # print(args.mask)
    # print(args.max_lenth)
    # args.mask = True
    # args.policy_learn_initial = 3
    # args.total_update_interval = 8
    # args.mask_update_start = 3
    # args.policy_update_start = 5
    # args.epoch = 60
    # args.max_lenth = 200
    return args


def env_make(task, args=get_args()):
    if task == "RunningShooter":
        # print(args.mask)
        # print(args.max_lenth)
        env = RunningMan(initial_chances=args.initial_chances, dist_interval=args.dist_interval,
                         max_step=args.max_lenth, action_penalty=args.action_penalty)
    else:
        env = gym.make(task)
    return env


def test_ppo(args=get_args()):

    env = env_make(args.task, args=args)



    # print("88")
    # print(args.mask)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = env_make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = SubprocVectorEnv(
        [lambda: env_make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = env_make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: env_make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    state_num = int(np.prod(args.state_shape))
    action_num = int(np.prod(args.action_shape))

    # p(a|s,s') model:inverse_dynamic model
    inv_model = Net(state_num*2, action_num, hidden_sizes=args.mask_hidden_sizes, device=args.device, softmax=True).to(args.device)
    # M(s,a) model: independence factor for mask
    # action : 1维两个选择
    # mask model输出s下每个a的因子
    mask_model = Net(state_num, action_num, hidden_sizes=args.mask_hidden_sizes, device=args.device).to(args.device)

    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        # print("cuda is available")
        # print(args.mask)
        actor = DataParallelNet(
            Actor(net, args.action_shape, device=args.device, mask=args.mask).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=args.device).to(args.device))
    else:
        # print("cuda is not available")
        # print(args.mask)
        actor = Actor(net, args.action_shape, device=args.device, mask=args.mask).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    inv_optim = torch.optim.Adam(inv_model.parameters(), lr=args.lr)
    mask_optim = torch.optim.Adam(mask_model.parameters(), lr=args.lr)

    dist = torch.distributions.Categorical

    log_name = "chances" + str(args.initial_chances) + '_' \
               "maxstep" + str(args.max_lenth) + '_' + \
               "acpenalty" + str(args.action_penalty) + '_' +\
               "mask" + str(args.mask) + '_' + \
               "mf" + str('{:.0e}'.format(args.mask_factor)) + '_' + \
               "totalinter" + str('{:.0e}'.format(args.total_update_interval)) + '_' + \
               "maskst" + str('{:.0e}'.format(args.mask_update_start)) + '_' + \
               "policyst" + str('{:.0e}'.format(args.policy_update_start)) + '_' + \
               "policyinitial" + str('{:.0e}'.format(args.policy_learn_initial)) + '_' + \
               time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log_path = os.path.join(args.logdir, args.task, 'ppo', log_name)

    if args.mask:
        policy = MaskPPOPolicy(
            actor,
            critic,
            optim,
            dist,
            save_dir=log_path,
            total_update_interval=args.total_update_interval,
            mask_update_start=args.mask_update_start,
            policy_update_start=args.policy_update_start,
            policy_learn_initial=args.policy_learn_initial,
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            action_space=env.action_space,
            deterministic_eval=True,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
            inv_model=inv_model,
            inv_optim=inv_optim,
            mask_model=mask_model,
            mask_optim=mask_optim
        )
    else:
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            action_space=env.action_space,
            deterministic_eval=True,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
        )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, args.training_num)
    )
    #load_buffer = ReplayBuffer.load_hdf5("data_collect\\20220405194952.hdf5")
    test_collector = Collector(policy, test_envs)
    # # log
    # print(args.mask)
    # "interval" + str(args.dist_interval) + '_' + \
    # parser.add_argument('--total_update_interval', type=int, default=200)
    # parser.add_argument('--mask_update_start', type=int, default=100)
    # parser.add_argument('--policy_update_start', type=int, default=150)
    # parser.add_argument('--policy_learn_initial', type=int, default=200)

    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= 110 #env.spec.reward_threshold

    # # ---for test fixed policy---
    # policy_pth = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/" \
    #              "chances8_maxstep200_acpenalty0_maskTrue_mf-1e+02_totalinter2e+10_maskst1e+10_policyst1e+10_policyinitial2e+02_2022-05-10-16-29-47" \
    #              "/policy.pth"
    #
    # load_policy = copy.deepcopy(policy)
    # model = torch.load(policy_pth)
    # inv_model_2 = Net(state_num * 2, action_num, hidden_sizes=[64, 64], device=args.device,
    #                 softmax=True).to(args.device)
    # mask_model_2 = Net(state_num, action_num, hidden_sizes=[64, 64], device=args.device).to(args.device)
    #
    # load_policy.inv_model = inv_model_2
    # load_policy.mask_model = mask_model_2
    #
    # load_policy.load_state_dict(torch.load(policy_pth))
    # #
    # policy.actor = copy.deepcopy(load_policy.actor)
    # # policy.inv_model = copy.deepcopy(load_policy.inv_model)
    # # ---for test fixed policy---


    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )
    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = env_make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_ppo()
