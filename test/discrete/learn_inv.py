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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# print(sys.path)
# import tianshou
# print(tianshou.utils.__path__)
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy import MaskPPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net, ConvNet
from tianshou.utils.net.discrete import Actor, Critic

from tianshou.env import RunningMan
#from minessweeper import RunningMan
from test_wrapper import ActionBudgetWrapper
import torch.nn as nn
#todo:读取离线样本
# buffer_dir = os.path.join("/media/yyq/data/zdy",
#                           "log/ActionBudget_ALE/AirRaid-v5/ppo/maskFalse_actionbudget100_seed20_2022-07-12-15-24-35")
buffer_dir = "D:\zhongdy\\research\\tianshou-master\\remote_log\垃圾"
filename = os.path.join(buffer_dir,'epoch_1step_2000.hdf5')
buf = ReplayBuffer.load_hdf5(filename)
print("load_success!")
# 训练inv_model,输入是s和s‘，输出标签是a，
obs = np.array(buf.obs)
next_obs = np.array(buf.obs_next)
act = np.array(buf.act)

ss = np.concatenate((obs, next_obs), axis=-1)
ss = torch.tensor(ss)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class InverseModel(nn.Module):
    def __init__(self, frames=4):
        super(InverseModel, self).__init__()
        self.network = nn.Sequential(
            # 8个channel，输入为S+s‘
            layer_init(nn.Conv2d(8, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*18, 256)),
            nn.ReLU(),)
        # 输出是env的空间，[256,6,4,4,4,4,7,49]
        # 暂时只用训练action_type的维度
        self.prob_predict = layer_init(nn.Linear(256, 6), std=0.01)
        #self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        bchw_x = x.permute((0, 3, 1, 2))
        xx = bchw_x/255
        a = self.network(xx)
        logits = self.prob_predict(a)
        # logits = torch.nn.functional.softmax(logits)
        return logits # "bhwc" -> "bchw" batchsize\height\width\channel


inversemodel = InverseModel()
optimizer_inversemodel = torch.optim.Adam(inversemodel.parameters(), lr=1e-4, eps=1e-5)
batchsize = 64
batch_ss_list = torch.split(ss,batchsize)
batch_act_list = torch.split(torch.tensor(act),batchsize)

for index in range(len(batch_ss_list)):
    batch_ss = batch_ss_list[index]
    batch_act = batch_act_list[index]
    logits = inversemodel.forward(batch_ss)
    # print(logits)

    loss_item = torch.nn.CrossEntropyLoss()
    loss = loss_item(logits, batch_act)
    print(loss)
    optimizer_inversemodel.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(inversemodel.parameters(), 0.5)
    optimizer_inversemodel.step()