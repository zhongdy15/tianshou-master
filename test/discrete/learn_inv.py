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
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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

#todo:读取离线样本
buffer_dir = os.path.join("/media/yyq/data/zdy","buffer0710", "data_collect")
filename = os.path.join(buffer_dir,'epoch_1step_2000.hdf5')
buf = ReplayBuffer.load_hdf5(filename)
print("load_success!")
