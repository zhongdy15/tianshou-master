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
sys.path.insert(0, package)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class InverseModel(nn.Module):
    def __init__(self):
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
            layer_init(nn.Linear(32 * 4 *3, 256)),
            nn.ReLU(), )
        self.prob_predict = layer_init(nn.Linear(256, 10), std=0.01)

    def forward(self, x):
        bchw_x = x.permute((0, 3, 1, 2))
        xx = bchw_x / 255
        a = self.network(xx)
        logits = self.prob_predict(a)
        return logits


if __name__ == '__main__':

    buffer_dir = os.path.join("/media/yyq/data/zdy",
                              "log/ActionBudget_ALE/Amidar-v5/ppo",
                              "maskTrue_actionbudget200_seed0_2022-08-15-16-47-23")
    # 之前较好的数据集："maskFalse_actionbudget100_seed0_2022-07-18-11-24-44")
    # buffer_dir = "D:\zhongdy\\research\\tianshou-master\\remote_log\垃圾"
    buffer_list = os.listdir(buffer_dir)

    log_name = "inv_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log_path = os.path.join('log', 'ActionBudget_ALE/Amidar-v5', 'ppo', log_name)
    writer = SummaryWriter(log_path)
    inversemodel = InverseModel().to(device)
    optimizer_inversemodel = torch.optim.Adam(inversemodel.parameters(), lr=1e-4, eps=1e-5)

    repeat = 200
    epoch_losses =[]
    for epoch in range(repeat):

        for file in buffer_list:
            print(file)
            if not file.endswith('hdf5'):
                continue

            losses = []
            #从filename的文件中取得输入：ss，target：act
            filename = os.path.join(buffer_dir,file)
            buf = ReplayBuffer.load_hdf5(filename)
            obs = np.array(buf.obs)
            next_obs = np.array(buf.obs_next)
            act = np.array(buf.act)
            act = torch.tensor(act).to(device)
            ss = np.concatenate((obs, next_obs), axis=-1)
            ss = torch.tensor(ss).to(device)

            batchsize = 64
            batch_ss_list = torch.split(ss,batchsize)
            batch_act_list = torch.split(act,batchsize)

            for index in range(len(batch_ss_list)):
                batch_ss = batch_ss_list[index]
                batch_act = batch_act_list[index]
                logits = inversemodel.forward(batch_ss)
                # print(logits)

                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(logits, batch_act)
                # print(loss)
                losses.append(loss.item())
                optimizer_inversemodel.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(inversemodel.parameters(), 0.5)
                optimizer_inversemodel.step()
        epoch_losses.append(np.mean(losses))
        print("epoch:"+str(epoch)+" average_loss:"+str(np.mean(losses)))
        print(losses)
        writer.add_scalar("loss", np.mean(losses), epoch)
        if epoch % 20 == 0:
            torch.save(inversemodel, os.path.join(log_path, "inv.pth"))
    print('all_loss')
    print(epoch_losses)
