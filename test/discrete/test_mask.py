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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from learn_inv import InverseModel
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# num_classes = 6
# model=torchvision.models.resnet18(pretrained=True)
# num_features=model.fc.in_features
# model.fc=nn.Linear(num_features,num_classes)
# model.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
class MyDataset(Dataset):
    def __init__(self, all_s, all_act, all_ss,all_act_logits, all_fuel):
        self.all_ss = all_ss
        self.all_act = all_act
        self.all_s = all_s
        self.all_act_logits = all_act_logits
        self.all_fuel = all_fuel
        self.len = len(self.all_ss)

    def __getitem__(self, index):
        return self.all_s[index], self.all_act[index], self.all_ss[index], self.all_act_logits[index], self.all_fuel[index]

    def __len__(self):
        return self.len


mask_pth = "D:\zhongdy\\research\\tianshou-master\\remote_log\\0722实验一\mask_2022-07-22-15-17-53\mask.pth"

mask_model = torch.load(mask_pth,map_location=device)

# print(mask_model)

#制作数据集
buffer_dir = "D:\zhongdy\\research\\tianshou-master\\remote_log\垃圾"
buffer_list = os.listdir(buffer_dir)
all_ss = None
all_act = None
all_s = None
all_act_logits = None
all_fuel = None
#读取总共150*2000 = 30 w条数据
for file in buffer_list:
    print(file)
    if not file.endswith('hdf5'):
        continue

    # 从filename的文件中取得输入：ss，target：act
    filename = os.path.join(buffer_dir, file)
    buf = ReplayBuffer.load_hdf5(filename)

    obs = np.array(buf.obs)
    next_obs = np.array(buf.obs_next)
    act = np.array(buf.act)
    act_prob = torch.tensor(buf.info["act_logits"])

    #获取动作
    act = torch.tensor(act)
    #获取ss
    ss = np.concatenate((obs, next_obs), axis=-1)
    ss = torch.tensor(ss)
    # ss = ss.permute((0, 3, 1, 2))
    # ss = ss/255
    #获取s
    s = torch.tensor(obs)
    # s = s.permute((0, 3, 1, 2))
    # s = s / 255

    #获取fuel
    fuel = torch.tensor(buf.info["fuel_remain"])

    del buf
    # all_ss.append(ss)
    # all_act.append(act)
    if all_ss is not None:
        print("ss_shape:"+str(all_ss.shape))
        all_ss = torch.cat((all_ss,ss),dim=0)
        all_act = torch.cat((all_act, act), dim=0)
        all_s = torch.cat((all_s, s), dim=0)
        all_act_logits = torch.cat((all_act_logits, act_prob), dim=0)
        all_fuel = torch.cat((all_fuel,fuel),dim=0)
    else:
        all_ss = ss
        all_act = act
        all_s = s
        all_act_logits = act_prob
        all_fuel = fuel


mydataset = MyDataset(all_ss=all_ss,all_act=all_act,all_s=all_s,all_act_logits=all_act_logits,all_fuel=all_fuel)
train_loader = DataLoader(dataset=mydataset,
                           batch_size=64,
                           shuffle=True)
print("all data loaded!")
#制作完毕
for step, (batch_s, batch_act, batch_ss, batch_act_logits, batch_fuel) in enumerate(train_loader):
    batch_ss = batch_ss.to(device)
    batch_act = batch_act.to(device)
    batch_s = batch_s.to(device)
    batch_act_logits = batch_act_logits.to(device)
    batch_fuel = batch_fuel.to(device)

    # maskmodel predict
    mask_pred_all_action = mask_model.forward(batch_s.permute((0, 3, 1, 2)) / 255)

    fuel_mask = (torch.cat((batch_fuel.unsqueeze(1), mask_pred_all_action), dim=1)).detach().numpy()
    fuel_mask_max = (torch.cat((batch_fuel.unsqueeze(1),torch.max(mask_pred_all_action,dim=1).values.unsqueeze(1)),dim=1)).detach().numpy()

    mask_factor = mask_pred_all_action

    mask_res_list = [[] for i in range(2)]
    # obs_copy = copy.deepcopy(b.obs)
    # rank = np.argsort(obs_copy, axis=0)[:, -1]
    fuel_remain_copy = copy.deepcopy(batch_fuel)
    # fuel_remain_flag = fuel_remain_copy == 0
    mask_res_list[0] = torch.max(mask_factor[fuel_remain_copy == 0], dim=1)
    mask_res_list[1] = torch.max(mask_factor[fuel_remain_copy > 0], dim=1)

    mask_res_list[0] = mask_res_list[0].values.detach().cpu().numpy()
    mask_res_list[1] = mask_res_list[1].values.detach().cpu().numpy()
    # mask_res_list[0] = mask_load_current_res

    threshold = 1
    ratio_0 = (mask_res_list[0] > 1).sum() / len(mask_res_list[0])
    ratio_1 = (mask_res_list[1] > 1).sum() / len(mask_res_list[1])

    initial = 0
    width = 0.5

    for fuel in range(2):
        plt.bar(range(initial, initial + len(mask_res_list[fuel])), mask_res_list[fuel],
                width=width)
        # plt.bar(range(initial,initial+len(action_1_list[fuel])),action_1_list[fuel],width=width)
        initial += len(mask_res_list[fuel])
    plt.plot([0,64],[1,1],"y")

    plt.title("ratio_no_fuel:%.2f,ratio_fuel:%.2f" % (ratio_0,ratio_1))



    plt.show()

    print("!!")
