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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision

class MyDataset(Dataset):
    def __init__(self, all_ss, all_act):
        self.all_ss = all_ss
        self.all_act = all_act
        self.len = len(self.all_ss)

    def __getitem__(self, index):
        return self.all_ss[index], self.all_act[index]

    def __len__(self):
        return self.len
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.004)
args = parser.parse_known_args()[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

buffer_dir = os.path.join("/media/yyq/data/zdy",
                          "log/ActionBudget_ALE/AirRaid-v5/ppo",
                          "maskFalse_actionbudget100_seed0_2022-07-18-11-24-44")
# buffer_dir = "D:\zhongdy\\research\\tianshou-master\\remote_log\垃圾"
buffer_list = os.listdir(buffer_dir)

log_name = "inv_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

log_path = os.path.join('log', 'ActionBudget_ALE/AirRaid-v5', 'ppo', log_name)
writer = SummaryWriter(log_path)

#修改为resnet18
num_classes = 6
model=torchvision.models.resnet18(pretrained=True)
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,num_classes)
model.conv1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

model = model.to(device)

inversemodel = model
optimizer_inversemodel = torch.optim.Adam(inversemodel.parameters(), lr=args.lr)

#增加学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_inversemodel, step_size=20, gamma=0.1)

#制作数据集
all_ss = None
all_act = None
#读取总共150*2000 = 30 w条数据
for file in buffer_list[0:150]:
    print(file)
    if not file.endswith('hdf5'):
        continue

    # 从filename的文件中取得输入：ss，target：act
    filename = os.path.join(buffer_dir, file)
    buf = ReplayBuffer.load_hdf5(filename)
    obs = np.array(buf.obs)
    next_obs = np.array(buf.obs_next)
    act = np.array(buf.act)
    act = torch.tensor(act)
    ss = np.concatenate((obs, next_obs), axis=-1)
    ss = torch.tensor(ss)
    ss = ss.permute((0, 3, 1, 2))
    ss = ss/255
    del buf
    # all_ss.append(ss)
    # all_act.append(act)
    if all_ss is not None:
        print("ss_shape:"+str(all_ss.shape))
        all_ss = torch.cat((all_ss,ss),dim=0)
        all_act = torch.cat((all_act, act), dim=0)
    else:
        all_ss = ss
        all_act = act


mydataset = MyDataset(all_ss=all_ss,all_act=all_act)
train_loader = DataLoader(dataset=mydataset,
                           batch_size=64,
                           shuffle=True)
print("all data loaded!")
#制作完毕


repeat = 200
epoch_losses =[]
for epoch in range(repeat):
    losses = []
    for step, (batch_ss, batch_act) in enumerate(train_loader):
        batch_ss = batch_ss.to(device)
        batch_act = batch_act.to(device)

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
    # print(losses)
    writer.add_scalar("loss", np.mean(losses), epoch)
    scheduler.step()
    if epoch%20 == 0:
        torch.save(inversemodel, os.path.join(log_path, "inv.pth"))
print('all_loss')
print(epoch_losses)
