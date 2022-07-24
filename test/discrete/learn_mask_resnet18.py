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
#mask数据集的构成
#输入是s：4通道的照片
#输出是6维向量：每维是状态下执行某个动作的评分
#监督学习训练的目标是输出的6维向量select出执行动作对应的factor
#然后factor再和另一个网络的kl factor做监督学习回归
#数据集包含状态s，target：一个模型过s产生factor，再通过a选择哪一个factor，再和概率做除法
class MyDataset(Dataset):
    def __init__(self, all_s, all_act, all_ss,all_act_logits):
        self.all_ss = all_ss
        self.all_act = all_act
        self.all_s = all_s
        self.all_act_logits = all_act_logits
        self.len = len(self.all_ss)

    def __getitem__(self, index):
        return self.all_s[index], self.all_act[index], self.all_ss[index], self.all_act_logits[index]

    def __len__(self):
        return self.len
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_known_args()[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

buffer_dir = os.path.join("/media/yyq/data/zdy",
                          "log/ActionBudget_ALE/AirRaid-v5/ppo",
                          "maskFalse_actionbudget100_seed0_2022-07-18-11-24-44")
# buffer_dir = "D:\zhongdy\\research\\tianshou-master\\remote_log\垃圾"
buffer_list = os.listdir(buffer_dir)

log_name = "mask_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

log_path = os.path.join('log', 'ActionBudget_ALE/AirRaid-v5', 'ppo', log_name)
writer = SummaryWriter(log_path)

#加载inv模型
inv_pth = "/home/zdy/home/zdy/tianshou/test/discrete/log/ActionBudget_ALE/AirRaid-v5/ppo/inv_2022-07-21-17-18-25/inv.pth"
# inv_pth = "D:\zhongdy\\research\\tianshou-master\\remote_log\\0721实验二\inv_2022-07-21-17-18-25\inv.pth"
inv_model = torch.load(inv_pth,map_location=device)


#修改为resnet18
num_classes = 6
model=torchvision.models.resnet18(pretrained=True)
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,num_classes)
model.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

model = model.to(device)

maskmodel = model
optimizer_maskmodel = torch.optim.Adam(maskmodel.parameters(), lr=args.lr)

#增加学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_maskmodel, step_size=20, gamma=0.1)

#制作数据集
all_ss = None
all_act = None
all_s = None
all_act_logits = None
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
    del buf
    # all_ss.append(ss)
    # all_act.append(act)
    if all_ss is not None:
        print("ss_shape:"+str(all_ss.shape))
        all_ss = torch.cat((all_ss,ss),dim=0)
        all_act = torch.cat((all_act, act), dim=0)
        all_s = torch.cat((all_s, s), dim=0)
        all_act_logits = torch.cat((all_act_logits, act_prob), dim=0)
    else:
        all_ss = ss
        all_act = act
        all_s = s
        all_act_logits = act_prob


mydataset = MyDataset(all_ss=all_ss,all_act=all_act,all_s=all_s,all_act_logits=all_act_logits)
train_loader = DataLoader(dataset=mydataset,
                           batch_size=64,
                           shuffle=True)
print("all data loaded!")
#制作完毕


repeat = 200
epoch_losses =[]
epsilon = 1e-8
for epoch in range(repeat):
    losses = []
    for step, (batch_s, batch_act, batch_ss, batch_act_logits) in enumerate(train_loader):
        batch_ss = batch_ss.to(device)
        batch_act = batch_act.to(device)
        batch_s = batch_s.to(device)
        batch_act_logits = batch_act_logits.to(device)

        #maskmodel predict
        mask_pred_all_action = maskmodel.forward(batch_s.permute((0, 3, 1, 2))/255)
        action_one_hot = nn.functional.one_hot(batch_act.long(), 6).bool()
        mask_pred_current_action = torch.masked_select(mask_pred_all_action, action_one_hot)

        #maskfactor target
        with torch.no_grad():
            # 从inv_model里面无梯度地取值用来训练mask
            pi_a = inv_model(batch_ss)
            pi_a = torch.nn.functional.softmax(pi_a)
        pred_pi_act = torch.masked_select(pi_a, action_one_hot)
        target_log_pia = batch_act_logits
        #KL factor
        indepence_factor = torch.log(pred_pi_act + epsilon) - torch.log(batch_act_logits)

        loss_func = torch.nn.MSELoss()
        loss = loss_func(mask_pred_current_action, indepence_factor)
        # print(loss)
        losses.append(loss.item())
        optimizer_maskmodel.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(maskmodel.parameters(), 0.5)
        optimizer_maskmodel.step()
    epoch_losses.append(np.mean(losses))
    print("epoch:"+str(epoch)+" average_loss:"+str(np.mean(losses)))
    # print(losses)
    writer.add_scalar("loss", np.mean(losses), epoch)
    scheduler.step()
    if epoch%20 == 0:
        torch.save(maskmodel, os.path.join(log_path, "mask.pth"))
print('all_loss')
print(epoch_losses)
