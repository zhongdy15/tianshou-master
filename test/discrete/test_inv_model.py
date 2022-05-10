import torch
import tianshou

inv_path = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/chances8_maxstep400_acpenalty0_maskTrue_mf-100_totalinter20000000000_maskst10000000000_policyst10000000000_policyinitial750_2022-05-05-17-07-19/inv.pth"
new_model = torch.load(inv_path)
new_model()