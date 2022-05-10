import torch
import tianshou

#inv_path = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/chances8_maxstep400_acpenalty0_maskTrue_mf-100_totalinter20000000000_maskst10000000000_policyst10000000000_policyinitial750_2022-05-05-17-07-19/inv.pth"
#new_model = torch.load(inv_path)
#new_model()
policy_pth = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/chances8_maxstep200_acpenalty0_maskFalse_mf-100_totalinter20000000000.0_maskst10000000000.0_policyst10000000000.0_policyinitial250_2022-05-05-15-24-55/policy.pth"
new_model = torch.load(policy_pth)