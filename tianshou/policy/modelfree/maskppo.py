from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
import copy
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
import matplotlib.pyplot as plt
import os

class MaskPPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        inv_model: torch.nn.Module,
        inv_optim: torch.optim.Optimizer,
        mask_model: torch.nn.Module,
        mask_optim: torch.optim.Optimizer,
        save_dir: str,
        total_update_interval: int = 200,
        mask_update_start: int = 100,
        policy_update_start: int = 150,
        policy_learn_initial: int = 200,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        self.inv_model = inv_model
        self.inv_optim = inv_optim
        self.mask_model = mask_model
        self.mask_optim = mask_optim
        self.total_update_interval = total_update_interval  # 80#200
        self.mask_update_start = mask_update_start  # 40#100
        self.policy_update_start = policy_update_start  # 60#150
        self.policy_learn_initial = policy_learn_initial  # 200#500
        self.save_dir = save_dir

        self.learn_index = 0
        # self.state_discrete_num = self.state_to_int([1, 1, 1]) + 1
        # self.action_num = 2
        # state_discrete_num = self.state_discrete_num
        # action_num = self.action_num
        # cal_ssa = np.zeros((state_discrete_num, state_discrete_num, action_num))
        # cal_ss = np.zeros((state_discrete_num, state_discrete_num))
        # cal_sa = np.zeros((state_discrete_num, action_num))
        # self.cal_ssa = cal_ssa
        # self.cal_ss = cal_ss
        # self.cal_sa = cal_sa

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def state_to_int(self,state):
        # 对于离散环境，把状态对应到int值上去
        max_lenth = 200
        action_chances = 8
        # state = [0.005,1.,0.875]
        return int(state[0] * max_lenth + state[1] * (max_lenth+1) + state[2] * action_chances * (max_lenth+1) * 2)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:

        # for test
        epsilon = 1e-5
        inv_path = os.path.join(self.save_dir, 'inv.pth')
        mask_path = os.path.join(self.save_dir, 'mask.pth')
        actor_path = os.path.join(self.save_dir, 'actor.pth')
        # torch.save(self.inv_model, inv_path)
        # new_model = torch.load(inv_path)
        fig_save_interval = 20


        #---todo in 4/28---
        #修改ppo的learn函数：
        #原本的learn函数间隔为d,学习actor、critic网络
        #学习另外两个网络（以不同的间隔）：
        #极大似然估计训练p(a|s,s'):1）简单网络交叉熵2）用boltzman函数增强不确定性
        #用极大似然估计的概率用来训练无关性因子

        #根据无关性因子向self.actor.mask传递mask
        #actor网络根据mask修改动作输出

        #固定策略训练mask、固定mask训练策略、交替训练
        #---todo in 4/28---

        # 每40个回合执行如下操作：
        # 前20个回合训练MASK、策略pi不变、mask=NONE：
        # 前10个回合只训练inverse model
        # 10-20回合既训练inv_model，又利用inv_model训练mask_model
        # 第20个回合替换原有的mask为新的mask，mask从None变成模型
        # 后20个回合只训练pi，mask不变：
        # 保持新的mask，不再变化，仅训练策略
        total_update_interval = self.total_update_interval #80#200
        mask_update_start = self.mask_update_start #40#100
        policy_update_start = self.policy_update_start #60#150

        # 一开始进行policy学习，直到学习得差不多了，再进行mask+policy迭代
        policy_learn_initial = self.policy_learn_initial #200#500


        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        inverse_losses, mask_losses = [],[]

        if self.learn_index < policy_learn_initial:
            for step in range(repeat):
                if self._recompute_adv and step > 0:
                    batch = self._compute_returns(batch, self._buffer, self._indices)
                for b in batch.split(batch_size, merge_last=True):
                    # calculate loss for actor
                    dist = self(b).dist
                    if self._norm_adv:
                        mean, std = b.adv.mean(), b.adv.std()
                        b.adv = (b.adv - mean) / std  # per-batch norm
                    ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                    ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                    surr1 = ratio * b.adv
                    surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                    if self._dual_clip:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self._dual_clip * b.adv)
                        clip_loss = -torch.where(b.adv < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()
                    # calculate loss for critic
                    value = self.critic(b.obs).flatten()
                    if self._value_clip:
                        v_clip = b.v_s + (value -
                                          b.v_s).clamp(-self._eps_clip, self._eps_clip)
                        vf1 = (b.returns - value).pow(2)
                        vf2 = (b.returns - v_clip).pow(2)
                        vf_loss = torch.max(vf1, vf2).mean()
                    else:
                        vf_loss = (b.returns - value).pow(2).mean()
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    loss = clip_loss + self._weight_vf * vf_loss \
                        - self._weight_ent * ent_loss
                    self.optim.zero_grad()
                    loss.backward()
                    if self._grad_norm:  # clip large gradient
                        nn.utils.clip_grad_norm_(
                            self._actor_critic.parameters(), max_norm=self._grad_norm
                        )
                    self.optim.step()
                    clip_losses.append(clip_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())
                    losses.append(loss.item())
            print("learn_initial_policy_inex:" + str(self.learn_index) + " policy_loss" + str(loss.item()))
        else:


            #learn inverse_model
            if self.learn_index % total_update_interval ==0:
                #ineversemodel是否需要重新初始化
                pass

            # inv_load_path = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/" \
            #            "chances8_maxstep200_acpenalty0_maskTrue_mf-1e+02_totalinter2e+10_maskst1e+10_policyst1e+10_policyinitial0e+00_2022-05-10-20-07-06" \
            #            "/inv.pth"
            # load_model = torch.load(inv_load_path)
            # load_res = load_model(ss)

            if self.learn_index % total_update_interval < mask_update_start:
                # 训练inv_model
                fig_save_index = 0
                for step in range(repeat):
                    for b in batch.split(batch_size, merge_last=True):
                        fig_save_index += 1
                        # 考虑采用状态的差作为输入
                        # ss = np.concatenate((b.obs, b.obs_next), axis=1)
                        delta_s = b.obs_next - b.obs
                        ss = np.concatenate((b.obs, delta_s), axis=1)
                        pred = self.inv_model(ss)

                        # action_shape = self.actor.net.module.action_shape if torch.cuda.is_available() else self.actor.action_shape
                        # target = nn.functional.one_hot(b.act.long(), action_shape)


                        # ---inv model plot in 05/11---
                        # for test 测试训练好的inv model的表现

                        action_0_list = [[] for i in range(9)]
                        action_1_list = [[] for i in range(9)]
                        with torch.no_grad():
                            load_res = pred
                        # KL factor
                        factor =torch.log(load_res[0])- torch.log(self(b).logits+epsilon)
                        # # TV factor
                        # factor =  abs(self(b).logits / (load_res[0]+epsilon) - 1)

                        # for items in  factor:
                        obs_copy = copy.deepcopy(b.obs)
                        rank = np.argsort(obs_copy,axis=0)[:,-1]
                        for i in rank:
                            fuel = int(obs_copy[i,-1]*8)
                            action_0_list[fuel].append(factor[i,0].item())
                            action_1_list[fuel].append(factor[i,1].item())
                            #print(obs_copy[i,-1])
                        initial = 0
                        width = 0.5

                        plt.cla()
                        plt.figure()
                        plt.title("action_0")

                        for fuel in range(9):
                            plt.bar(range(initial,initial+len(action_0_list[fuel])),action_0_list[fuel],width=width)
                            #plt.bar(range(initial,initial+len(action_1_list[fuel])),action_1_list[fuel],width=width)
                            initial += len(action_0_list[fuel])
                        # plt.show()
                        inv_dir = os.path.join(self.save_dir, 'inv')
                        if not os.path.isdir(inv_dir):
                            os.makedirs(inv_dir)
                        if fig_save_index % fig_save_interval == 0:
                            plt.savefig(os.path.join(inv_dir, "action0_"+str(self.learn_index)+ "_" + str(fig_save_index) + ".png"))

                        plt.cla()
                        plt.figure()
                        plt.title("action_1")
                        # plt.subplot(1,2,1)
                        initial = 0
                        for fuel in range(9):
                            plt.bar(range(initial, initial + len(action_1_list[fuel])), action_1_list[fuel],
                                    width=width)
                            # plt.bar(range(initial,initial+len(action_1_list[fuel])),action_1_list[fuel],width=width)
                            initial += len(action_1_list[fuel])
                        if fig_save_index % fig_save_interval == 0:
                            plt.savefig(os.path.join(inv_dir, "action1_"+str(self.learn_index)+ "_" + str(fig_save_index) + ".png"))
                        # plt.show()
                        #---inv model plot in 05/11---


                        # print(load_res)
                        # print(b.act.long())
                        action_shape = self.actor.net.module.action_shape if torch.cuda.is_available() else self.actor.action_shape

                        action_one_hot = nn.functional.one_hot(b.act.long(), action_shape).float()

                        inv_loss = nn.functional.mse_loss(pred[0], action_one_hot)
                        self.inv_optim.zero_grad()
                        inv_loss.backward()
                        self.inv_optim.step()
                        inverse_losses.append(inv_loss.item())
                print("learn_inv_inex:" + str(self.learn_index) + " inv_loss" + str(inv_loss.item()))

            # ---train mask with fixed inv model in 05/12---
            # for test
            # self.inv_model = load_model
            # ---train mask with fixed inv model in 05/12---

            if policy_update_start > self.learn_index % total_update_interval >= mask_update_start:
                #更新mask
                # epsilon = 1e-5
                fig_save_index = 0
                for step in range(repeat):

                    for b in batch.split(batch_size, merge_last=True):
                        fig_save_index +=1
                        # 利用p(a|s,s')学习mask(s,a)
                        #---todo in 04/29---
                        #学习mask(s,a),并且查看流程训练的步骤
                        #---todo in 04/29---
                        # ss = np.concatenate((b.obs, b.obs_next), axis=1)
                        # sa = np.concatenate((b.obs, b.act.unsqueeze(1)), axis=1)

                        delta_s = b.obs_next - b.obs
                        ss = np.concatenate((b.obs, delta_s), axis=1)
                        action_shape = self.actor.net.module.action_shape if torch.cuda.is_available() else self.actor.action_shape

                        action_one_hot = nn.functional.one_hot(b.act.long(), action_shape).bool()

                        mask_pred_all_action = self.mask_model(b.obs)[0]
                        mask_pred_current_action = torch.masked_select(mask_pred_all_action,action_one_hot)

                        dist = self(b).dist
                        target_log_pia = dist.log_prob(b.act)
                        with torch.no_grad():
                            # 从inv_model里面无梯度地取值用来训练mask
                            pi_a = self.inv_model(ss)[0]

                        pred_pi_act = torch.masked_select(pi_a, action_one_hot)
                        # KL factor
                        indepence_factor = torch.log(pred_pi_act) - target_log_pia
                        # TV factor
                        # indepence_factor = abs(target_log_pia.exp().float() / (pred_pi_act+epsilon) - 1)

                        #--- plot mask model in 5/12---
                        # mask_load_path = "/home/zdy/tianshou/test/discrete/log/RunningShooter/ppo/" \
                        #            "chances8_maxstep200_acpenalty0_maskTrue_mf-1e+02_totalinter2e+10_maskst0e+00_policyst1e+10_policyinitial0e+00_2022-05-12-15-18-46" \
                        #            "/mask.pth"
                        # mask_load_model = torch.load(mask_load_path)
                        # mask_load_all_res = mask_load_model(b.obs)[0]
                        # mask_load_current_res = torch.masked_select(mask_load_all_res, action_one_hot)

                        with torch.no_grad():
                            mask_load_current_res = mask_pred_current_action
                        mask_res_list = [[] for i in range(9)]
                        obs_copy = copy.deepcopy(b.obs)
                        rank = np.argsort(obs_copy, axis=0)[:, -1]
                        for i in rank:
                            fuel = int(obs_copy[i, -1] * 8)
                            mask_res_list[fuel].append(mask_load_current_res[i].item())
                            # action_1_list[fuel].append(factor[i, 1].item())
                            # print(obs_copy[i,-1])
                        initial = 0
                        width = 0.5
                        plt.cla()
                        plt.figure()
                        plt.title("mask")

                        for fuel in range(9):
                            plt.bar(range(initial, initial + len(mask_res_list[fuel])), mask_res_list[fuel],
                                    width=width)
                            # plt.bar(range(initial,initial+len(action_1_list[fuel])),action_1_list[fuel],width=width)
                            initial += len(mask_res_list[fuel])

                        mask_dir = os.path.join(self.save_dir, 'mask')
                        if not os.path.isdir(mask_dir):
                            os.makedirs(mask_dir)
                        if fig_save_index % fig_save_interval == 0:
                            plt.savefig(os.path.join(mask_dir,"mask_"+str(self.learn_index)+ "_" + str(fig_save_index) + ".png"))
                        # plt.show()
                        # --- plot mask model in 5/12---
                        mask_loss = (mask_pred_current_action - indepence_factor).pow(2).mean()
                        self.mask_optim.zero_grad()
                        mask_loss.backward()
                        self.mask_optim.step()
                        mask_losses.append(mask_loss.item())
                print("learn_mask_inex:" + str(self.learn_index) + " mask_loss" + str(mask_loss.item()))


            # 更新策略=更新pi+更新mask
            # 每40个回合里的后20个回合更新一次策略
            if self.learn_index % total_update_interval >=  policy_update_start:
                # 更新mask
                if torch.cuda.is_available():
                    self.actor.net.module.mask_model = copy.deepcopy(self.mask_model)
                else:
                    self.actor.mask_model = copy.deepcopy(self.mask_model)
                for step in range(repeat):
                    if self._recompute_adv and step > 0:
                        batch = self._compute_returns(batch, self._buffer, self._indices)
                    for b in batch.split(batch_size, merge_last=True):
                        # calculate loss for actor
                        dist = self(b).dist
                        if self._norm_adv:
                            mean, std = b.adv.mean(), b.adv.std()
                            b.adv = (b.adv - mean) / std  # per-batch norm
                        ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                        surr1 = ratio * b.adv
                        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                        if self._dual_clip:
                            clip1 = torch.min(surr1, surr2)
                            clip2 = torch.max(clip1, self._dual_clip * b.adv)
                            clip_loss = -torch.where(b.adv < 0, clip2, clip1).mean()
                        else:
                            clip_loss = -torch.min(surr1, surr2).mean()
                        # calculate loss for critic
                        value = self.critic(b.obs).flatten()
                        if self._value_clip:
                            v_clip = b.v_s + (value -
                                              b.v_s).clamp(-self._eps_clip, self._eps_clip)
                            vf1 = (b.returns - value).pow(2)
                            vf2 = (b.returns - v_clip).pow(2)
                            vf_loss = torch.max(vf1, vf2).mean()
                        else:
                            vf_loss = (b.returns - value).pow(2).mean()
                        # calculate regularization and overall loss
                        ent_loss = dist.entropy().mean()
                        loss = clip_loss + self._weight_vf * vf_loss \
                            - self._weight_ent * ent_loss
                        self.optim.zero_grad()
                        loss.backward()
                        if self._grad_norm:  # clip large gradient
                            nn.utils.clip_grad_norm_(
                                self._actor_critic.parameters(), max_norm=self._grad_norm
                            )
                        self.optim.step()
                        clip_losses.append(clip_loss.item())
                        vf_losses.append(vf_loss.item())
                        ent_losses.append(ent_loss.item())
                        losses.append(loss.item())
                print("learn_policy_inex:" + str(self.learn_index) + " policy_loss" + str(loss.item()))
            # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.learn_index % 100 == 0:
            print(self.learn_index)
            torch.save(self.inv_model, inv_path)
            torch.save(self.mask_model, mask_path)
            torch.save(self.actor, actor_path)

        self.learn_index += 1

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "loss/inv": inverse_losses,
            "loss/mask": mask_losses,
        }
