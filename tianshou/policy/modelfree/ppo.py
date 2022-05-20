from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
import matplotlib.pyplot as plt

class PPOPolicy(A2CPolicy):
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

        for key,value in kwargs.items():
            print("{}={}".format(key,value))
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
        # # 初始化一个矩阵
        # state_discrete_num = self.state_discrete_num
        # action_num = self.action_num
        #
        # cal_ssa = np.zeros((state_discrete_num, state_discrete_num, action_num))
        # cal_ss = np.zeros((state_discrete_num, state_discrete_num))
        # cal_sa = np.zeros((state_discrete_num, action_num))
        # # 遍历buffer
        # buffer_len = batch["obs"].shape[0]
        # for ii in range(buffer_len):
        #     st = batch["obs"][ii]
        #     st_next = batch["obs_next"][ii]
        #     at = batch["act"][ii]
        #
        #     st_index = self.state_to_int(st)
        #     st_next_index = self.state_to_int(st_next)
        #     at_index = int(at)
        #     cal_ssa[st_index][st_next_index][at_index] += 1
        #     cal_ss[st_index][st_next_index] += 1
        #     cal_sa[st_index][at_index] += 1
        # self.cal_sa = self.cal_sa + cal_sa
        # self.cal_ss = self.cal_ss + cal_ss
        # self.cal_ssa =self.cal_ssa + cal_ssa
        #
        # dist = self(batch).dist
        # pi_p = dist.log_prob(batch.act).exp().float()
        # pi_p = pi_p.detach().numpy()
        # # p2 = P(at|st,st')
        # p2 = np.zeros(batch.act.shape)
        #
        # for ii in range(buffer_len):
        #     st = batch["obs"][ii]
        #     st_next = batch["obs_next"][ii]
        #     at = batch["act"][ii]
        #
        #     st_index = self.state_to_int(st)
        #     st_next_index = self.state_to_int(st_next)
        #     at_index = int(at)
        #
        #     p2[ii] = self.cal_ssa[st_index][st_next_index][at_index] / self.cal_ss[st_index][st_next_index]
        # self.actor.cal_ssa = self.cal_ssa
        # self.actor.cal_sa = self.cal_sa
        # self.actor.cal_ss = self.cal_ss
        #
        #
        # C_phi = p2 / pi_p - 1
        # fuel = batch["obs"][:, -1]
        #
        # #plt.plot(C_phi, label = "dependency factor", color='r')
        # #plt.plot(fuel, label = "fuel remain", color='g')
        # #plt.show()
        # fuel_average_C_phi = np.zeros((9))
        # for fuel_remain in range(9):
        #     a = C_phi[fuel == fuel_remain/8]
        #     a = np.abs(a)
        #     fuel_average_C_phi[fuel_remain] = np.average(a)
        #print("fuel_C_phi")
        #print(fuel_average_C_phi)
        #plt.bar(range(len(fuel_average_C_phi)), fuel_average_C_phi)
        #plt.show()


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





        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
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
                # clip_loss.backward()
                # vf_loss.backward()
                # ent_loss.backward()

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
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
