from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch
from tianshou.utils.net.common import MLP
import matplotlib.pyplot as plt
from tianshou.env import RunningMan

class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
        mask: bool = False
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.action_shape = action_shape
        self.output_dim = int(np.prod(action_shape))
        # 这里网络直接输出相当于是每一维动作可选动作数量的乘积
        # mask的时候就需要注意，之后在开发
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self.softmax_output = softmax_output
        self.mask = mask
        self.cal_ss = None
        self.cal_ssa = None
        self.cal_sa = None
        self.fuel_C_phi = [[0.0,] for i in range(9)]
        self.index = 0
        # todo:
        #
        self.env_for_singlestepsim = RunningMan()
        self.target = None
        self.max_lenth = 200
        self.action_chances = 8

    def state_to_int(self,state):
        # 对于离散环境，把状态对应到int值上去
        max_lenth = self.max_lenth
        action_chances = self.action_chances
        # state = [0.005,1.,0.875]
        return int(state[0] * max_lenth + state[1] * (max_lenth+1) + state[2] * action_chances * (max_lenth+1) * 2)



    # def fuel_mask(self,state):
    #     max_lenth = self.max_lenth
    #     action_chances = self.action_chances


    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state)
        logits = self.last(logits)
        # logits: 也许很多行
        # tensor([[-0.0378, -0.0917],
        #         [-0.0202, -0.1122]])
        logits = F.softmax(logits, dim=-1)
        invalid_action_masks = torch.ones_like(logits)
        # print(logits.shape)
        # s:ndarry,和logits行数相同，最后一列是燃料剩余量
        # [[0.11000,0.00000,1.00000],
        # [0.11000,1.00000,1.00000]]
        if self.softmax_output:
            if self.mask:
                # todo:
                # 依次对s里的每一个状态做处理
                # 如果s里的info_list 是空{}，那么就直接返回logits的值
                # 反之，如果infolist里带有原环境的信息，就重建环境，然后单步仿真

                # 给定的状态，按照不同的动作单步仿真，如果结果相同，就mask成为一个动作
                # 构建一个环境，遍历一遍动作，如果下个状态完全一致，就不执行了。
                for index in range(len(s)):
                    if len(info.shape) == 0 :
                        break
                    dist_samples = 10
                    # print(info)
                    # print(info[index])discrete.py
                    temp_env = RunningMan()
                    temp_env.copy_as_infolist(info[index])

                    # todo:
                    # 2022/04/06 利用相关性判据来判断特定状态下采样的动作与下一个状态是否独立
                    # 维护两个列表，一个保存动作，另一个保存下一个状态
                    # 再利用两个列表的数据计算相关性因子（0-1）
                    # 如果相关性因子小于特定阈值，开始屏蔽
                    # 先不屏蔽，暂时评判屏蔽机制与fuel剩余量的关系

                    state_next = []
                    act_next = torch.randint(0, temp_env.n_actions, [dist_samples, 1])
                    # state_next有n_actions个列表，每个列表记录了每个动作对应的后一个状态

                    for ii in range(dist_samples):
                        temp_env = RunningMan()
                        temp_env.copy_as_infolist(info[index])
                        act_test = act_next[ii]
                        obs, reward, done, _ = temp_env.step(act_test)
                        state_next.append(obs)
                    #仅使用状态的值是否相同来判定动作是否应该被mask
                    # 找到结果相同的状态对应的动作，只执行其中的第一个动作，其他全部概率压到0
                    # todo：
                    # 用分布的衡量评判两个动作之后的状态是否一致
                    state_next = np.array(state_next)
                    act_next = act_next.numpy()

                    # 比较两个numpy数组的相关性
                    r = correlation_dist(state_next, act_next)
                    # 统计fuel数与c_phi_max的关系：fuel剩余不同数目时的c_phi_max值

                    self.fuel_C_phi[int(8 * s[index][-1])].append(r)
                    fuel_average_C_phi = [np.average(self.fuel_C_phi[i]) for i in range(9)]

                    plt.bar(range(len(fuel_average_C_phi)), fuel_average_C_phi)
                    # plt.legend()
                    if self.index % 100 == 0:
                        plt.savefig('D:\\zhongdy\\research\\tianshou-master\\tianshou-master\\tianshou\\utils\\net\\fig_r\\fig_{}.jpg'.format(str(self.index)))
                    self.index += 1
                    plt.close()

                    if r < 0.4:#(state_next[0][0] == state_next[1][0]).all():
                        # 只比较两个动作的后一个状态集合的第一个元素【因为只做了一次实验】
                        # 如果相同的话，就mask后一个
                        invalid_action_masks[index][-1] = 0
                    # print(" action nonsense")

                invalid_action_masks = invalid_action_masks.type(torch.BoolTensor).to(self.device)
                logits = torch.where(invalid_action_masks, logits, torch.tensor(-1e+8).to(self.device))

                logits = F.softmax(logits, dim=-1)

                # # 针对燃料受限制的问题，把燃料为0的时候的动作全mask掉
                # # s[i,-1] <= 1e-5 时，mask[i]=[1,0]
                # # else , mask[i] = [1,1]
                # # logits = mask dot logits
                # invalid_action_masks = torch.ones_like(logits)
                # if self.cal_ssa is None or self.cal_ssa is None or self.cal_sa is None:
                #     pass
                #     # invalid_action_masks = torch.ones_like(logits)
                # else:
                #     state_discrete_num = self.state_to_int([1, 1, 1]) + 1
                #     epsilon = 0.6
                #
                #     for index in range(s.shape[0]):
                #         # index 是指 在s里的多个状态的index顺序
                #         st_index = self.state_to_int(s[index])
                #         #st_index 是指某个状态对应的离散值
                #         for at_index in range(self.action_shape):
                #             c_phi_max = -1.0
                #             for st_next_index in range(1, state_discrete_num):
                #             # 对所有的后续状态来说
                #                 first_identifier = 0
                #                 if self.cal_ss[st_index][st_next_index] > 0:
                #                     first_identifier = 1
                #                     p2 = self.cal_ssa[st_index][st_next_index][at_index] / self.cal_ss[st_index][st_next_index]
                #                     logits = F.softmax(logits, dim=-1)
                #                     pi_p = logits[index][at_index].exp().float()
                #                     C_phi = p2 / pi_p - 1
                #                     C_phi = first_identifier * abs(C_phi.item())
                #                 else:
                #                     first_identifier = 0
                #                     C_phi = -1.0
                #
                #
                #
                #
                #
                #                 if C_phi > c_phi_max:
                #                     c_phi_max = C_phi
                #                     #print("c_phi_max:", c_phi_max)
                #             #分别统计 fuel剩余燃料不同时候的c_phi_average
                #             #print("index",index)
                #             #print("fuel",s[index][-1])
                #             #print("c_phi_max:", c_phi_max)
                #             #统计fuel数与c_phi_max的关系：fuel剩余不同数目时的c_phi_max值
                #
                #             self.fuel_C_phi[int(8 * s[index][-1])].append(c_phi_max)
                #             fuel_average_C_phi = [np.average(self.fuel_C_phi[i]) for i in range(9)]
                #
                #             plt.bar(range(len(fuel_average_C_phi)), fuel_average_C_phi)
                #             #plt.legend()
                #             if self.index % 5000 == 0:
                #                 plt.savefig('fig_{}.jpg'.format(str(self.index)))
                #             self.index += 1
                #             #plt.plot(range(9),fuel_average_C_phi)
                #             plt.close()
                #
                #
                #             if c_phi_max < epsilon:
                #                 #pass
                #                 #如果index对应的状态下做at_index的动作无效，加一个mask
                #                 invalid_action_masks[index][at_index] = 0
                #     for index in range(s.shape[0]):
                #         #对所有的mask，如果所有的值都比较小（都被mask了），执行第一个动作
                #         if sum(invalid_action_masks[index]) <= 1e-5:
                #
                #             invalid_action_masks[index][0] = 1
                # # for index in range(s.shape[0]):
                # #     if s[index][-1] <= 1e-5:
                # #         invalid_action_masks[index][-1] = 0
                #     invalid_action_masks = invalid_action_masks.type(torch.BoolTensor).to(self.device)
                #     logits = torch.where(invalid_action_masks, logits, torch.tensor(-1e+8).to(self.device))
                # logits = F.softmax(logits, dim=-1)
            else:
                logits = F.softmax(logits, dim=-1)
        return logits, h


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, last_size, hidden_sizes, device=self.device)

    def forward(
        self, s: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(s, state=kwargs.get("state", None))
        return self.last(logits)


class CosineEmbeddingNetwork(nn.Module):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list \
    of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device
        ).view(1, 1, self.num_cosines)
        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi
                            ).view(batch_size * N, self.num_cosines)
        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)
        return tau_embeddings


class ImplicitQuantileNetwork(Critic):
    """Implicit Quantile Network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        last_size = np.prod(action_shape)
        super().__init__(
            preprocess_net, hidden_sizes, last_size, preprocess_net_output_dim, device
        )
        self.input_dim = getattr(
            preprocess_net, "output_dim", preprocess_net_output_dim
        )
        self.embed_model = CosineEmbeddingNetwork(num_cosines,
                                                  self.input_dim).to(device)

    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor], sample_size: int, **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(
            batch_size, sample_size, dtype=logits.dtype, device=logits.device
        )
        embedding = (logits.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), h


class FractionProposalNetwork(nn.Module):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_fractions)
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        torch.nn.init.constant_(self.net.bias, 0)
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def forward(
        self, state_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate (log of) probabilities q_i in the paper.
        m = torch.distributions.Categorical(logits=self.net(state_embeddings))
        taus_1_N = torch.cumsum(m.probs, dim=1)
        # Calculate \tau_i (i=0,...,N).
        taus = F.pad(taus_1_N, (1, 0))
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        # Calculate entropies of value distributions.
        entropies = m.entropy()
        return taus, tau_hats, entropies


class FullQuantileFunction(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__(
            preprocess_net, action_shape, hidden_sizes, num_cosines,
            preprocess_net_output_dim, device
        )

    def _compute_quantiles(
        self, obs: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (obs.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        quantiles = self.last(embedding).view(batch_size, sample_size,
                                              -1).transpose(1, 2)
        return quantiles

    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor],
        propose_model: FractionProposalNetwork,
        fractions: Optional[Batch] = None,
        **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits.detach())
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        # Calculate quantiles_tau for computing fraction grad
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), h


class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param int in_features: the number of input features.
    :param int out_features: the number of output features.
    :param float noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(
        self, in_features: int, out_features: int, noisy_std: float = 0.5
    ) -> None:
        super().__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.randn(x.size(0), device=x.device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample(self) -> None:
        self.eps_p.copy_(self.f(self.eps_p))  # type: ignore
        self.eps_q.copy_(self.f(self.eps_q))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * (
                self.eps_q.ger(self.eps_p)  # type: ignore
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


def sample_noise(model: nn.Module) -> bool:
    """Sample the random noises of NoisyLinear modules in the model.

    :param model: a PyTorch module which may have NoisyLinear submodules.
    :returns: True if model has at least one NoisyLinear submodule;
        otherwise, False.
    """
    done = False
    for m in model.modules():
        if isinstance(m, NoisyLinear):
            m.sample()
            done = True
    return done

def correlation_dist(v1, v2):
    A_v1 = cal_A(v1)
    A_v2 = cal_A(v2)

    X_dot_Y = A_v1 * A_v2
    X_dot_X = A_v1 * A_v1
    Y_dot_Y = A_v2 * A_v2

    if np.sqrt(np.mean(X_dot_X) * np.mean(Y_dot_Y)) <= 1e-7:
        r = 0
    else:
        r = np.mean(X_dot_Y) / np.sqrt(np.mean(X_dot_X) *np.mean(Y_dot_Y))

    return r
def cal_A(v):
    # num为数据的条数，p是数据的维度
    [num, p] = v.shape
    matrix_v = np.zeros([num, num])
    for i in range(num):
        for j in range(i, num):
            delta = v[i] - v[j]
            matrix_v[i][j] = np.linalg.norm(delta, ord=p)
            matrix_v[j][i] = matrix_v[i][j]
    row_mean = np.mean(matrix_v, axis=1)
    row_mean = np.repeat(np.expand_dims(row_mean, axis=1), num, axis=1)
    col_mean = np.mean(matrix_v, axis=0)
    col_mean = np.repeat(np.expand_dims(col_mean, axis=0), num, axis=0)
    total_mean = np.mean(matrix_v)
    A_v1 = matrix_v - row_mean - col_mean + total_mean
    return  A_v1