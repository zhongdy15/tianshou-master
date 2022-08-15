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
        mask: bool = False,
        mask_factor: float = -1e10,
        default_actionindex:int = 0
    ) -> None:
        super().__init__()
        self.device = device
        # print("line51__device:"+str(device))
        self.preprocess = preprocess_net
        self.action_shape = action_shape
        self.output_dim = int(np.prod(action_shape))
        # 这里网络直接输出相当于是每一维动作可选动作数量的乘积
        # mask的时候就需要注意，之后在开发
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self.softmax_output = softmax_output

        self.mask_model = None
        self.mask = mask
        # self.cal_ss = None
        # self.cal_ssa = None
        # self.cal_sa = None
        # self.fuel_C_phi = [[0.0,] for i in range(9)]
        self.index = 0

        # todo:
        #
        # self.env_for_singlestepsim = RunningMan()
        # self.target = None
        # self.max_lenth = 200
        # self.action_chances = 8
        self.mask_factor = mask_factor

        self.use_prior_mask =  True
        self.default_actionindex = default_actionindex

        self.filesavedir = "/home/zdy/home/zdy/tianshou/test/discrete/log/ActionBudget_ALE/AirRaid-v5/ppo/state_save/"
        import os
        if not os.path.isdir(self.filesavedir):
            os.makedirs(self.filesavedir)
        self.filesavepath = os.path.join(self.filesavedir,"state.txt")

        if self.mask and not self.use_prior_mask:
            #如果要用mask，但是不用先验的mask
            self.threshold = 0.45
            mask_pth = "/home/zdy/home/zdy/tianshou/test/discrete/log/ActionBudget_ALE/AirRaid-v5/ppo/mask_2022-08-13-21-09-37/"
            self.mask_model = torch.load(mask_pth+"mask.pth", map_location=device)
            self.mask_pth = mask_pth


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
        # logits = F.softmax(logits, dim=-1)
        # print(logits)
        invalid_action_masks = torch.ones_like(logits)
        # print("initial_mask")
        # print(invalid_action_masks)
        r = self.mask_factor
        # import os
        # file_path = os.path.dirname(os.path.abspath(__file__))
        # print(file_path)
        # print("use discrete in 104")
        # print(logits.shape)
        # s:ndarry,和logits行数相同，最后一列是燃料剩余量
        # [[0.11000,0.00000,1.00000],
        # [0.11000,1.00000,1.00000]]
        if self.mask:
            # print("use discrete in 110")
            if self.use_prior_mask:
                # print("use discrete in 112")
                # print("use prior mask!!")
                if "fuel_remain" in info.keys():
                    fuel_remain = info["fuel_remain"]
                    fuel_flag = fuel_remain > 0
                    fuel_flag = torch.tensor(fuel_flag).to(self.device)
                else:
                    fuel_flag = torch.ones(s.shape[0]).type(torch.BoolTensor).to(self.device)
                # print("fuel_flag")
                # print(fuel_flag)

                # print(fuel_flag)
                # print("s_shape"+str(s.shape[0]))
                for ii in range(s.shape[0]):
                    if fuel_flag[ii]:
                        # invalid_action_masks[ii] = 1
                        pass
                    else:
                        logits[ii] = torch.ones_like(logits[ii])
                        # invalid_action_masks[ii] = 0
                        # invalid_action_masks[ii][self.default_actionindex] = 1

                # invalid_action_masks = invalid_action_masks.type(torch.BoolTensor).to(self.device)
                # logits = torch.where(invalid_action_masks, logits, torch.tensor(-1e+8).to(self.device))
            else:
                # if s[0,0,0,0] % 10 == 1 :
                #     print("test!")
                # if "fuel_remain" in info.keys():
                #     fuel_remain = info["fuel_remain"]
                #     print(fuel_remain)
                with torch.no_grad():
                    mask_pred_all_action = self.mask_model.forward(s.permute((0, 3, 1, 2)) / 255)
                mask_factor_max = torch.max(mask_pred_all_action, dim=1).values
                # print(mask_factor_max)

                filename = self.filesavepath

                file_object =  open(filename, 'a')


                for ii in range(s.shape[0]):
                    file_object.write(str(s[ii,0,0,0].item()) + "    "+str(mask_factor_max[ii].item())+"\n")


                    if s[ii,0,0,0] > 0:
                        invalid_action_masks[ii] = 1
                    else:
                        invalid_action_masks[ii] = 0
                        invalid_action_masks[ii][self.default_actionindex] = 1
                invalid_action_masks = invalid_action_masks.type(torch.BoolTensor).to(self.device)
                # logits_clone = logits.clone()
                logits = torch.where(invalid_action_masks, logits, torch.tensor(-1e+8).to(self.device))
                file_object.close()


        else:
            # for test in 0725
            # test maskmodel in no mask ppo

            # mask_factor
            save_csv_flag = False
            if save_csv_flag:
                with torch.no_grad():
                    mask_pred_all_action = self.mask_model.forward(s.permute((0, 3, 1, 2)) / 255)
                mask_factor_max = torch.max(mask_pred_all_action, dim=1).values
                mask_pred_all_action = mask_pred_all_action.cpu().numpy()
                #fuel_remain
                if "fuel_remain" in info.keys():
                    fuel_remain = info["fuel_remain"]
                else:
                    fuel_remain = np.ones(s.shape[0]) * 800

                if "frame_number" in info.keys():
                    frame_number = info["frame_number"]
                else:
                    frame_number = np.ones(s.shape[0]) * 0

                import csv
                import os
                file = self.mask_pth + "save_mask_factor_0727.csv"
                with open(file, 'a', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['frame_number', 'fuel_remain', 'action_0', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5'])
                    row =  np.concatenate((np.atleast_2d(frame_number).T,np.atleast_2d(fuel_remain).T,mask_pred_all_action),axis=1)
                    # writer.writerows([[0,10,0.2,0.2,0.1,0.1,0.1,0.2]])
                    writer.writerows(row)

            # print("to save in csv")

        # self.invalid_action_masks = invalid_action_masks
        # print(invalid_action_masks)
        # print("use discrete in 154")
        if self.softmax_output:
            # print("use discrete in 156")
            logits = F.softmax(logits, dim=-1)

        # if "fuel_remain" in info.keys():
        #     if min(info["fuel_remain"])<=0:
        #         print("info:")
        #         print(info["fuel_remain"])
        #         print("logits:")
        #         print(logits)
        # if "fuel_remain" in info.keys():
        #     for ii in range(logits.shape[0]):
        #         if info["fuel_remain"][ii] == 0:
        #             no_fuel_logits = logits[ii]
        #             if torch.min(no_fuel_logits) > 1e-6:
        #                 print("error in discrete.py:166")
        #                 print(no_fuel_logits)
        # print("use discrete in 172")
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