"""
This file implements the SRL algorithms.
Author: Tal Daniel
"""

# imports
import torch
import utils.utils as utils
import copy


def calc_fqi_matrices(nets, tgt_nets, batch, gamma, n_srl, m_batch_size=512, device='cpu',
                      use_boosting=False, train_actor=False):
    """
    This function calculates A and b tensors for the FQI solution.
    :param batch: batch of samples to extract features from (list)
    :param nets: networks to extract features from (nn.Module)
    :param tgt_nets: target networks from which Q values of next states are calculated (nn.Module)
    :param gamma: discount factor (float)
    :param n_srl: number of samples to include in the FQI solution
    :param m_batch_size: number of samples to calculate simultaneously (int)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_boosting: whether or not to use Boosted FQI
    :param: train_actor: whether or not to train actor net (bool)
    :return: A, A_bias, b, b_bias parameters for calculating the LS (np.arrays)
    """
    num_batches = n_srl // m_batch_size
    act_net, crt_net = nets
    tgt_act_net, tgt_crt_net = tgt_nets
    dim_act = act_net.fc2.out_features
    dim_crt = crt_net.out_fc1.out_features
    num_actions = act_net.fc3.out_features

    if train_actor:
        A_act = torch.zeros([dim_act * num_actions, dim_act * num_actions], dtype=torch.float32).to(device)
        A_act_bias = torch.zeros([1 * num_actions, 1 * num_actions], dtype=torch.float32).to(device)
        b_act = torch.zeros([dim_act * num_actions, 1], dtype=torch.float32).to(device)
        b_act_bias = torch.zeros([1 * num_actions, 1], dtype=torch.float32).to(device)

    A_crt = torch.zeros([dim_crt * 1, dim_crt * 1], dtype=torch.float32).to(device)
    A_crt_bias = torch.zeros([1 * 1, 1 * 1], dtype=torch.float32).to(device)
    b_crt = torch.zeros([dim_crt * 1, 1], dtype=torch.float32).to(device)
    b_crt_bias = torch.zeros([1 * 1, 1], dtype=torch.float32).to(device)

    for i in range(num_batches):
        idx = i * m_batch_size
        if i == num_batches - 1:
            states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch(batch[idx:], device)
        else:
            states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch(
                batch[idx: idx + m_batch_size], device)
        if train_actor:
            states_features_act = act_net.forward_to_last_hidden(states_v)
        states_features_crt = crt_net.forward_to_last_hidden(states_v, actions_v)

        # augmentation
        states_features_crt_bias = torch.ones([states_features_crt.shape[0], 1 * 1],
                                              dtype=torch.float32).to(device)
        if train_actor:
            states_features_act_aug = states_features_act.detach().repeat(
                (1, num_actions)).to(
                device)
            states_features_act_bias_aug = torch.ones([states_features_act.shape[0], 1 * num_actions],
                                                      dtype=torch.float32).to(device)

            states_features_act_mat = torch.mm(torch.t(states_features_act_aug), states_features_act_aug)
            states_features_act_bias_mat = torch.mm(torch.t(states_features_act_bias_aug), states_features_act_bias_aug)

        states_features_crt_mat = torch.mm(torch.t(states_features_crt.detach()), states_features_crt.detach())
        states_features_crt_bias_mat = torch.mm(torch.t(states_features_crt_bias), states_features_crt_bias)

        q_v = crt_net(states_v, actions_v)
        last_act_v = tgt_act_net.target_model(last_states_v)
        q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
        q_last_v[dones_mask] = 0.0
        q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma  # y_i

        if use_boosting:
            # calculate truncated bellman error
            bellman_error = q_ref_v.detach() - q_v.detach()
            truncated_bellman_error = bellman_error.clamp(-1, 1)

            if train_actor:
                b_act += torch.mm(torch.t(states_features_act_aug.detach()),
                                  truncated_bellman_error.detach().view(-1, 1))
                b_act_bias += torch.mm(torch.t(states_features_act_bias_aug),
                                       truncated_bellman_error.detach().view(-1, 1))

            b_crt += torch.mm(torch.t(states_features_crt.detach()),
                              truncated_bellman_error.detach().view(-1, 1))
            b_crt_bias += torch.mm(torch.t(states_features_crt_bias),
                                   truncated_bellman_error.detach().view(-1, 1))
        else:
            if train_actor:
                b_act += torch.mm(torch.t(states_features_act_aug.detach()),
                                  q_ref_v.detach().view(-1, 1))
                b_act_bias += torch.mm(torch.t(states_features_act_bias_aug),
                                       q_ref_v.detach().view(-1, 1))

            b_crt += torch.mm(torch.t(states_features_crt.detach()),
                              q_ref_v.detach().view(-1, 1))
            b_crt_bias += torch.mm(torch.t(states_features_crt_bias),
                                   q_ref_v.detach().view(-1, 1))
        if train_actor:
            A_act += states_features_act_mat.detach()
            A_act_bias += states_features_act_bias_mat
        A_crt += states_features_crt_mat.detach()
        A_crt_bias += states_features_crt_bias_mat
    if train_actor:
        A_act = (1.0 / n_srl) * A_act
        A_act_bias = (1.0 / n_srl) * A_act_bias
        b_act = (1.0 / n_srl) * b_act
        b_act_bias = (1.0 / n_srl) * b_act_bias

    A_crt = (1.0 / n_srl) * A_crt
    A_crt_bias = (1.0 / n_srl) * A_crt_bias
    b_crt = (1.0 / n_srl) * b_crt
    b_crt_bias = (1.0 / n_srl) * b_crt_bias

    if train_actor:
        return A_act, A_act_bias, b_act, b_act_bias, A_crt, A_crt_bias, b_crt, b_crt_bias
    else:
        return A_crt, A_crt_bias, b_crt, b_crt_bias


def calc_fqi_w_srl(a, a_bias, b, b_bias, w, w_b, lam=1.0, device='cpu'):
    """
    This function calculates the closed-form solution of the DQI algorithm.
    :param a: A matrix built from features (np.array)
    :param a_bias: same, but for bias
    :param b: b vector built from features and rewards (np.array)
    :param b_bias: same, but for bias
    :param w: weights of the last hidden layer in the DQN (np.array)
    :param w_b: bias weights
    :param lam: regularization parameter for the Least-Square (float)
    :param device: on which device to perform the calculation (cpu/gpu)
    :return: w_srl: retrained weights using FQI closed-form solution (np.array)
    """
    num_actions = w.shape[0]
    dim = w.shape[1]
    w = w.view(-1, 1)
    w_b = w_b.view(-1, 1)
    w_srl = torch.mm(torch.inverse(a + lam * torch.eye(num_actions * dim).to(device)), b + lam * w.detach())
    w_b_srl = torch.mm(torch.inverse(a_bias + lam * torch.eye(num_actions * 1).to(device)), b_bias + lam * w_b.detach())
    return w_srl.view(num_actions, dim), w_b_srl.squeeze()


def ls_step(nets, tgt_nets, batch, gamma, n_srl, lam=1.0, m_batch_size=256, device='cpu', use_boosting=False,
            sync_tgt=False):
    """
    This function performs the least-squares update on the last hidden layer weights.
    :param batch: batch of samples to extract features from (list)
    :param nets: networks to extract features from (nn.Module)
    :param tgt_nets: target networks from which Q values of next states are calculated (nn.Module)
    :param gamma: discount factor (float)
    :param n_srl: number of samples to include in the FQI solution
    :param lam: regularization parameter for the Least-Square (float)
    :param m_batch_size: number of samples to calculate simultaneously (int)
    :param device: on which device to perform the calculation (cpu/gpu)
    :param use_boosting: whether or not to use Boosted FQI
    :param sync_tgt: whether or not to sync target networks (bool)
    :return:
    """
    train_actor = False
    act_net, crt_net = nets
    tgt_act_net, tgt_crt_net = tgt_nets
    a_b_s = calc_fqi_matrices(nets, tgt_nets, batch, gamma,
                              n_srl, m_batch_size=m_batch_size, device=device, use_boosting=use_boosting,
                              train_actor=train_actor
                              )
    if train_actor:
        a_act, a_act_bias, b_act, b_act_bias, a_crt, a_crt_bias, b_crt, b_crt_bias = a_b_s
    else:
        a_crt, a_crt_bias, b_crt, b_crt_bias = a_b_s

    if train_actor:
        w_act_last_dict = copy.deepcopy(act_net.fc3.state_dict())
        w_act_last_dict_before = copy.deepcopy(act_net.fc3.state_dict())
        w_act_srl, w_b_act_srl = calc_fqi_w_srl(a_act.detach(), a_act_bias.detach(), b_act.detach(),
                                                b_act_bias.detach(),
                                                w_act_last_dict['weight'], w_act_last_dict['bias'], lam=lam,
                                                device=device)

    w_crt_last_dict = copy.deepcopy(crt_net.out_fc2.state_dict())
    w_crt_last_dict_before = copy.deepcopy(crt_net.out_fc2.state_dict())
    w_crt_srl, w_b_crt_srl = calc_fqi_w_srl(a_crt.detach(), a_crt_bias.detach(), b_crt.detach(),
                                            b_crt_bias.detach(),
                                            w_crt_last_dict['weight'], w_crt_last_dict['bias'], lam=lam,
                                            device=device)

    if train_actor:
        w_act_last_dict['weight'] = w_act_srl.detach()
        w_act_last_dict['bias'] = w_b_act_srl.detach()
        act_net.fc3.load_state_dict(w_act_last_dict)

        weight_diff_act = torch.sum((w_act_last_dict['weight'] - w_act_last_dict_before['weight']) ** 2)
        bias_diff_act = torch.sum((w_act_last_dict['bias'] - w_act_last_dict_before['bias']) ** 2)
        total_weight_diff_act = torch.sqrt(weight_diff_act + bias_diff_act)

    w_crt_last_dict['weight'] = w_crt_srl.detach()
    w_crt_last_dict['bias'] = w_b_crt_srl.detach().unsqueeze(-1)
    crt_net.out_fc2.load_state_dict(w_crt_last_dict)
    # weight diff
    weight_diff_crt = torch.sum((w_crt_last_dict['weight'] - w_crt_last_dict_before['weight']) ** 2)
    bias_diff_crt = torch.sum((w_crt_last_dict['bias'] - w_crt_last_dict_before['bias']) ** 2)
    total_weight_diff_crt = torch.sqrt(weight_diff_crt + bias_diff_crt)

    if sync_tgt:
        if train_actor:
            tgt_act_net.alpha_sync(alpha=1 - 1e-3)
        tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

    if train_actor:
        print("total weight difference of ls-update:: actor: %.3f" % total_weight_diff_act.item(),
              " critic: %.3f" % total_weight_diff_crt.item())
    else:
        print("total weight difference of ls-update:: critic: %.3f" % total_weight_diff_crt.item())
    print("least-squares step done.")
