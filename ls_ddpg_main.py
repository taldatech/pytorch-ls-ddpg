"""
Main application for Least-Squares Deep Deterministic Policy Gradients
"""

import os
import time
import gym
import pybullet_envs
from tensorboardX import SummaryWriter
import numpy as np
import utils.nn_agent_models as agent_model
import utils.Experience as Experience
import utils.utils as utils
from utils.srl_algorithms import ls_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from utils.utils import test_net
import argparse

REWARD_TO_SOLVE = 300  # mean reward the environment is considered SOLVED

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train and play an LS-DQN agent")
    # modes
    parser.add_argument("-t", "--train", help="train or continue training an agent",
                        action="store_true")
    parser.add_argument("-k", "--lsddpg", help="use LS-DDPG",
                        action="store_true")
    parser.add_argument("-j", "--boosting", help="use boosting",
                        action="store_true")
    parser.add_argument("-p", "--play", help="play the environment using an a pretrained agent",
                        action="store_true")
    parser.add_argument("-y", "--path", type=str, help="path to agent checkpoint, for playing")
    # arguments
    # for training and playing
    parser.add_argument("-n", "--name", type=str,
                        help="model name, for saving and loading,"
                             " if not set, training will continue from a pretrained checkpoint")
    parser.add_argument("-e", "--env", type=str,
                        help="environment to play: MinitaurBulletEnv-v0, BipedalWalker-v2", default="BipedalWalker-v2")
    # for training
    parser.add_argument("-d", "--decay_rate", type=int,
                        help="number of episodes for epsilon decaying, default: 500000")
    parser.add_argument("-o", "--optimizer", type=str,
                        help="optimizing algorithm ('RMSprop', 'Adam'), deafult: 'Adam'")
    parser.add_argument("--lr_actor", type=float,
                        help="learning rate for the Actor optimizer, default: 0.0001")
    parser.add_argument("--lr_critic", type=float,
                        help="learning rate for the Critic optimizer, default: 0.0001")
    parser.add_argument("-l", "--lam", type=float,
                        help="regularization parameter value, default: 1, 10000 (boosting)")
    parser.add_argument("-g", "--gamma", type=float,
                        help="gamma parameter for the Q-Learning, default: 0.99")
    parser.add_argument("-s", "--buffer_size", type=int,
                        help="Replay Buffer size, default: 1000000")
    parser.add_argument("-a", "--n_drl", type=int,
                        help="number of drl updates before an srl update, default: 100000")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="number of samples in each batch, default: 64")
    parser.add_argument("-i", "--steps_to_start_learn", type=int,
                        help="number of steps before the agents starts learning, default: 10000")
    parser.add_argument("-c", "--test_iter", type=int,
                        help="number of iterations between policy testing, default: 10000")
    # for playing
    parser.add_argument("-x", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.lsddpg:
        use_ls_ddpg = True
    else:
        use_ls_ddpg = False
    if args.boosting:
        use_boosting = True
        lam = 1000
    else:
        use_boosting = False
        lam = 1

    # Training
    if args.train:
        if args.name:
            model_name = args.name
        else:
            model_name = ''
        if args.decay_rate:
            decay_rate = args.decay_rate
        if args.lr_actor:
            lr_actor = args.lr_actor
        if args.lr_critic:
            lr_critic = args.lr_critic
        if args.lam:
            lam = args.lam
        if args.gamma:
            gamma = args.gamma
        else:
            gamma = 0.99
        if args.buffer_size:
            replay_size = args.buffer_size
        else:
            replay_size = 100000
        if args.n_drl:
            n_drl = args.n_drl
        else:
            n_drl = 100000  # steps of DRL between SRL
        if args.batch_size:
            batch_size = args.batch_size
        else:
            batch_size = 64
        if args.steps_to_start_learn:
            steps_to_start_learn = args.steps_to_start_learn
        else:
            steps_to_start_learn = 10000
        if args.test_iter:
            test_iter = args.test_iter
        else:
            test_iter = 10000

        # training_random_seed = 2019
        save_freq = 50000
        n_srl = replay_size  # size of batch in SRL step
        # use_constant_seed = False  # to compare performance independently of the randomness

        model_saving_path = './agent_ckpt/agent_' + model_name + ".pth"
        # if use_constant_seed:
        #     model_name += "-SEED-" + str(training_random_seed)
        #     np.random.seed(training_random_seed)
        #     random.seed(training_random_seed)
        #     env.seed(training_random_seed)
        #     torch.manual_seed(training_random_seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(training_random_seed)
        #     print("training using constant seed of ", training_random_seed)
        env = gym.make(args.env)
        test_env = gym.make(args.env)
        name = model_name + "_agent_" + args.env
        if use_ls_ddpg:
            print("using LS-DDPG")
            name += "-LS-LAM-" + str(lam) + "-" + str(int(1.0 * n_drl / 1000)) + "K"
        if use_boosting:
            print("using boosting")
            name += "-BOOSTING"
        # if use_constant_seed:
        #     name += "-SEED-" + str(training_random_seed)
        #     np.random.seed(training_random_seed)
        #     random.seed(training_random_seed)
        #     env.seed(training_random_seed)
        #     test_env.seed(training_random_seed)
        #     torch.manual_seed(training_random_seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(training_random_seed)
        #     print("training using constant seed of ", training_random_seed)
        name += "-BATCH-" + str(batch_size)
        save_path = os.path.join("saves", "ddpg-" + name)
        os.makedirs(save_path, exist_ok=True)
        ckpt_save_path = './agent_ckpt/' + name + ".pth"
        if not os.path.exists('./agent_ckpt/'):
            os.makedirs('./agent_ckpt')

        act_net = agent_model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        crt_net = agent_model.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        print(act_net)
        print(crt_net)
        tgt_act_net = agent_model.TargetNet(act_net)
        tgt_crt_net = agent_model.TargetNet(crt_net)

        writer = SummaryWriter(comment="-ddpg-" + name)
        if decay_rate is not None:
            agent = agent_model.AgentDDPG(act_net, device=device, ou_decay_steps=decay_rate)
        else:
            agent = agent_model.AgentDDPG(act_net, device=device)
        exp_source = Experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
        buffer = Experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)
        if args.optimizer and args.optimizer == "RMSprop":
            act_opt = optim.RMSprop(act_net.parameters(), lr=lr_actor)
            crt_opt = optim.RMSprop(crt_net.parameters(), lr=lr_critic)
        else:
            act_opt = optim.Adam(act_net.parameters(), lr=lr_actor)
            crt_opt = optim.Adam(crt_net.parameters(), lr=lr_critic)

        utils.load_agent_state(act_net, crt_net, [act_opt, crt_opt], path=ckpt_save_path)

        frame_idx = 0
        drl_updates = 0
        best_reward = None
        with utils.RewardTracker(writer) as tracker:
            with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
                while True:
                    frame_idx += 1
                    buffer.populate(1)
                    rewards_steps = exp_source.pop_rewards_steps()
                    if rewards_steps:
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track("episode_steps", steps[0], frame_idx)
                        mean_reward = tracker.reward(rewards[0], frame_idx)
                        if mean_reward is not None and mean_reward > REWARD_TO_SOLVE:
                            print("environment solved in % steps" % frame_idx,
                                  " (% episodes)" % len(tracker.total_rewards))
                            utils.save_agent_state(act_net, crt_net, [act_opt, crt_opt], frame_idx,
                                                   len(tracker.total_rewards), path=ckpt_save_path)
                            break

                    if len(buffer) < steps_to_start_learn:
                        continue

                    batch = buffer.sample(batch_size)
                    states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch(batch, device)

                    # train critic
                    crt_opt.zero_grad()
                    q_v = crt_net(states_v, actions_v)
                    last_act_v = tgt_act_net.target_model(last_states_v)
                    q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                    q_last_v[dones_mask] = 0.0
                    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma
                    critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                    critic_loss_v.backward()
                    crt_opt.step()
                    tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                    tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                    drl_updates += 1
                    # LS-UPDATE STEP for Critic (Q)
                    if use_ls_ddpg and (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
                        # if len(buffer) > 1:
                        print("performing ls step...")
                        batch = buffer.sample(n_srl)
                        ls_step([act_net, crt_net], [tgt_act_net, tgt_crt_net], batch, gamma, len(buffer),
                                lam=lam, m_batch_size=256, device=device, use_boosting=use_boosting)

                    # train actor
                    act_opt.zero_grad()
                    cur_actions_v = act_net(states_v)
                    actor_loss_v = -crt_net(states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    act_opt.step()
                    tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                    tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if frame_idx % test_iter == 0:
                        ts = time.time()
                        rewards, steps = test_net(act_net, test_env, device=device)
                        print("Test done in %.2f sec, reward %.3f, steps %d" % (
                            time.time() - ts, rewards, steps))
                        writer.add_scalar("test_reward", rewards, frame_idx)
                        writer.add_scalar("test_steps", steps, frame_idx)
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                                fname = os.path.join(save_path, name)
                                torch.save(act_net.state_dict(), fname)
                                utils.save_agent_state(act_net, crt_net, [act_opt, crt_opt], frame_idx,
                                                       len(tracker.total_rewards), path=ckpt_save_path)
                            best_reward = rewards

        pass
    elif args.play:
        # play
        if args.path:
            path_to_model_ckpt = args.path
        else:
            raise SystemExit("must include path to agent checkpoint")
        render = True
        spec = gym.envs.registry.spec(args.env)
        if spec._kwargs.get('render') and render:
            spec._kwargs['render'] = True
        env = gym.make(args.env)
        if args.record:
            env = gym.wrappers.Monitor(env, args.record)

        net = agent_model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
        net.load_state_dict(torch.load(path_to_model_ckpt))

        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if render:
                env.render()
            if done:
                env.close()
                break
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    else:
        raise SystemExit("must choose between train or play")

