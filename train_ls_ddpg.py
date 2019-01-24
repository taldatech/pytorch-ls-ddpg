import os
import time
import gym
import pybullet_envs
# import argparse
from tensorboardX import SummaryWriter
import numpy as np
import utils.nn_agent_models as agent_model
import utils.Experience as Experience
import utils.utils as utils
# import utils
from utils.srl_algorithms import ls_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import random


# ENV_ID = "MinitaurBulletEnv-v0"
ENV_ID = "BipedalWalker-v2"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000  # 10000
N_DRL = 100000
N_SRL = REPLAY_SIZE
REWARD_TO_SOLVE = 300


TEST_ITERS = 10000  # 1000 for Minitaur


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = agent_model.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()
    training_random_seed = 2019
    use_constant_seed = True  # to compare performance independently of the randomness
    use_ls_ddpg = False
    use_boosting = False
    lam = 100  # regularization parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)
    name = "agent_" + ENV_ID
    if use_ls_ddpg:
        print("using LS-DDPG")
        name += "-LS-LAM-" + str(lam) + "-" + str(int(1.0 * N_DRL / 1000)) + "K"
    if use_boosting:
        print("using boosting")
        name += "-BOOSTING"
    if use_constant_seed:
        name += "-SEED-" + str(training_random_seed)
        np.random.seed(training_random_seed)
        random.seed(training_random_seed)
        env.seed(training_random_seed)
        test_env.seed(training_random_seed)
        torch.manual_seed(training_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_random_seed)
        print("training using constant seed of ", training_random_seed)
    name += "-BATCH-" + str(BATCH_SIZE)
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
    agent = agent_model.AgentDDPG(act_net, device=device)
    exp_source = Experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = Experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

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
                        print("environment solved in % steps" % frame_idx, " (% episodes)" % len(tracker.total_rewards))
                        utils.save_agent_state(act_net, crt_net, [act_opt, crt_opt], frame_idx,
                                               len(tracker.total_rewards), path=ckpt_save_path)
                        break

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                drl_updates += 1
                # LS-UPDATE STEP for Critic (Q)
                if use_ls_ddpg and (drl_updates % N_DRL == 0) and (len(buffer) >= N_SRL):
                    # if len(buffer) > 1:
                    print("performing ls step...")
                    batch = buffer.sample(N_SRL)
                    ls_step([act_net, crt_net], [tgt_act_net, tgt_crt_net], batch, GAMMA, len(buffer),
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

                if frame_idx % TEST_ITERS == 0:
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
