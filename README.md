# pytorch-ls-ddpg
PyTorch Implementation of Least-Squares Deep Deterministic Policy Gradients

Based on the paper:

Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017

Video:

[YouTube](https://youtu.be/i8Cnas7QrMc) - https://youtu.be/i8Cnas7QrMc


![minitaur](https://github.com/taldatech/pytorch-ls-ddpg/blob/master/images/minitaur.gif)
![bipedal](https://github.com/taldatech/pytorch-ls-ddpg/blob/master/images/bipedal.gif)

![ddpg](https://github.com/taldatech/pytorch-ls-ddpg/blob/master/images/ddpg_graph.png)

[LS-DQN](https://github.com/taldatech/pytorch-ls-dqn) - https://github.com/taldatech/pytorch-ls-dqn

- [pytorch-ls-ddpg](#pytorch-ls-ddpg)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [API (`ls_ddpg_main.py --help`)](#api---ls-ddpg-mainpy---help--)
  * [Playing](#playing)
  * [Training](#training)
  * [TensorBoard](#tensorboard)
  * [References](#references)

## Background
The idea of this algorithm is to combine between Deep RL (DRL) to Shallow RL (SRL), where in this case, we use Deep Deterministic Policy Gradient (DDPG) as the DRL algorithm and
Fitted Q-Iteration (FQI) and the Boosted version (B-FQI) as the SRL algorithm (which can be approximated using least-squares, full derivation is in the original paper).
Every N_DRL (number of DDPG Critic backprop steps) we apply LS-UPDATE to the very last layer of the Critic NN, by using the complete Replay Buffer, a fetaure augmentation technique and
Bayesian regularization (prevents overfitting, makes the LS matrix invertible) to solve the FQI equations.
The assumptions are that the features extracted from the last layer form a rich representation, and that the large batch size used by the SRL algorithm improves stability and performance.
For full derivations and theory, please refer to the original paper.

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|
|`gym`|  `0.10.9`|
|`tensorboard`|  `1.12.0`|
|`tensorboardX`|  `1.5`|
|`pybullet`| `2.4.2`, https://pypi.org/project/pybullet/|
|`Box2D`| `2.3.8`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`ls_ddpg_main.py`| general purpose main application for training/playing a LS-DDPG agent|
|`play_ls_ddpg.py`| sample code for playing an environment, also in `ls_ddpg_main.py`|
|`train_ls_ddpg.py`| sample code for training an environment, also in `ls_ddpg_main.py`|
|`nn_agent_models.py`| agent and DDPG classes, holds the network, action selector and current state|
|`Experience.py`| Replay Buffer classes|
|`srl_algorithms.py`| Shallow RL algorithms, LS-UPDATE|
|`utils.py`| utility functions|
|`*.pth` / `*.dat`| Checkpoint files for the Agents (playing/continual learning)|
|`Deep_RL_Shallow_Updates_for_Deep_Reinforcement_Learning.pdf`| Writeup - theory and results|

## API (`ls_ddpg_main.py --help`)


You should use the `ls_ddpg_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading |
|-k, --lsddpg	| use LS-DDPG (apply LS-UPDATE every N_DRL), default: false |
|-j, --boosting| use Boosted-FQI as SRL algorithm, default: false |
|-y, --path| path to agent checkpoint, for playing |
|-e, --env| environment to play: MinitaurBulletEnv-v0, BipedalWalker-v2, default="BipedalWalker-v2" |
|-d, --decay_rate| number of episodes for epsilon decaying, default: 500000 |
|-o, --optimizer| optimizing algorithm ('RMSprop', 'Adam'), deafult: 'Adam' |
|--lr_critic| learning rate for the Critic optimizer, default: 0.0001 |
|--lr_actor| learning rate for the Actor optimizer, default: 0.0001 |
|-g, --gamma| gamma parameter for the Q-Learning, default: 0.99 |
|-l, --lam| regularization parameter value, default: 1, 10000 (boosting) |
|-s, --buffer_size| Replay Buffer size, default: 1000000 |
|-b, --batch_size| number of samples in each batch, default: 64 |
|-i, --steps_to_start_learn| number of steps before the agents starts learning, default: 10000 |
|-c, --test_iter| number of iterations between policy testing, default: 10000 |
|-x, --record| Directory to store video recording when playing (only Linux) |
|--no-visualize| if not typed, render the environment when playing |

## Playing
Agents checkpoints (files ending with `.pth`) are saved and loaded from the `agent_ckpt` directory.
Playing a pretrained agent for one episode:

`python ls_ddpg_main.py --play -y ./saves/ddpg-agent_BipedalWalker-v2-LS-LAM-10000-100K-BOOSTING-SEED-2019-BATCH-64/best_+316.064_2410000.dat -x ./Videos/`

## Training

Examples:

* `python ls_ddpg_main.py --train --lsddpg -e MinitaurBulletEnv-v0 -l 1 -b 64`
* `python ls_ddpg_main.py --train --lsddpg --boosting -e BipedalWalker-v2 -l 10000 -b 64`

For full description of the flags, see the full API.

## TensorBoard

TensorBoard logs are written dynamically during the runs, and it possible to observe the training progress using the graphs. In order to open TensoBoard, navigate to the source directory of the project and in the terminal/cmd:

`tensorboard --logdir=./runs`

* make sure you have the correct environment activated (`conda activate env-name`) and that you have `tensorboard`, `tensorboardX` installed.

## References
* [PyTorch Agent Net: reinforcement learning toolkit for pytorch](https://github.com/Shmuma/ptan) by [Max Lapan](https://github.com/Shmuma)
* Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://arxiv.org/abs/1705.07461), NIPS 2017


