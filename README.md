## 💡 Grasp - Pick-and-place with a robotic hand 👨🏻‍💻

You can see the live demo [here](http://mohammadzainabbas.tech/Reinforcement-Learning-CS/).

### Table of contents

- [🚀 Quickstart 💻](#quickstart)
- [💻 Introduction 👨🏻‍💻](#introduction)
- [🌊 Physics Simulation Engines 🦿](#physics-simulation-engines)
- [🌪 Environment 🦾](#environment)
	* [🔭 `Observations` 🔍](#observations)
	* [🏄‍♂️ `Actions` 🤸‍♂️](#actions)
	* [🏆 `Reward` 🥇](#reward)
- [🔬 Algorithms 💻](#algorithms)
	* [💡 `Proximal policy optimization (PPO)` 👨🏻‍💻](#ppo)
	* [💡 `Evolution Strategy (ES)` 👨🏻‍💻](#es)
	* [💡 `Augmented Random Search (ARS)` 👨🏻‍💻](#ars)
	* [💡 `Soft Actor-Critic (SAC)` 👨🏻‍💻](#sac)
- [🚀 Run locally 🖲️](#run-locally)

#

<a id="quickstart" />

### 1. 🚀 Quickstart 💻

Explore the project easily and quickly through the following _colab_ notebooks:

- [`Grasp: Pick-and-place with a robotic hand`](https://colab.research.google.com/github/mohammadzainabbas/Reinforcement-Learning-CS/blob/main/notebooks/demo.ipynb) - this demo notebook compares first three [algorithms](#algorithms) and train agents on `Grasp` environment by `Brax`. At the end, it also shows trained `PPO agent` interaction with the environment.

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/dev/docs/assets/compare_algorithms.jpeg?raw=true" width="auto" height="225">
</p>

- [`Step-by-step training with PPO`](https://colab.research.google.com/github/mohammadzainabbas/Reinforcement-Learning-CS/blob/main/notebooks/demo_ppo_train.ipynb) - this notebook shows step-by-step training of `PPO agent` on `Grasp` environment by `Brax`.

#

<a id="introduction" />

### 2. 💻 Introduction 👨🏻‍💻

The field of robotics has seen incredible advancements in recent years, with the development of increasingly sophisticated machines capable of performing a wide range of tasks. One area of particular interest is the ability for robots to manipulate objects in their environment, known as grasping. In this project, we have chosen to focus on a specific grasping task - training a robotic hand to pick up a moving ball object and place it in a specific target location using the [`Brax` physics simulation engine](https://arxiv.org/pdf/2106.13281.pdf).

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/dev/docs/assets/figure_1.jpeg?raw=true" width="500" height="300">
</p>
<p align="center">Grasp – robotic hand which picks a moving ball and moves it to a specific target</p>

The reason for choosing this project is twofold. Firstly, the ability for robots to grasp and manipulate objects is a fundamental skill that is crucial for many real-world applications, such as manufacturing, logistics, and service industries. Secondly, the use of a physics simulation engine allows us to train our robotic hand in a realistic and controlled environment, without the need for expensive hardware and the associated costs and safety concerns.

Reinforcement learning is a powerful tool for training robots to perform complex tasks, as it allows the robot to learn through trial and error. In this project, we will be using reinforcement learning techniques to train our robotic hand, and we hope to demonstrate the effectiveness of this approach in solving the grasping task.

#

<a id="physics-simulation-engines" />

### 3. 🌊 Physics Simulation Engines 🦿

The use of a physics simulation engine is essential for training a robotic hand to perform the grasping task, as it allows us to simulate the real-world physical interactions between the robot and the ball. Without a physics simulation engine, it would be difficult to accurately model the dynamics of the task, including the forces and torques required for the robotic hand to pick up the ball and move it to the target location.

In this project, we explored several different physics simulation engines, including:

- [x] [`MuJoCo`](https://mujoco.org/) ([`dm_control`](https://github.com/deepmind/dm_control/), [`Gym`](https://www.gymlibrary.dev/) and [`Gymnasium`](https://gymnasium.farama.org/))
- [x] [`TinyDiffSim`](https://github.com/erwincoumans/tiny-differentiable-simulator)
- [x] [`DiffTaichi`](https://github.com/taichi-dev/difftaichi)
- [x] [`Nimble`](https://github.com/keenon/nimblephysics)
- [x] [`PyBullet`](https://github.com/bulletphysics/bullet3)
- [x] [`Brax`](https://github.com/google/brax/). 

Each of these engines has its own strengths and weaknesses, and we carefully considered the trade-offs between them before making a final decision.

Ultimately, we chose to use [`Brax`](https://github.com/google/brax/) due to [_its highly scalable and parallelizable architecture_](https://ai.googleblog.com/2021/07/speeding-up-reinforcement-learning-with.html), which makes it well-suited for accelerated hardware (XLA backends such as `GPUs` and `TPUs`). This allows us to simulate the grasping task at a high level of realism and detail, while also taking advantage of the increased computational power of modern hardware to speed up the training process.

#

<a id="environment" />

### 4. 🌪 Environment 🦾

The [grasping environment provided by `Brax`](https://github.com/google/brax/blob/198dee3ac4/brax/envs/grasp.py#L25-L1297) is a simple pick-and-place task, where a 4-fingered claw hand must pick up and move a ball to a target location. The environment is designed to simulate the physical interactions between the robotic hand and the ball, including the forces and torques required for the hand to grasp the ball and move it to the target location.

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/dev/docs/assets/figure_2.jpeg?raw=true" width="500" height="300">
</p>
<p align="center">The hand is able to pick up the ball and carry it to a series of red targets. Once the ball gets close to the red target, the red target is respawned at a different random location</p>

In the environment, the robotic hand is represented by a 4-fingered claw, which is capable of opening and closing to grasp the ball. The ball is placed in a random location at the beginning of each episode, and the target location is also randomly chosen. The goal of the robotic hand is to move the ball to the target location as quickly and efficiently as possible. For more details, check [_4.2.2_](https://arxiv.org/pdf/2106.13281.pdf).

#

<a id="observations" />

#### 4.1. 🔭 Observations 🔍

The environment observes _three_ main bodies: the `Hand`, the `Object`, and the `Target`. The agent uses these observations to learn how to control the robotic hand and move the object to the target location.

1. The `Hand` observation includes information about the state of the robotic hand, such as the position and orientation of the fingers, the joint angles, and the forces and torques applied to the hand. This information is used by the agent to control the hand and pick up the object.

2. The `Object` observation includes information about the state of the object, such as its position, velocity, and orientation. This information is used by the agent to track the object and move it to the target location.

3. The `Target` observation includes information about the target location, such as its position and orientation. This information is used by the agent to navigate the hand and the object to the target location.

When the object reaches the target location, the agent is rewarded. The agent is also given a penalty if the object falls or if the hand collides with any obstacle. The agent's goal is to maximize the reward, which means reaching the target location as quickly and efficiently as possible.

Overall, the observations provided by the [`Grasp environment`](https://github.com/google/brax/blob/198dee3ac4/brax/envs/grasp.py#L25-L1297) are designed to give the agent the information it needs to learn how to control the robotic hand and move the object to the target location. The combination of the Hand, Object, and Target observations allows the agent to learn from the environment and improve its performance over time.

#

<a id="actions" />

#### 4.2. 🏄‍♂️ Actions 🤸‍♂️

The action has `19` dimensions, it’s the hand’s position and the joints’ angles, and it is normalized to the `[-1, 1]` as _continuous_ values.

#

<a id="reward" />

#### 4.3. 🏆 Reward 🥇

The [reward function](https://github.com/google/brax/blob/198dee3ac4/brax/envs/grasp.py#L90-L121) is calculated using following equation:

```math
\text{reward} = \text{moving to object} + \text{close to object} + \text{touching object} + 5 * \text{target hit} + \text{moving to target}
```

where,

```math
\text{moving to object} : \text{small reward for moving towards the object.} \nonumber \\
```

```math
\text{close to object} : \text{small reward for being close to the object.} \nonumber \\
```

```math
\text{touching object} : \text{small reward for touching the object.} \nonumber \\
```

```math
\text{target hit} : \text{high reward for hitting the target (max. reward).} \nonumber \\
```

```math
\text{moving to target} : \text{high reward for moving towards the target.} \nonumber
```

where each minor step approaching the task completeness will be rewarded, while the $\text{target hit}$ will gain the biggest reward.

#

<a id="algorithms" />

### 5. 🔬 Algorithms 💻

We will use the brax’s optimized algorithms: `PPO`, `ES`, `ARS` and `SAC`.

<a id="ppo" />

#### 5.1. 💡 Proximal policy optimization (PPO) 👨🏻‍💻

[`Proximal Policy Optimization (PPO)`](https://arxiv.org/abs/1707.06347) is a model-free online policy gradient reinforcement learning algorithm, developed at OpenAI in 2017. `PPO` strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small. Generally speaking, it is a clipper version [`A2C`](https://huggingface.co/blog/deep-rl-a2c) algorithm.

<a id="es" />

#### 5.2. 💡 Evolution Strategy (ES) 👨🏻‍💻

[`Evolution Strategy (ES)`](https://arxiv.org/abs/1707.06347) is inspired by natural evolution, it is a powerful black-box optimization technique. A group of random noise is tested for the network parameters, and the highest scoring parameter vectors are chosen to evolute the network. It is a different method compared with using the loss function to back propagate the network. `ES` can be parallelized using XLA backend (`CPU`/`GPU`/`TPU`) to speed up the training.

<a id="ars" />

#### 5.3. 💡 Augmented Random Search (ARS) 👨🏻‍💻

[`Augmented Random Search (ARS)`](https://arxiv.org/abs/1803.07055) is a random search method for training linear policies for continuous control problems. It operates directly on the policy weights, each epoch the agent perturbs its current policy `N` times, and collects `2N` rollouts using the modified policies. The rewards from these rollouts are used to update the current policy weights, repeat until completion. The algorithm is known to have high variance; not all seeds obtain high rewards, but to our knowledge their work in many ways represents the state of the art on these benchmarks.

<a id="sac" />

#### 5.4. 💡 Soft Actor-Critic (SAC) 👨🏻‍💻

[`Soft Actor-Critic (SAC)`](https://arxiv.org/abs/1801.01290) is an off-policy model-free reinforcement framework. The actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible, and that is why it’s called _soft_. `SAC` has better sample efficiency than `PPO`.

#

<a id="run-locally" />

### 6. 🚀 Run locally 🖲️

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Clone the repository

```bash
git clone https://github.com/mohammadzainabbas/Reinforcement-Learning-CS.git
cd Reinforcement-Learning-CS/
```

2. Create a new enviornment and install all dependencies

First, [install `mamba`](https://mamba.readthedocs.io/en/latest/installation.html), a fast and efficient package manager for `conda`.

```bash
conda install mamba -n base -c conda-forge
```

Then, create a new environment and install all dependencies, and activate it.

```bash
mamba env create -n reinforcement_learning -f docs/config/reinforcement_learning_env.yaml
conda activate reinforcement_learning
```

Or you can run the `train_ppo.py` file locally by following the steps below:

```bash
python src/train_ppo.py
```

You will get the following output files:

1. `ppo_training.png` - Training progress plot
2. `result_with_ppo.html` - Simulation of the trained agent (in HTML format)
3. `ppo_params` - Trained parameters of the agent

#
