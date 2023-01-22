## 💡 Reinforcement Learning: Grasp - Pick-and-place with a robotic hand 👨🏻‍💻

### Table of contents

- [👨🏻‍💻 Introduction 👨🏻‍💻](#introduction)
- [🌊 Physics Simulation Engines 🦿](#physics-simulation-engines)
- [🌪 Environment 🦾](#environment)
	* [🔭 Observations 🔍](#observations)
	* [🏄‍♂️ Actions 🤸‍♂️](#actions)
	* [🏆 Reward 🥇](#reward)
- [🚀 Getting started 🖲️](#getting-started)
	* [📝 Prerequisites ⚙️](#prerequisites)
	* [🔨 Installation 🔧](#installation)
	* [🏃‍♂️ Running Demo(s) 🏃‍♂️](#running-demos)
	* [🎉 Results 🎉](#results)
- [🛠 Built With 🛠](#built-with)

#

<a id="introduction" />

### 1. 👨🏻‍💻 Introduction 👨🏻‍💻

The field of robotics has seen incredible advancements in recent years, with the development of increasingly sophisticated machines capable of performing a wide range of tasks. One area of particular interest is the ability for robots to manipulate objects in their environment, known as grasping. In this project, we have chosen to focus on a specific grasping task - training a robotic hand to pick up a moving ball object and place it in a specific target location using the [`Brax` physics simulation engine](https://arxiv.org/pdf/2106.13281.pdf).

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/dev/docs/assets/figure_1.jpeg?raw=true" width="500" height="300">
</p>
<p align="center">Grasp – robotic hand which picks a moving ball and moves it to a specific target</p>

The reason for choosing this project is twofold. Firstly, the ability for robots to grasp and manipulate objects is a fundamental skill that is crucial for many real-world applications, such as manufacturing, logistics, and service industries. Secondly, the use of a physics simulation engine allows us to train our robotic hand in a realistic and controlled environment, without the need for expensive hardware and the associated costs and safety concerns.

Reinforcement learning is a powerful tool for training robots to perform complex tasks, as it allows the robot to learn through trial and error. In this project, we will be using reinforcement learning techniques to train our robotic hand, and we hope to demonstrate the effectiveness of this approach in solving the grasping task.

#

<a id="physics-simulation-engines" />

### 2. 🌊 Physics Simulation Engines 🦿

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

### 3. 🌪 Environment 🦾

The [grasping environment provided by `Brax`](https://github.com/google/brax/blob/198dee3ac4/brax/envs/grasp.py#L25-L1297) is a simple pick-and-place task, where a 4-fingered claw hand must pick up and move a ball to a target location. The environment is designed to simulate the physical interactions between the robotic hand and the ball, including the forces and torques required for the hand to grasp the ball and move it to the target location.

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/dev/docs/assets/figure_2.jpeg?raw=true" width="500" height="300">
</p>
<p align="center">The hand is able to pick up the ball and carry it to a series of red targets. Once the ball gets close to the red target, the red target is respawned at a different random location</p>

In the environment, the robotic hand is represented by a 4-fingered claw, which is capable of opening and closing to grasp the ball. The ball is placed in a random location at the beginning of each episode, and the target location is also randomly chosen. The goal of the robotic hand is to move the ball to the target location as quickly and efficiently as possible. For more details, check [_4.2.2_](https://arxiv.org/pdf/2106.13281.pdf).

#

<a id="observations" />

#### 2.1. 🔭 Observations 🔍

The environment observes _three_ main bodies: the `Hand`, the `Object`, and the `Target`. The agent uses these observations to learn how to control the robotic hand and move the object to the target location.

1. The `Hand` observation includes information about the state of the robotic hand, such as the position and orientation of the fingers, the joint angles, and the forces and torques applied to the hand. This information is used by the agent to control the hand and pick up the object.

2. The `Object` observation includes information about the state of the object, such as its position, velocity, and orientation. This information is used by the agent to track the object and move it to the target location.

3. The `Target` observation includes information about the target location, such as its position and orientation. This information is used by the agent to navigate the hand and the object to the target location.

When the object reaches the target location, the agent is rewarded. The agent is also given a penalty if the object falls or if the hand collides with any obstacle. The agent's goal is to maximize the reward, which means reaching the target location as quickly and efficiently as possible.

Overall, the observations provided by the [`Grasp environment`](https://github.com/google/brax/blob/198dee3ac4/brax/envs/grasp.py#L25-L1297) are designed to give the agent the information it needs to learn how to control the robotic hand and move the object to the target location. The combination of the Hand, Object, and Target observations allows the agent to learn from the environment and improve its performance over time.

#

<a id="actions" />

#### 2.2. 🏄‍♂️ Actions 🤸‍♂️

The action has `19` dimensions, it’s the hand’s position and the joints’ angles, and it is normalized to the `[-1, 1]` as _continuous_ values.

#

<a id="reward" />

#### 2.3. 🏆 Reward 🥇

The reward function goes like the following:



#

<a id="getting-started" />

### 2. 🚀 Getting started 🚀

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#

<a id="prerequisites" />

#### 2.1. ⚙️ Prerequisites ⚙️

- [x] Python 3.7 or higher
- [x] Brax 0.1.0 or higher
- [x] Jax 0.4.1 or higher
- [x] Flax 0.6.3 or higher
- [x] PyTorch 1.13.1 or higher

#

<a id="installation" />

#### 2.2. 🔧 Installation 🔧

1. Clone the repository

```bash
git clone https://github.com/mohammadzainabbas/Reinforcement-Learning-CS.git
cd Reinforcement-Learning-CS/
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

#

<a id="running-demos" />

#### 2.3. 🏃‍♂️ Running Demo(s) 🏃‍♂️

- [x] [Grasp: Pick-and-Place with a robotic hand](https://colab.research.google.com/github/mohammadzainabbas/Reinforcement-Learning-CS/blob/main/notebooks/demo.ipynb)
- [x] [Step-by-step training with PPO](https://colab.research.google.com/github/mohammadzainabbas/Reinforcement-Learning-CS/blob/main/notebooks/demo_ppo_train.ipynb)

Or you can run the `train_ppo.py` file locally by following the steps below:

```bash
python src/train_ppo.py
```

You will get the following output files:

1. `ppo_training.png` - Training progress plot
2. `result_with_ppo.html` - Simulation of the trained agent (in HTML format)
3. `ppo_params` - Trained parameters of the agent

#

<a id="results" />

#### 2.4. 🎉 Results 🎉

The following plot shows the comparsion between training progress of `Proximal policy optimization (PPO)`, `Evolution Strategy (ES)` and `Augmented Random Search (ARS)` algorithms:

<p align="center">
  <img src="https://github.com/mohammadzainabbas/Reinforcement-Learning-CS/blob/main/docs/assets/compare_algorithms.jpeg?raw=true" width="500" height="300">
</p>
