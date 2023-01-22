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

The field of robotics has seen incredible advancements in recent years, with the development of increasingly sophisticated machines capable of performing a wide range of tasks. One area of particular interest is the ability for robots to manipulate objects in their environment, known as grasping. In this project, we have chosen to focus on a specific grasping task - training a robotic hand to pick up a moving ball object and place it in a specific target location using the [Brax physics simulation engine](https://arxiv.org/pdf/2106.13281.pdf).

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

Ultimately, we chose to use [Brax](https://github.com/google/brax/) due to [its highly scalable and parallelizable architecture](https://ai.googleblog.com/2021/07/speeding-up-reinforcement-learning-with.html), which makes it well-suited for accelerated hardware (XLA backends such as GPUs and TPUs). This allows us to simulate the grasping task at a high level of realism and detail, while also taking advantage of the increased computational power of modern hardware to speed up the training process.


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
