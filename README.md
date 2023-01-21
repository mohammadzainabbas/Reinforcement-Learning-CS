## 💡 Reinforcement Learning: Grasp - Pick-and-place with a robotic hand 👨🏻‍💻

### Table of contents

- [👨🏻‍💻 Introduction 👨🏻‍💻](#introduction)
- [🚀 Getting started 🚀](#getting-started)
	* [⚙️ Prerequisites ⚙️](#prerequisites)
	* [🔧 Installation 🔧](#installation)
	* [🏃‍♂️ Running Demo(s) 🏃‍♂️](#running-demos)
	* [🎉 Results 🎉](#results)
- [🛠 Built With 🛠](#built-with)

#

<a id="introduction" />

### 1. 👨🏻‍💻 Introduction 👨🏻‍💻

This repository contains the code and resources for a reinforcement learning project where we trained a robotic hand to grasp a moving ball and move it to a certain target location using PPO algorithm and using Brax physics simulation engine.

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

You'd get the following two output files:

1. `ppo_training.png` - Training progress plot
2. `ppo_training.mp4` - Training progress video
