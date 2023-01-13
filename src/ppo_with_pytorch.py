#@title Import Brax and some helper modules
from IPython.display import clear_output

import collections
from datetime import datetime
import functools
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence

try:
	import brax
except ImportError:
	!pip install git+https://github.com/google/brax.git@main
	clear_output()
	import brax

from brax import envs
from brax.envs import to_torch
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
v = torch.ones(1, device=DEVICE)
