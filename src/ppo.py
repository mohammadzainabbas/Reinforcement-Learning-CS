import torch
import numpy as np

from typing import List, Tuple

class PPO:
    """
    The PPO agent.
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 lr: float,
                 clip_ratio: float,
                 value_loss_coef: float,
                 entropy_coef: float,
                 max_grad_norm: float) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.policy = Policy(state_dim, action_dim, hidden_dim)
        self.value_function = ValueFunction(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value_function.parameters()), lr=lr)

        self.old_log_probs = None

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using the current policy.
        """
        with torch.no_grad():
            action = self.policy(state)
        return action

    def update(self, rollouts: RolloutStorage) -> float:
        """
        Updates the PPO agent using the provided rollout storage.
        """
        states, actions, log_probs, returns, advantages = rollouts.get()

        for _ in range(self.ppo_epoch):
            for state, action, old_log_prob, return_, advantage in ppo_iter(self.mini_batch_size, states, actions, log_probs, returns, advantages):
                loss = self.train_step(state, action, return_, advantage)

        return loss

    def train_step(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> float:
        """
        Performs a single training step on the PPO agent.

        This function takes as input the current states, actions, returns, and advantages, and uses these to compute the policy and value loss terms. 
        The losses are then combined and used to update the policy and value function parameters using the Adam optimizer. 
        The function also computes and returns the entropy of the policy.
        """
        log_probs = self.policy.get_log_prob(states, actions)
        values = self.value_function(states)
        entropy = self.policy.get_entropy()

        ratio = (log_probs - self.old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (returns - values).pow(2).mean()

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.old_log_probs = log_probs.detach()

        return loss.item()
    
    def save(self, path: str) -> None:
        """
        Saves the current policy and value function parameters to the specified path.
        """
        torch.save(self.policy.state_dict(), path + "_policy.pt")
        torch.save(self.value_function.state_dict(), path + "_value_function.pt")

    def load(self, path: str) -> None:
        """
        Loads the policy and value function parameters from the specified path.
        """
        self.policy.load_state_dict(torch.load(path + "_policy.pt"))
        self.value_function.load_state_dict(torch.load(path + "_value_function.pt"))
    
    def get_value(self, state: torch.Tensor) -> float:
        """
        Returns the value of the provided state.
        """
        return self.value_function(state).item()
    
    def get_entropy(self, state: torch.Tensor) -> float:
        """
        Returns the entropy of the current policy.
        """
        return self.policy.get_entropy(state).item()
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Returns the log probability of the provided action under the current policy.
        """
        return self.policy.get_log_prob(state, action).item()
    
    def get_kl(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Returns the KL divergence between the current policy and the old policy.
        """
        return (self.policy.get_log_prob(state, action) - self.old_log_probs).item()
    
    def get_policy_loss(self, state: torch.Tensor, action: torch.Tensor, return_: torch.Tensor, advantage: torch.Tensor) -> float:
        """
        Returns the policy loss.
        """
        log_probs = self.policy.get_log_prob(state, action)
        ratio = (log_probs - self.old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss.item()
    
    def get_value_loss(self, state: torch.Tensor, return_: torch.Tensor) -> float:
        """
        Returns the value function loss.
        """
        values = self.value_function(state)
        value_loss = 0.5 * (return_ - values).pow(2).mean()
        return value_loss.item()
    
    def get_entropy_loss(self, state: torch.Tensor) -> float:
        """
        Returns the entropy loss.
        """
        entropy = self.policy.get_entropy(state)
        entropy_loss = -self.entropy_coef * entropy
        return entropy_loss.item()
    
    def get_loss(self, state: torch.Tensor, action: torch.Tensor, return_: torch.Tensor, advantage: torch.Tensor) -> float:
        """
        Returns the total loss.
        """
        log_probs = self.policy.get_log_prob(state, action)
        values = self.value_function(state)
        entropy = self.policy.get_entropy(state)

        ratio = (log_probs - self.old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (return_ - values).pow(2).mean()

        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        return loss.item()
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns an action sampled from the current policy.
        """
        return self.policy(state).detach()
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an action sampled from the current policy and the log probability of the action.
        """
        action = self.policy(state).detach()
        log_prob = self.policy.get_log_prob(state, action)
        return action, log_prob
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an action sampled from the current policy and the value of the state.
        """
        action = self.policy(state).detach()
        value = self.value_function(state).detach()
        return action, value
    
    def get_action_and_value_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns an action sampled from the current policy, the value of the state, and the log probability of the action.
        """
        action = self.policy(state).detach()
        value = self.value_function(state).detach()
        log_prob = self.policy.get_log_prob(state, action)
        return action, value, log_prob
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> None:
        """
        Updates the policy and value function parameters.
        """
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss = self.get_loss(states, actions, returns, advantages)
        loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        self.old_log_probs = self.policy.get_log_prob(states, actions).detach()

class ValueFunction(torch.nn.Module):
    """
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Name: {self.name}, Age: {self.age}"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    def __hash__(self):
        return hash((self.name, self.age))
    The value function network maps states to values.
    """
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueFunction, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Policy(torch.nn.Module):
    """
    The policy network maps states to actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        actions_pred = self.forward(states)
        log_probs = -(actions - actions_pred).pow(2) / 2
        return log_probs

    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        return (1 - self.forward(state).pow(2)).mean()

class ValueFunction(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueFunction, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_step(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor) -> float:
    """
    Performs a single training step on the PPO agent.

    This function takes as input the current states, actions, returns, and advantages, and uses these to compute the policy and value loss terms. 
    The losses are then combined and used to update the policy and value function parameters using the Adam optimizer. 
    The function also computes and returns the entropy of the policy.
    """
    log_probs = self.policy.get_log_prob(states, actions)
    values = self.value_function(states)
    entropy = self.policy.get_entropy()

    ratio = (log_probs - self.old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()

    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.old_log_probs = log_probs.detach()

    return loss.item()

def compute_gae(rewards: List[float],
                values: List[float],
                next_value: float,
                gamma: float,
                tau: float) -> List[float]:
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * tau * gae
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size: int,
                states: List[torch.Tensor],
                actions: List[torch.Tensor],
                log_probs: List[torch.Tensor],
                returns: List[torch.Tensor],
                advantage: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(states)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(ppo_epochs: int,
                mini_batch_size: int,
                states: List[torch.Tensor],
                actions: List[torch.Tensor],
                log_probs: List[torch.Tensor],
                returns: List[torch.Tensor],
                advantages: List[torch.Tensor],
                clip_param: float,
                ppo: PPO) -> None:
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = ppo.policy(state), ppo.value_function(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = (return_ - value).pow(2).mean()

                loss = action_loss + ppo.value_loss_coef * value_loss - ppo.entropy_coef * entropy

                ppo.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo.policy.parameters(), ppo.max_grad_norm)
                ppo.optimizer.step()

# Path: src/ppo.py
import gym
import torch
import numpy as np

from typing import List, Tuple

from ppo import PPO, Policy, ValueFunction, compute_gae, ppo_update

def main() -> None:
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    lr = 3e-4
    clip_ratio = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    ppo_epochs = 10
    mini_batch_size = 5
    gamma = 0.99
    tau = 0.95
    num_steps = 128
    total_num_steps = 1000000
    update_timestep = 2000
    action_std = 0.6
    action_std_decay_rate = 0.999
    log_interval = 1
    save_interval = 10

    ppo = PPO(state_dim, action_dim, hidden_dim, lr, clip_ratio, value_loss_coef, entropy_coef, max_grad_norm)

    state = env.reset()
    for timestep in range(1, total_num_steps + 1):
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
        for _ in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0)
            dist, _ = ppo.policy(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.detach().numpy()[0])

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(torch.FloatTensor([reward]).unsqueeze(1))
            next_states.append(torch.FloatTensor(next_state).unsqueeze(0))
            dones.append(torch.FloatTensor([done]).unsqueeze(1))

            state = next_state

            if done:
                state = env.reset()

        _, next_value = ppo.policy(next_states[-1])
        returns = compute_gae(rewards, next_value, dones, gamma, tau)

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        advantages = returns - ppo.value_function(states).detach()

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_ratio, ppo)

        if timestep % update_timestep == 0
            ppo.update_target()

        if timestep % log_interval == 0:
            print(f"Timestep {timestep}/{total_num_steps}")

        if timestep % save_interval == 0:
            ppo.save(f"ppo_{timestep}.pt")

if __name__ == '__main__':
    main()
