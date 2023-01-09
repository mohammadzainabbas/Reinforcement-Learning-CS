import tensorflow as tf

class SACAgent:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.target_actor = tf.keras.models.clone_model(actor)
        self.target_actor.set_weights(actor.get_weights())
        self.target_critic = tf.keras.models.clone_model(critic)
        self.target_critic.set_weights(critic.get_weights())
        self.replay_buffer = []
        self.discount_factor = 0.99
        self.target_entropy = None
        self.learning_rate = None

    def select_action(self, time_step):
        observation = time_step.observation
        logits = self.actor(observation[None, :])
        action = tf.random.categorical(logits, num_samples=1)[0, 0]
        action = tf.cast(action, tf.float32)
        return action[None]

    def train_step(self, observations, actions, rewards, next_observations, dones, optimizer):
        # Store the data in the replay buffer.
        for i in range(observations.shape[0]):
            self.replay_buffer.append((observations[i], actions[i], rewards[i], next_observations[i], dones[i]))

        # Sample a batch of data from the replay buffer.
        batch_size = 128
        indexes = np.random.randint(len(self.replay_buffer), size=batch_size)
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        for i in indexes:
            o, a, r, no, d = self.replay_buffer[i]
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            next_observations.append(no)
            dones.append(d)
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        # Compute the Q-values and target Q-values.
        q_values = self.critic(observations, actions)
        with tf.GradientTape() as tape:
            logits = self.actor(next_observations)
            target_q_values = self.target_critic(next_observations, self.target_actor(next_observations))
            target_q_values -= self.target_entropy * tf.exp(logits)
            target_q_values = rewards + (1 - dones) * self.discount_factor * target_q_values
            critic_loss = tf.reduce_mean((q_values - target_q_values) ** 2)
            