import numpy as np
from task import Task

class Naive():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.last_state = task.last_state

        self.mean = 0;
        self.reward = 0;
        self.count = 0;

        # Score tracker and learning parameters
        self.best_score = -np.inf
        self.noise_scale = 0.1

    def reset_episode(self):
        self.reward = 0.0
        self.count = 0
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.reward = reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.mean)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.reward / float(self.count) if self.count else 0.0

        self.mean = (self.mean + self.score)/2
        self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        self.mean = self.mean + self.noise_scale * np.random.normal(size=1)  # equal noise in all directions