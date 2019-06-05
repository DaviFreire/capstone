import numpy as np

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, state, date):
        """Initialize a Task object."""
        
        self.adj_close = 0
        self.state_size = 6
        self.action_size = 6
        self.last_date = date

        self.last_state = state

    def get_reward(self, state):

        diff = state.sub(self.last_state.squeeze())
        value = diff['Adj Close']
        if (abs(value.iloc[0]) != 0 ):
            reward = 100/(.01*(abs(value.iloc[0])))
        else:
            reward = 1000
        return reward

    def step(self, action, state):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        done = 0
        self.last_state += action
        reward += self.get_reward(state) 
        next_state = self.last_state

        if state.index.tolist()[0].strftime('%Y-%m-%d') == self.last_date:
            done = 1;
        return next_state, reward, done