import gymnasium
import numpy as np

class QLearningAgent:
    def __init__(self, env: gymnasium.Env, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, epsilonDecay: float = 0.99):
        self.env = env # Environment that the agent is working in
        self.alpha = alpha # The agent's learning rate
        self.gamma = gamma # Discount factor. To be COMPLETELY honest, I still have no clue what this does.
        self.epsilon = epsilon # The initial exploration rate. Basically controlling how often a random action is taken.
        self.epsilonDecay = epsilonDecay # The decay rate for the exploration rate.
        self.qTable = np.zeros((env.observation_space.n, env.action_space.n)) # The Q-Table the agent will use to learn

    def chooseAction(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Take a random action.
            # print("USING EPSILON")
            return self.env.action_space.sample()
        else:
            # Choose the action with the highest likely Q-value for the current state
            return np.argmax(self.qTable[state, :])

    def learn(self, state: gymnasium.Env, action, reward: float, nextState: gymnasium.Env, done: bool):
        # Update the Q-value for the current state-action pair
        currentQValue = self.qTable[state, action]
        nextQValue = np.max(self.qTable[nextState, :])
        tdTarget = reward + self.gamma * nextQValue * (1 - done)
        tdError = tdTarget - currentQValue
        self.qTable[state, action] += self.alpha * tdError

    def decay(self) -> None:
        self.epsilon *= self.epsilonDecay
        self.epsilon = max(0.01, self.epsilon) # Ensure that the exploration rate does not go below 1%

    def train(self, totalEpisodes: int):
        for _ in range(totalEpisodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.chooseAction(state)
                nextState, reward, done, info = self.env.step(action)
                self.learn(state, action, reward, nextState, done)
                state = nextState
