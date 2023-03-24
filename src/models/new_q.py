import gymnasium, numpy as np, random

class NewQAgent:
    def __init__(self, env: gymnasium.Env, alpha: float = 0.5, gamma: float = 0.9, epsilon: float = 1, epsilonDecay: float = 0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay

        self.numActions = self.env.action_space.n
        self.qTable = {}

    def epsilonGreedy(self, state):
        r = np.random.uniform(0, 1)
        if r < self.epsilon:
           # return random.randint(0, self.numActions - 1)
           return self.env.action_space.sample()
        else:
            if state in self.qTable:
                if len(set(self.qTable[state])) == 1:
                    return random.randint(0, self.numActions - 1)
                else:
                    return self.qTable[state].index(max(self.qTable[state]))
            else:
                return 0

    def learn(self, state: gymnasium.Env, action, reward: float, nextState: gymnasium.Env, nextAction, done: bool):
        if state in self.qTable:
            if nextState in self.qTable:
                self.qTable[state][action] += self.alpha * (
                    reward
                    + self.gamma
                    * (
                        self.qTable[nextState][nextAction]
                        - self.qTable[state][action]
                    )
                )
            else:
                self.qTable[nextState] = [0 for _ in range(self.numActions)]

                self.qTable[state][action] += self.alpha * (
                    reward
                    + self.gamma
                    * (
                        self.qTable[nextState][nextAction]
                        - self.qTable[state][action]
                    )
                )
        else:
            self.qTable[state] = [0 for _ in range(self.numActions)]

            if nextState in self.qTable:
                self.qTable[state][action] += self.alpha * (
                    reward + self.gamma * self.qTable[nextState][nextAction]
                )

            else:
                self.qTable[nextState] = [0 for _ in range(self.numActions)]

                self.qTable[state][action] += self.alpha * reward

    def evalGreedy(self, state):
        print(state)

        if state not in self.qTable:
            print("no state")
            return random.randint(0, self.numActions - 1), ""

        elif len(set(self.qTable[state])) == 1:
            print("random choice")
            return random.randint(0, self.numActions - 1), ""

        else:
            print("smart")
            return self.qTable[state].index(max(self.qTable[state])), [
                str(round(d, 2)) for d in self.qTable[state]
            ]
        
    def decay(self) -> None:
        self.epsilon *= self.epsilonDecay
        self.epsilon = max(0.05, self.epsilon) # Ensure that the exploration rate does not go below 5%
