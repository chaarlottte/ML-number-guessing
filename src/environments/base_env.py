from gymnasium import spaces
import random, gymnasium, math

class BaseNumberGuessingEnv(gymnasium.Env):
    def __init__(self, minNum: int = 0, maxNum: int = 100):
        self.min = minNum
        self.max = maxNum

        self.observation_space = spaces.Discrete(int(self.diff(self.min, self.max) + 1))
        self.action_space = spaces.Discrete(int(self.diff(self.min, self.max) + 1))
        self.reward_range = (-1, 1)

        self.target = None
        self.guess = None
        self.done = False
        pass
        
    def reset(self):
        self.target = random.randint(self.min, self.max)
        self.guess = None
        self.done = False
        self.guesses = 0
        # print(self.target)
        return self.target
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        prevGuess = self.guess
        self.guess = action

        if prevGuess is None and self.guess != self.target:
            reward = 0.0
        else:
            self.guesses += 1
            if self.guess == self.target:
                reward = 1.0
                self.done = True
            else:
                reward = self.getReward(self.guess, prevGuess)
                
        return self.guess, reward, self.done, { "total_guesses": self.guesses }
    
    def getReward(self, guess: int, prevGuess: int) -> float:
        prevDiff = self.diff(prevGuess, self.target)
        currentDiff = self.diff(guess, self.target)

        if prevDiff == currentDiff:
            reward = -1
        elif currentDiff < prevDiff:
            reward = 0.1 * (prevDiff - currentDiff)  # Reward a higher number for a closer guess
        else:
            reward = -0.1 * (currentDiff - prevDiff) # Reward a lower number for a further guess
        
        return reward

    # No parameter type declaration so that multiple types can be used (float, int)
    def diff(self, num1, num2) -> float:
        return math.dist((num1,), (num2,))