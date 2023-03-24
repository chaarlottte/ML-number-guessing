from .base_env import BaseNumberGuessingEnv
import math

class MagnitudeGuessingEnv(BaseNumberGuessingEnv):
    
    def getReward(self, guess: int, prevGuess: int) -> float:
        if prevGuess is None:
            return 0.0

        prevOrder = int(math.log10(abs(prevGuess - self.target)))
        currOrder = int(math.log10(abs(guess - self.target)))

        if prevOrder == currOrder:
            return -1.0
        elif currOrder < prevOrder:
            return 0.1 * (prevOrder - currOrder)
        else:
            return -0.1 * (currOrder - prevOrder)


