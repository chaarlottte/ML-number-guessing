from src.players import QLearningPlayer, HumanPlayer, NewQPlayer
from src.environments import BaseNumberGuessingEnv, MagnitudeGuessingEnv

if __name__ == "__main__":
    # HumanPlayer().start()
    ai = NewQPlayer(envType=MagnitudeGuessingEnv)
    ai.train(100000)
    
    for _ in range(3):
        ai.play()