from .models import QLearningAgent
import wandb, os

class WeightsBiasesHandler():
    def __init__(self, agent: QLearningAgent) -> None:
        os.environ["WANDB_CONSOLE"] = "wrap"
        self.agent = agent
        pass

    def setup(self, episodes: int) -> None:
        wandb.init(
            project="number_guessing", 
            entity="quickdaffy",
            config={
                "learning_rate": self.agent.alpha,
                "discount_factor": self.agent.gamma,
                "episodes": episodes,
                "initial_epsilon": self.agent.epsilon,
                "epsilon_decay": self.agent.epsilonDecay
            }
        )
        pass

    def log(self, totalTries: int, avgReward: float, avgGuess: float, warmerGuesses: int, colderGuesses: int) -> None:
        wandb.log({
          "total_tries": totalTries,
          "average_reward": avgReward,
          "average_guess": avgGuess,
          "warmer_guesses": warmerGuesses,
          "colder_guesses": colderGuesses,
          "epsilon": self.agent.epsilon,
          # "q_table_size": len(self.agent.qTable)
        })