from ..environments import BaseNumberGuessingEnv
from ..models import NewQAgent
from ..wandb_handler import WeightsBiasesHandler
import gymnasium, charlogger.logger, time

class NewQPlayer():
    def __init__(self, envType = BaseNumberGuessingEnv) -> None:
        self.logger = charlogger.Logger(True)
        self.trainingLogger = charlogger.Logger(True, defaultPrefix="<TIME> Q-Learning")

        self.env: BaseNumberGuessingEnv = envType()
        self.done = False
        self.obs = None
        self.lastGuess = None

        self.agent = NewQAgent(
            env=self.env,
            alpha=0.5,
            gamma=0.9,
            epsilon=1.0, 
            epsilonDecay=0.999
        )

        self.wandb = WeightsBiasesHandler(self.agent)
        pass

    def train(self, totalEpisodes: int = 1000) -> gymnasium.Env:
        self.wandb.setup(totalEpisodes)
        self.trainingLogger.info("Starting in five seconds.")
        time.sleep(5)
        self.trainingLogger.info("Starting...")
        for episode in range(totalEpisodes):
            # self.trainingLogger.debug(f"Episode: {episode}/{totalEpisodes} ({100 * float(episode) / float(totalEpisodes)}% complete)")
            self.setup()

            totalGuesses = 0

            warmerGuesses = 0
            colderGuesses = 0
            
            allGuesses = 0
            allRewards = 0

            avgGuess = 0
            avgReward = 0

            while not self.done:
                self.lastGuess = action = self.agent.epsilonGreedy(self.obs)
                nextState, reward, self.done, info = self.env.step(action)
                nextAction = self.agent.epsilonGreedy(nextState)
                self.agent.learn(state=self.obs, 
                                action=action, 
                                reward=reward, 
                                nextState=nextState, 
                                nextAction=nextAction, 
                                done=self.done)
                self.obs = nextState

                if not self.done:
                    if reward > 0:
                        warmerGuesses += 1
                    elif reward < 0:
                        colderGuesses += 1

                totalGuesses += 1
                allGuesses += action
                allRewards += reward
                avgGuess = allGuesses / totalGuesses
                avgReward = allRewards / totalGuesses

            self.agent.decay()
            
            self.wandb.log(
                totalTries=totalGuesses,
                avgReward=avgReward,
                avgGuess=avgGuess,
                warmerGuesses=warmerGuesses,
                colderGuesses=colderGuesses
            )
            
            # Print log every x episodes, where x is 10% of the total episodes
            if episode % (totalEpisodes * 0.1) == 0:
                self.trainingLogger.debug(f"{100 * float(episode) / float(totalEpisodes)}% complete | Epsilon: {self.agent.epsilon} | guesses: {totalGuesses} | H/C: {warmerGuesses}/{colderGuesses} | avg guess: {avgGuess} | avg reward: {avgReward}", title=f"{episode}/{totalEpisodes}")
                # self.trainingLogger.debug(f"guesses: {totalGuesses} | H/C: {warmerGuesses}/{colderGuesses} | avg guess: {avgGuess} | avg reward: {avgReward}", title=f"{episode}/{totalEpisodes}")


    def play(self) -> None:
        self.obs = self.env.reset()
        self.done = False
        print(self.obs)

        while not self.done:
            # action = np.argmax(self.agent.qTable[self.obs, :])
            action = self.agent.epsilonGreedy(self.obs)
            self.logger.debug(title="AI", data=f"Guess: {action}")
            nextState, reward, self.done, info = self.env.step(action)
            self.agent.learn(self.obs, action, reward, nextState, nextAction=1, done=self.done)
            self.obs = nextState
            if not self.done:
                self.render(reward, action)
            else:
                self.logger.valid(f"Congratulations! You guess the number {self.obs} in {info.get('total_guesses')} guesses.")
                return False
            
            self.lastGuess = action
            self.agent.decay()

    def setup(self) -> None:
        self.done = False
        self.lastGuess = None
        self.obs = self.env.reset()
        self.agent.epsilon 
        pass

    def render(self, reward: float, action: int) -> None:
        if reward == 0:
            self.logger.info(f"Initial guess: {action}")
        elif reward == -1:
            self.logger.plus(f"Your guess is the same temperature as {self.lastGuess}!")
        elif reward < 0:
            self.logger.warn(f"Your guess is colder than {self.lastGuess}.")
        elif reward > 0:
            self.logger.plus(f"Your guess is warmer than {self.lastGuess}!")

