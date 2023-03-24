from ..environments import BaseNumberGuessingEnv
import charlogger.logger

class HumanPlayer():
    def __init__(self) -> None:
        self.logger = charlogger.Logger(True)
        self.env = BaseNumberGuessingEnv()
        self.done = False
        self.obs = None
        self.lastGuess = None
        pass

    def start(self) -> None:
        self.setup()
        self.gameLoop()

    def setup(self) -> None:
        self.done = False
        self.lastGuess = None
        self.obs = self.env.reset()

    def gameLoop(self) -> None:
        while self.play():
            continue

        self.start() if self.logger.ask(f"Would you like to play again? (Y/N)").lower() == "y" else None

    def play(self) -> bool:
        action = int(self.logger.ask("What number would you like to guess?"))

        while action == self.lastGuess:
            self.logger.error(f"That was your last guess! Please guess a different number...")
            action = int(self.logger.ask("What number would you like to guess?"))

        self.obs, reward, self.done, info = self.env.step(action)
        if not self.done:
            self.render(reward, action)
        else:
            self.logger.valid(f"Congratulations! You guess the number {self.obs} in {info.get('total_guesses')} guesses.")
            return False

        self.lastGuess = action
        return True
    
    def render(self, reward: float, action: int) -> None:
        if reward == 0:
            self.logger.info(f"Initial guess: {action}")
        elif reward == -1:
            self.logger.plus(f"Your guess is the same temperature as {self.lastGuess}!")
        elif reward < 0:
            self.logger.warn(f"Your guess is colder than {self.lastGuess}.")
        elif reward > 0:
            self.logger.plus(f"Your guess is warmer than {self.lastGuess}!")
    