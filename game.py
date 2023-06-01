from random import random


class Game:

    def __init__(self, solution=None, max_rounds=6):
        with open('valid_solutions.csv') as f:
            self.valid_solutions = f.read().splitlines()[1:-1]
        with open('valid_guesses.csv') as f:
            self.valid_guesses_list = f.read().splitlines()[1:-1] + self.valid_solutions
        if solution is None:
            # choose random solution
            self.solution = self.valid_solutions[int(random() * len(self.valid_solutions))]
        else:
            self.solution = solution
        self.round_num = 0
        self.max_rounds = max_rounds

    @staticmethod
    def start_game():
        return "I thought of a 5-letter word. Try to guess it!"

    def evaluate_guess(self, guess):
        if guess not in self.valid_guesses_list:
            return False, f"Guess {guess} is not a valid 5-letter word"
        self.round_num += 1
        if guess == self.solution:
            return True, f"You won in {self.round_num} rounds!"
        if self.round_num == self.max_rounds:
            return True, f"You lost after {self.round_num} rounds. The solution was {self.solution}"
        return False, self.get_eval(self.solution, guess)

    @staticmethod
    def get_eval(sol, guess):
        """ a function that takes in a solution and a guess and returns the evaluation """
        res = ['_'] * 5
        non_green_counts = {}
        # first fill in greens
        for i in range(0, 5):
            if guess[i] == sol[i]:
                res[i] = 'G'
            else:
                non_green_counts[sol[i]] = 1 if sol[i] not in non_green_counts else non_green_counts[sol[i]] + 1
        # now fill in yellows, needed to do separately to take into account duplicates
        for i in range(0, 5):
            if res[i] != 1:
                if guess[i] in non_green_counts and non_green_counts[guess[i]] > 0:
                    res[i] = 'Y'
                    non_green_counts[guess[i]] -= 1
        return "".join(res)
