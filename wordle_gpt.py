from langchain.callbacks import get_openai_callback

from agents.cot_with_helper import CotWithHelperAgent
from agents.simple_agents import ZeroShotSimpleAgent, FewShotSimpleAgent, SmartSimpleAgent, CoTAgent, \
    CoTLessContextAgent
from game import Game

# gpt-3.5-turbo

words_to_test = ['train', 'slick', 'bring', 'flank']

# agent = ZeroShotSimpleAgent(model='gpt-4')
# agent = FewShotSimpleAgent(model='gpt-4')
# agent = SmartSimpleAgent(model='gpt-4')
# agent = CoTLessContextAgent(model='gpt-4')
# agent = CotWithHelperAgent(model='gpt-4')
successes = {}
failures = {}
with get_openai_callback() as cb:
    for sol in words_to_test:
        game = Game(max_rounds=6, solution=sol)
        # agent = ZeroShotSimpleAgent(model='gpt-4')
        # agent = FewShotSimpleAgent(model='gpt-4')
        # agent = SmartSimpleAgent(model='gpt-4')
        # agent = CoTAgent(model='gpt-4')
        # agent = CoTLessContextAgent(model='gpt-4')
        agent = CotWithHelperAgent(model='gpt-4')
        msg = game.start_game()
        print(msg)
        end = False
        i = 1
        max_tries = 8

        i += 1
        try:
            while not end and i < max_tries:
                guess = agent.guess(msg)
                end, msg = game.evaluate_guess(guess)
                if 'not a valid' in msg:
                    repeated_failures += 1
                    if repeated_failures > 1:
                        failures[sol] = game.round_num
                        break
                else:
                    repeated_failures = 0
                if end:
                    if 'won' in msg:
                        successes[sol] = game.round_num
                    else:
                        failures[sol] = game.round_num
            print(msg)
        except Exception as e:
            raise e

    print(successes)
    print(failures)
print(cb)