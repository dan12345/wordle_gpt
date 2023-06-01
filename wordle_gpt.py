from langchain.callbacks import get_openai_callback
from agents.simple_agents import ZeroShotSimpleAgent, FewShotSimpleAgent, SmartSimpleAgent, CoTAgent, \
    CoTLessContextAgent
from game import Game
# gpt-3.5-turbo
game = Game(max_rounds=6)
# agent = ZeroShotSimpleAgent(model='gpt-4')
# agent = FewShotSimpleAgent(model='gpt-4')
# agent = SmartSimpleAgent(model='gpt-4')
# agent = CoTAgent(model='gpt-4')
agent = CoTLessContextAgent(model='gpt-4')
msg = game.start_game()
end = False
with get_openai_callback() as cb:
    try:
        while not end:
            guess = agent.guess(msg)
            end, msg = game.evaluate_guess(guess)
        print(msg)
    except Exception as e:
        print(e)
    print(cb)
