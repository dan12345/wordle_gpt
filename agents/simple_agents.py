from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import json


class SimpleAgent:
    """ A simple agent where all instructions are in the system prompt, and the agent can only respond with 5-letter words"""

    def __init__(self, model):
        api_key = os.environ.get('OPEN_API_KEY')
        self.chat = ChatOpenAI(temperature=0, openai_api_key=api_key, model=model)

    @staticmethod
    def ai_message_to_guess(ai_message):
        return ai_message

    @staticmethod
    def parse_human_msg(msg):
        if len(msg) == 5:
            return '[' + ','.join(msg) + ']'
        return msg

    def guess(self, msg):
        msg = self.parse_human_msg(msg)
        print(f"\033[33mHuman: {msg}\033[0m")
        self.messages.append(HumanMessage(content=msg))
        ai_message = self.chat(self.messages)
        self.messages.append(ai_message)
        guess = self.ai_message_to_guess(ai_message)
        print(f"\033[32mAI: {ai_message.content}\033[0m")
        return guess


zero_shot_system_msg = """You are an expert wordle player, and your goal is to guess a 5 letter word that I will choose, in the least rounds of possible. The possible solutions list consists of the most 2000 (give or take) common 5 letter words in english. 
You MUST guess a valid 5 letter word, not necessarily one of the possible solutions. You will get 6 possible guesses to guess the word, after which you will lose.
I will respond each time with an evaluation of your guess in the form of 5 letters, where G means you guessed the correct letter in the correct position, Y means you guessed the correct letter in the wrong position, and _ means you guessed the wrong letter. 
Note that if you guessed the same letter more than once and it appears more than once in the solution, you will get multiple Ys. If it appears only once, only the first instance will be counted as a Y. In case that one of the letters appears in the correct position, it will be prioritized as a G.
You MUST respond only with a 5 letter word and nothing else.   
"""

few_shot_examples = '''
Examples: 
1. 
I thought of a 5-letter word. Try to guess it!
tarse
_G___
cloud
_____
gimpy
____Y
hanky
_G__G
jazzy
You won in 6 rounds!

2. 
I thought of a 5-letter word. Try to guess it!
cloud
_YY__
loved
GG___
bossy
_GGGG
lossy
You won in 5 rounds!
'''

smart_logic = '''
As your goal is to guess the word with as few tries as possible, you should think smartly about your guesses. Some common strategies are:
1. Guessing the most common letters in english (e.g. E, T, A, O, I, N, S, H, R, D, L, U)
2. In case you got a 'G' which indicates right letter in right place - try to guess the same letter in the same place again. 
3. In case you got a _ which indicates wrong letter - try to guess a different letter in the same place.
4. In case you got a 'Y' which indicates right letter in wrong place - try to guess the same letter in a different place.
5. Think of the remaining valid solutions, and make a guess that with the highest expectation will eliminate the most words from the list of remaining solutions.
'''


class ZeroShotSimpleAgent(SimpleAgent):
    """ Zero shots - there is no good example in system message """

    def __init__(self, model='gpt-4'):
        super().__init__(model)
        self.messages = [SystemMessage(content=zero_shot_system_msg)]
        print(f'\033[34mSystem message: {self.messages[0].content}\033[0m')


class FewShotSimpleAgent(SimpleAgent):
    """ few shots - provide a few good examples in system message """

    def __init__(self, model='gpt-4'):
        super().__init__(model)
        self.messages = [SystemMessage(content=zero_shot_system_msg + few_shot_examples)]
        print(f'\033[34mSystem message: {self.messages[0].content}\033[0m')


class SmartSimpleAgent(SimpleAgent):
    """ Suggest logic in system message """

    def __init__(self, model='gpt-4'):
        super().__init__(model)
        self.messages = [SystemMessage(content=zero_shot_system_msg + smart_logic + few_shot_examples)]
        print(f'\033[34mSystem message: {self.messages[0].content}\033[0m')


cot_system_msg = '''You are an expert wordle player, and your goal is to guess a 5 letter word that I will choose, in the least rounds of possible. The possible solutions list consists of the most 2000 (give or take) common 5 letter words in english. 
You MUST guess each round a valid 5 letter word, not necessarily one of the possible solutions. You will get 6 possible guesses to guess the word, after which you will lose.
I will respond each time with an evaluation of your guess in the form of 5 letters, in the format ```[*,*,*,*,*]``` where G means you guessed the correct letter in the correct position, Y means you guessed the correct letter in the wrong position, and _ means you guessed the wrong letter. 
Note that if you guessed the same letter more than once and it appears more than once in the solution, you will get multiple Ys. If it appears only once, only the first instance will be counted as a Y. In case that one of the letters appears in the correct position, it will be prioritized as a G.
For example, if the word was scarf and you guessed sandy, the response would be GY___ - G because S was in the right place, Y because A was in the wrong place, while n,d and y were not in the word at all. 
You should state your reasoning for each guess explicitly. You should state explicitly your knowledge so far specifying which letters you know appear in the word and in which positions, which letters you know do not appear in the word, and which letters you are not sure about.
Use the following format:
{"knowledge":"Letters I know appear in the word: [a (second position), y (fifth position), j (first or third position)]\nLetters I know don't appear in the word: [b,c,d,e,h]\nLetters I am still unsure about: [f,s)", "reasoning": "your reasoning", "guess":" "your [g,u,e,s,s] "}
'''

cot_examples_msg = '''
Example: 
Human: I thought of a 5-letter word. Try to guess it!
AI: {"knowledge":"Letters I know appear in the word: [] Letters I know don't appear in the word: [] Letters I am still unsure about: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]", "reasoning": "I want to try a word that has many common letters, and also ends with an e since it is a common pattern. I will guess tarse", "guess": "[t,a,r,s,e]"}
Human: [_,G_,_,_,Y]
AI: {"knowledge":"Letters I know appear in the word: [a (second position), e (first/second/third or fourth position)] Letters I know don't appear in the word: [r,s,t] Letters I am still unsure about: [b,c,d,f,g,h,i,j,k,l,m,n,o,p,q,u,v,w,x,y,z]", "reasoning": "I want to try a word that has many common letters, and also ends with an e since it is a common pattern. I will guess candy", "guess": "[c,a,n,d,y]"}
.
.
.
AI: {"knowledge":"Letters I know appear in the word: [m (first position), a (second position), g (fourth position), y (fifth position) Letters I know don't appear in the word: [e,r,s,t] Letters I am still unsure about: [n,z]", "reasoning - I now know the solution is of the form ma*gy. The only word I have left that fits this is mangy":, "guess": "[m,a,n,g,y]"} 
You won, mangy is indeed the word I thought of!
'''


class CoTAgent(SimpleAgent):

    def __init__(self, model='gpt-4'):
        super().__init__(model)
        self.messages = [SystemMessage(content=cot_system_msg + cot_examples_msg + smart_logic)]
        print(f'\033[34mSystem message: {self.messages[0].content}\033[0m')

    def ai_message_to_guess(self, ai_message):
        import json
        guess = json.loads(ai_message.content)["guess"]
        guess = guess[1:-1].replace(",", "")
        return guess


cot_less_context_system_msg = '''You are an expert wordle player, and your goal is to guess a 5 letter word that I will choose. The possible solutions list consists of the most 2000 (give or take) common 5 letter words in english. 
You MUST guess a valid 5 letter word english, not necessarily one of the possible solutions. 
I will describe your previous guess, and the evaluation of it in the form of 5 letters, in the format ```[*,*,*,*,*]``` where G means you guessed the correct letter in the correct position, Y means you guessed the correct letter in the wrong position, and _ means you guessed the wrong letter. 
For example, if the word was scarf and you guessed sandy, the response would be GY___ - G because 's' was in the right place, Y because 'a' is in the solution but the wrong place, while n,d and y were not in the word at all.
To assist you, I will also tell you which letters you already know appear in the word and in which positions, which letters you know do not appear in the word, and which letters you are not sure about. In case there is no previous knowledge, I will simply ask you to guess a 5 letter word.
In your response you should state an updated knowledge of the word, your reasoning for each guess explicitly, and the guess itself. 
Use the following format:
{"knowledge":"Letters I know appear in the word: [a (second position), y (fifth position), j (first or third position)]\nLetters I know don't appear in the word: [b,c,d,e,h]\nLetters I am still unsure about: [f,s]", "reasoning": "your reasoning", "guess":" "your [g,u,e,s,s]"}

Example:
1. (no previous knowledge) 
Human: I thought of a 5-letter word. Try to guess it!
AI: {"knowledge":"Letters I know appear in the word: [] Letters I know don't appear in the word: [] Letters I am still unsure about: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]", "reasoning": "I want to try a word that has many common letters, and also ends with an e since it is a common pattern. I will guess tarse", "guess": "[t,a,r,s,e]"} 
 
2. (with previous knowledge)
Human: 
Your previous knowledge was - "Letters I know appear in the word: [e (first or second position)], Letters I know don't appear in the word: [r,s,t], Letters I am still unsure about: [a,b,c,d,f,g,h,i,j,k,l,m,n,o,p,q,u,v,w,x,y,z]"
You guessed ['c','a','n','d','y']. 
The evaluation was [_,G_,_,_,Y].
AI: {"knowledge":"Letters I know appear in the word: [a (second position), e (first position), y (third/fourth position)], Letters I know don't appear in the word: [c,d,n,r,s,t], Letters I am still unsure about: [b,d,f,g,h,i,j,k,l,m,o,p,q,u,v,w,x,z]", "reasoning": "your reasoning", "guess": "[m,a,y,b,e]"}
'''

cot_prompt_template = '''Your previous knowledge was - "{knowledge}"
You guessed {guess}. 
The evaluation was {evaluation}
'''


class CoTLessContextAgent(SimpleAgent):

    def __init__(self, model='gpt-4'):
        super().__init__(model)
        self.knowledge = None
        self.previous_guess = None
        print(f'\033[34mSystem message: {cot_less_context_system_msg}\033[0m')

    def ai_message_to_guess(self, ai_message):
        guess = json.loads(ai_message.content)["guess"]
        guess = guess[1:-1].replace(",", "")
        return guess

    def guess(self, msg):
        messages = [SystemMessage(content=cot_less_context_system_msg)]
        if self.knowledge is None:  # first guess
            print(f"\033[33mHuman: {msg}\033[0m")
            messages.append(HumanMessage(content=msg))
        else:
            # not first guess
            evaluation = self.parse_human_msg(msg)
            prompt = cot_prompt_template.format(knowledge=self.knowledge, guess=self.previous_guess, evaluation=evaluation)
            print(f"\033[33mHuman: {prompt}\033[0m")
            messages.append(HumanMessage(content=prompt))
        ai_message = self.chat(messages)
        print(f"\033[32mAI: {ai_message.content}\033[0m")
        self.knowledge = json.loads(ai_message.content)['knowledge']
        self.previous_guess = self.ai_message_to_guess(ai_message)
        return self.previous_guess
