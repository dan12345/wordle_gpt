class CoTAgent:
    def __init__(self, model):
        api_key = os.environ.get('OPEN_API_KEY')
        self.chat = ChatOpenAI(temperature=0, openai_api_key=api_key, model=model)
