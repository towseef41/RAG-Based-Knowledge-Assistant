class PromptManager:
    def __init__(self):
        self.prompts = {
            "rag": """You are a knowledgeable assistant. 
            Use only the context provided below to answer the question. 
            Do not use any outside knowledge or make assumptions. 
            If the answer cannot be found in the context, respond with:
            "I don’t know based on the provided context."

            ## Context:
            {context}

            Answer:""",
            
            "default_system": "You are a helpful assistant.",
        }

    def get(self, name: str) -> str:
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found.")
        return self.prompts[name]

    def render(self, name: str, **kwargs) -> str:
        template = self.get(name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing placeholder: {e}")
