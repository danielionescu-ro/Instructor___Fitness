from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq


class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = Groq()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model
