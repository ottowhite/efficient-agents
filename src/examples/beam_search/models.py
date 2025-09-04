from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI, OpenAI
from src.utils import system_prompt
import numpy as np

class Tokenizer:
    def __init__(self, model_name: str, chat_template: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.chat_template = chat_template

    def apply_chat_template(self, conversation: list[dict]) -> str:
        result = self.tok.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        # Ensure we return a string
        if isinstance(result, str):
            return result
        else:
            raise ValueError(f"Expected string from apply_chat_template, got {type(result)}")

class LLM:
    def __init__(self, model_name: str, base_url: str, temperature: float, tokenizer: Tokenizer):
        self.model = OpenAI(
            model=model_name,
            base_url=base_url,
            api_key="EMPTY",  # type: ignore
            streaming=True,
            temperature=temperature
        )
        self.tokenizer = tokenizer

    async def ainvoke(self, prompt: str) -> str:
        return await self.model.ainvoke(prompt, stop=["\n\n"])

    @staticmethod
    def get_system_prompt():
        return system_prompt

class PRM:
    def __init__(self, model_name: str, base_url: str):
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="EMPTY",  # type: ignore
            streaming=True,
        )
        self.num_top_logprobs = 5

    async def generate_score(self, conversation: list[dict]) -> float:
        prm_output = await self.model.ainvoke(
            conversation,
            stop=["\n\n"],
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.num_top_logprobs
        )

        plus_logprobs = None
        minus_logprobs = None
        assert len(prm_output.response_metadata["logprobs"]["content"]) == 1
        token_metadata = prm_output.response_metadata["logprobs"]["content"][0]
        for top_logprobs in token_metadata["top_logprobs"]:
            if top_logprobs["token"] == "+":
                plus_logprobs = top_logprobs["logprob"]
            elif top_logprobs["token"] == "-":
                minus_logprobs = top_logprobs["logprob"]

        assert not (plus_logprobs is None or minus_logprobs is None), "No logprobs found"

        return np.exp(plus_logprobs) / (np.exp(plus_logprobs) + np.exp(minus_logprobs))