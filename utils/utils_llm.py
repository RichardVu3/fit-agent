from ENV import OPENAI_API_KEY, OPENAI_API_BASE, MODEL
# from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM


class LLMHelper:
    @classmethod
    def get_llm(*args, **kwargs) -> OllamaLLM:
        return OllamaLLM(
            base_url=kwargs.pop("base_url", OPENAI_API_BASE),
            model=kwargs.pop("model", MODEL),
            temperature=kwargs.pop("temperature", 0.5),
            **kwargs
        )