from langchain_ollama.llms import OllamaLLM


def get_llm(model: str = "medllama2", temperature: float = 0.5, *args, **kwargs) -> OllamaLLM:
    return OllamaLLM(
        model=model,
        temperature=temperature,
        *args, **kwargs
    )
