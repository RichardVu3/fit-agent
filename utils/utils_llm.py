from langchain_ollama.llms import OllamaLLM


def get_llm(model: str = "llama3.2:1b", temperature: float = 0.5, *args, **kwargs) -> OllamaLLM:
    return OllamaLLM(
        model=model,
        temperature=temperature,
        **kwargs
    )
