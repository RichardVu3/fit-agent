from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM


class RetrievalTool(BaseTool):
    name: str = ""
    description: str = ""
    llm: OllamaLLM

    def __init__(
        self,
        llm: OllamaLLM,
    ):
        # Define llm and preprocess the element tree here
        pass

    def _run(
        self,
        type: str,
        most_recent_value: int,
        describe: bool = False
    ) -> str:
        pass
        # Retrieve the data from the element tree
        # Call the LLM to generate a description on the data (don't do summarization, we need as much data as possible)
        # Return the description
