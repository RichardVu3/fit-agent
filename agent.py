from graph import FitAgentGraph
from langgraph.graph import StateGraph
from typing import Literal


class FitAgent:
    def __init__(
        self,
        strategy: Literal["general", "medical", "mixed"],
        *args, **kwargs
    ):
        self.graph: StateGraph = FitAgentGraph(
            *args, **kwargs
        ).get_graph()

    async def run(
        self,
        arguments: dict,
        *args, **kwargs
    ):
        pass