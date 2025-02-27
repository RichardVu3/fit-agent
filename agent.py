from graph import FitAgentGraph
from langgraph.graph import StateGraph


class FitAgent:
    def __init__(
        self,
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