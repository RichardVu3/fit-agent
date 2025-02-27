from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from utils import LLMHelper


class FitAgentState:
    pass


class FitAgentGraph:
    def __init__(
        self,
        *args, **kwargs
    ):
        self.general_llm: ChatOpenAI = kwargs.get("general_llm", LLMHelper.get_llm())
        self.medical_llm: ChatOpenAI = kwargs.get("medical_llm", LLMHelper.get_llm())

    async def describe_incoming_data(
        self,
        state: FitAgentState,
    ):
        pass

    async def making_retrieval_plan(
        self,
        state: FitAgentState,
    ):
        pass

    async def retrieving_data(
        self,
        state: FitAgentState,
    ):
        pass
    
    async def generating_insights(
        self,
        state: FitAgentState,
    ):
        pass

    def get_graph(
        self,
    ) -> StateGraph:
        graph = StateGraph(FitAgentState)
        graph.add_node(
            "describe_incoming_data",
            self.describe_incoming_data
        )
        graph.add_node(
            "making_retrieval_plan",
            self.making_retrieval_plan
        )
        graph.add_node(
            "retrieving_data",
            self.retrieving_data
        )
        graph.add_node(
            "generating_insights",
            self.generating_insights
        )

        graph.add_edge(START, "describe_incoming_data")
        graph.add_edge("describe_incoming_data", "making_retrieval_plan")
        graph.add_edge("making_retrieval_plan", "retrieving_data")
        graph.add_edge("retrieving_data", "generating_insights")
        graph.add_edge("generating_insights", END)

        compiled_graph = graph.compile()

        with open("graph.png", "wb") as f:
            f.write(compiled_graph.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    FitAgentGraph().get_graph()
