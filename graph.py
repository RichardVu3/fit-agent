from langgraph.graph import START, END, StateGraph
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, TypedDict
from tools import RetrievalTool
from utils import get_llm
from langchain.output_parsers import PydanticOutputParser


class Data(BaseModel):
    type: str = Field(..., description="The type of data, e.g. heart-rate, sleep, vitals, etc.")
    value: int = Field(..., description="The value of the data, e.g. 120, 8 hours, etc.")
    unit: str = Field(..., description="The unit of the data, e.g. rate/min, hours, etc.")
    date: str = Field(..., description="The date of the data, e.g. 2024-01-04")

    @property
    def info(self):
        return f"{self.type}: {self.value} {self.unit} on {self.date}"


class RetrievalStep(BaseModel):
    type: str = Field(..., description="The type of data to retrieve, e.g. heart-rate, sleep, vitals, etc.")
    range: int = Field(..., description="The latest number of records to retrieve, e.g. 100, 14, etc.")


class RetrievalPlan(BaseModel):
    plan: List[RetrievalStep]


class FitAgentState(TypedDict):
    incoming_data: Data
    incoming_data_description: str
    retrieval_plan: List[RetrievalStep]
    retrieved_data: List[str]


class FitAgentGraph:
    def __init__(
        self,
        strategy: str,
        stream: bool = False,
        *args, **kwargs
    ):
        self.strategy: str = strategy
        # TODO: find a way to pass the LLM models
        self.general_llm: OllamaLLM = kwargs.get("general_llm", get_llm())
        self.medical_llm: OllamaLLM = kwargs.get("medical_llm", get_llm())
        self.stream = stream

    def describing_incoming_data(
        self,
        state: FitAgentState,
    ):
        prompt: str = """
You are a medical data analyst specializing in wearable health metrics. Your task is to analyze incoming health data and generate a detailed medical description of the provided measurements. The description should be precise, medically accurate, and free from hallucinations. Follow these guidelines:
1. Identify and describe the health data:
- Clearly define the type of measurement (e.g., heart rate, sleep duration, respiratory rate, blood oxygen saturation, workout data, weight, height, etc.).
- Include the value, unit of measurement, and the timestamp of the recorded data.
2. Provide a medically sound explanation:
- Explain what the measurement represents physiologically.
- Describe the normal reference range if applicable (but do not assume whether the value is normal or abnormal).
3. Maintain factual accuracy and safety:
- Do not make assumptions about the user's health condition.
- Do not generate speculative or misleading information.
- Do not provide a diagnosis or medical advice—only factual information.
4. Format the response clearly:
Example:
- Heart Rate: Recorded at 85 bpm at 7:45 AM. The heart rate (beats per minute) represents the number of times the heart contracts per minute. Normal resting heart rate typically ranges between 60-100 bpm in adults, depending on factors such as fitness level and stress.
- Sleep Duration: Recorded as 6 hours 30 minutes on March 15. Sleep duration reflects the total amount of time spent asleep and is a key factor in cognitive and physical recovery. The recommended sleep duration for adults is generally 7-9 hours.
5. Ensure neutrality and professionalism:
- Do not include personal opinions.
- Keep the explanation scientific and objective.
"""
        touse_llm = self.general_llm if self.strategy == "general" else self.medical_llm
        messages = [SystemMessage(prompt)] + [HumanMessage(f"Describe in details the incoming data: {state['incoming_data'].info}")]
        if self.stream:
            response = ""
            for chunk in touse_llm.stream(
                messages
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = touse_llm.invoke(
                messages
            )
        return {"incoming_data_description": response}

    def making_retrieval_plan(
        self,
        state: FitAgentState,
    ):
        prompt: str = """
You are an expert in medical AI tasked with retrieving relevant historical health data to enhance contextual analysis. Your goal is to formulate a retrieval plan that will complement the incoming data to provide deeper insights. Follow these structured steps:
1. Understand the incoming data description:
- Carefully review the provided health data (type, value, unit, timestamp) based on the incoming data description.
2. Identify the necessary historical data:
- Select which types of historical data would provide meaningful context for analysis (e.g., past heart rate trends, previous sleep records, weight history, etc.).
- Consider whether the data type requires continuous tracking over multiple days (e.g., heart rate, sleep) or if the latest recorded value is sufficient (e.g., height, weight).
3. Determine the appropriate retrieval range:
- Define a suitable time window for historical data retrieval based on medical reasoning.
- For time-series data (e.g. heart rate, sleep, vitals): Suggest a historical period that provides useful trend insights (e.g., past 14 days of sleep records, last 100 heart rate measurements).
- For static or infrequently updated data (e.g. weight, height): Retrieve only the latest recorded value.
4. Ensure completeness and accuracy:
- The plan should be structured logically and must not overlook any crucial health parameters.
- Avoid unnecessary or excessive data retrieval that does not contribute to meaningful insights.
- Example of a well-structured retrieval plan:
    Incoming Data: Heart Rate = 92 bpm recorded at 8:00 AM
    Retrieval Plan:
        + Retrieve last 100 heart rate records to observe trends.
        + Retrieve last 14 days of sleep duration to assess any sleep-related impact on heart rate.
        + Retrieve latest weight and height to calculate BMI if necessary for cardiovascular assessment.
5. Safety and compliance:
- Do not assume or predict a medical condition.
- Ensure that the retrieval plan aligns with established health monitoring best practices.
Output format: Clearly structured retrieval plan specifying data types and time ranges.
"""
        messages = [SystemMessage(prompt)] + [HumanMessage(f"Make a plan to retrieve the data to contextualize the incoming data: {state['incoming_data_description']}")]
        touse_llm = self.general_llm if self.strategy == "general" else self.medical_llm
        if self.stream:
            streamed_response = ""
            for chunk in touse_llm.stream(
                messages
            ):
                streamed_response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
            parser = PydanticOutputParser(pydantic_object=RetrievalPlan)
            response: RetrievalPlan = parser.parse(streamed_response)
        else:
            response: RetrievalPlan = touse_llm.with_structured_output(RetrievalPlan).invoke(
                messages
            )
        return {"retrieval_plan": response.plan}

    def retrieving_data(
        self,
        state: FitAgentState,
    ):
        outputs = []
        for step in state["retrieval_plan"]:
            print(f"\n---{step}---\n")
            tool_output = RetrievalTool._run(step.type, step.range, describe=True)
            outputs.append(
                tool_output
            )
        return {"retrieved_data": outputs}
    
    def generating_insights(
        self,
        state: FitAgentState,
    ):
        prompt: str = """
You are a medical AI system responsible for generating insights from health data. Your analysis must be medically accurate, factual, and safe. Your tasks include trend analysis, anomaly detection, and providing actionable recommendations. Follow these structured guidelines:
1. Trend Analysis:
- Compare the incoming data against historical records.
- Identify patterns or deviations (e.g., increased heart rate over the last week, decreasing sleep duration, stable weight trends, etc.).
2. Anomaly Detection:
- Detect any unusual changes or irregularities.
- If an anomaly is detected, describe it in a neutral, factual manner. Example: "The heart rate recorded this morning (92 bpm) is elevated compared to the user's average resting heart rate of 72 bpm over the last 14 days. Such fluctuations can occur due to factors like stress, dehydration, or recent physical activity."
3. Generate Meaningful Insights:
- Provide an evidence-based interpretation of the trends.
- Use retrieved data to explain potential correlations. Example: "Sleep duration has decreased by an average of 1.5 hours per night over the past week, which may contribute to increased resting heart rate."
4. Offer Recommendations with Caution:
- If applicable, suggest general wellness recommendations backed by medical knowledge.
- Ensure that recommendations are non-diagnostic and do not replace medical consultation. Example: "If you are experiencing persistent elevated heart rate and fatigue, consider staying hydrated, getting adequate rest, and managing stress. If symptoms persist, consulting a healthcare professional is advisable."
5. Ensure Clarity and Safety:
- Avoid speculative or misleading statements.
- Do not diagnose or provide medical treatment plans—only insights and general health guidance.

Output Structure:
- Trend Summary: A concise overview of observed trends.
- Anomaly Detection: If applicable, a factual description of irregularities.
- Insights & Possible Correlations: A medically relevant interpretation of findings.
- General Wellness Recommendations: Practical and safe health tips without overstepping medical expertise.

Example Output:
- Trend Summary: The user's average sleep duration has declined over the past two weeks, with an increasing trend in resting heart rate.
- Anomaly Detected: Today's resting heart rate is 15% higher than the recent average.
- Insights: Sleep deprivation is a known factor in elevated heart rate, and the reduced sleep trend over the past two weeks may be influencing this change.
- General Wellness Recommendations: Ensuring adequate sleep and hydration may help regulate heart rate. If the elevated heart rate persists or is accompanied by other symptoms, consulting a healthcare professional is recommended.
"""
        touse_llm = self.general_llm if self.strategy == "general" else self.medical_llm
        messages = [SystemMessage(prompt)] + [HumanMessage("Generate insights from the retrieved data.")]
        if self.stream:
            response = ""
            for chunk in touse_llm.stream(
                messages
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = touse_llm.invoke(
                messages
            )
        return {"insights": response}

    def get_graph(
        self,
    ) -> StateGraph:
        graph = StateGraph(FitAgentState)
        graph.add_node(
            "describing_incoming_data",
            self.describing_incoming_data
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

        graph.add_edge(START, "describing_incoming_data")
        graph.add_edge("describing_incoming_data", "making_retrieval_plan")
        graph.add_edge("making_retrieval_plan", "retrieving_data")
        graph.add_edge("retrieving_data", "generating_insights")
        graph.add_edge("generating_insights", END)

        compiled_graph = graph.compile()

        with open("graph.png", "wb") as f:
            f.write(compiled_graph.get_graph().draw_mermaid_png())

        return compiled_graph
