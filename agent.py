from langgraph.graph import START, END, StateGraph
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, TypedDict, Literal
from tools import RetrievalTool
from utils import get_llm
from langchain.output_parsers import PydanticOutputParser
from data_types import ALL_DATA_TYPES
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


ENV = "dev"


class Data(BaseModel):
    type: str = Field(..., description=f"The type of data, must be exactly one of {ALL_DATA_TYPES}")
    value: int = Field(..., description="The value of the data, e.g. 120, 8 hours, etc.")
    unit: str = Field(..., description="The unit of the data, e.g. rate/min, hours, etc.")
    start_date: str = Field(..., description="The date of the data, e.g. 2024-01-04")
    end_date: str = Field(..., description="The date of the data, e.g. 2024-01-04")

    @property
    def info(self):
        return f"{self.type}: {self.value} {self.unit} on {self.start_date} to {self.end_date}"


class RetrievalStep(BaseModel):
    type: str = Field(..., description=f"The type of data to retrieve. Must be exactly one of {ALL_DATA_TYPES}")
    range: int = Field(..., description="The latest number of records to retrieve, e.g. 100, 14, etc.")


class RetrievalPlan(BaseModel):
    plan: List[RetrievalStep]


class FitAgentState(TypedDict):
    incoming_data: Data
    incoming_data_description: str
    retrieval_plan: List[RetrievalStep]
    retrieved_data: List[str]
    insights: str


class FitAgentGraph:
    def __init__(
        self,
        strategy: str,
        stream: bool = False,
        *args, **kwargs
    ):
        self.strategy: str = strategy
        # TODO: find a way to pass the LLM models
        self.general_llm: OllamaLLM = get_llm(model="llama3")
        self.medical_llm: OllamaLLM = get_llm(model="medllama2")
        self.stream = stream
        self.tool = RetrievalTool(
            llm=self.medical_llm,
        )

    def describing_incoming_data(
        self,
        state: FitAgentState,
    ):
        print(f"Describing incoming data:\n")
        template: str = """
You are a medical data analyst specializing in describing wearable health metrics. Your task is to analyze incoming personal health and fitness data and generate a detailed description of the provided measurements. The description should be precise, medically accurate, and free from hallucinations. Follow these guidelines:
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
- Be concise, straightforward and informative without unnecessary elaboration. You do not need to tell the user to consult with a healthcare professional.

-----------------------------------------------------------
Now, describe in details the incoming data: {incoming_data}
"""
        prompt = ChatPromptTemplate.from_template(template)
        touse_llm = prompt | self.medical_llm
        if self.stream:
            response = ""
            for chunk in touse_llm.stream(
                {"incoming_data": state["incoming_data"].info}
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = touse_llm.invoke(
                {"incoming_data": state["incoming_data"].info}
            )
        return {"incoming_data_description": response}

    async def making_retrieval_plan(
        self,
        state: FitAgentState,
    ):
        print(f"Making a retrieval plan:\n")
        parser = PydanticOutputParser(pydantic_object=RetrievalPlan)
        template: str = f"""
You are an expert in medical AI tasked with retrieving relevant historical health data to enhance contextual analysis. Your goal is to formulate a retrieval plan that will complement the incoming data to provide deeper insights. Follow these structured steps:
1. Understand the incoming data description:
- Carefully review the provided health data (type, value, unit, timestamp) based on the incoming data description.
2. Identify the necessary historical data:
- Select which types of historical data would provide meaningful context for analysis (e.g., past heart rate trends, previous sleep records, weight history, etc.).
- Consider whether the data type requires continuous tracking over multiple days (e.g., heart rate, sleep) or if the latest recorded value is sufficient (e.g., height, weight).
- Note that the data type must be exactly as in this list: """ + str(ALL_DATA_TYPES) + """.
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
- The user clearly understands this is just the suggestion. Thus, you do not need to tell the user to consult with a healthcare professional.
- Be consise, straight to the point and informative without unnecessary elaboration.

-----------------------------------------------------------
Here is how to structure the output: {format_instructions}

-----------------------------------------------------------
Now, make a plan to retrieve the data to contextualize the incoming data:
""" + " {incoming_data_description}"
        prompt = PromptTemplate(
            template=template,
            input_variables=["incoming_data_description"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        touse_llm = prompt | self.general_llm | parser

        response = await touse_llm.ainvoke(
            {"incoming_data_description": state["incoming_data_description"]}
        )

        if self.stream:
            for step in response.plan:
                print(step)
            print(f"\n---\n")
        
        return {"retrieval_plan": response.plan}

    def retrieving_data(
        self,
        state: FitAgentState,
    ):
        print(f"Retrieving data:\n")
        outputs = []
        for step in state["retrieval_plan"]:
            print(f"\n---{step}---\n")
            tool_output = self.tool._run(step.type, step.range, describe=True)
            outputs.append(
                tool_output
            )
        print(f"\n---\n")
        return {"retrieved_data": "\n------------------------------\n".join(outputs)}
    
    def generating_insights(
        self,
        state: FitAgentState,
    ):
        print(f"Generating insights:\n")
        template: str = f"""
You are a medical AI system responsible for generating insights on the INCOMING health data. You must use the historical and related health metrics records retrieved from the database to contextualize and use as the background of the user current health status.
Your analysis must be medically accurate, factual, and safe. Your tasks include trend analysis, anomaly detection, and providing actionable recommendations. Follow these structured guidelines:
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
- Trend Summary: An overview of observed trends of the historical data and the incoming data.
- Anomaly Detection: If applicable, a factual description of irregularities.
- Insights & Possible Correlations: A medically relevant interpretation of findings.
- General Wellness Recommendations: Practical and safe health tips without overstepping medical expertise.

Example Output:
- Trend Summary: The user's average sleep duration has declined over the past two weeks, with an increasing trend in resting heart rate.
- Anomaly Detected: Today's resting heart rate is 15% higher than the recent average.
- Insights: Sleep deprivation is a known factor in elevated heart rate, and the reduced sleep trend over the past two weeks may be influencing this change.
- General Wellness Recommendations: Ensuring adequate sleep and hydration may help regulate heart rate. If the elevated heart rate persists or is accompanied by other symptoms, consulting a healthcare professional is recommended.

-----------------------------------------------------------
Here is the description of the historical and related health metrics:
{state['retrieved_data']}
-----------------------------------------------------------

Now, Give me the insights of this incoming data: 
""" + " {incoming_data_description}"
        prompt = ChatPromptTemplate.from_template(template)
        touse_llm = prompt | self.general_llm
        if self.stream:
            response = ""
            for chunk in touse_llm.stream(
                {"incoming_data_description": state["incoming_data_description"]}
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = touse_llm.invoke(
                {"incoming_data_description": state["incoming_data_description"]}
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


class FitAgent:
    def __init__(
        self,
        strategy: Literal["general", "medical", "mixed"],
        *args, **kwargs
    ):
        self.strategy = strategy
        self.graph: StateGraph = FitAgentGraph(
            strategy=strategy,
            stream=(ENV=="dev"),
            *args, **kwargs
        ).get_graph()

    async def run(
        self,
        arguments: dict,
        *args, **kwargs
    ) -> str:
        data = Data(
            **arguments
        )
        response = await self.graph.ainvoke(
            {
                "incoming_data": data
            }
        )
        return response.get("insights", "An error happened. Please try again.")


if __name__ == "__main__":
    agent = FitAgent(
        strategy="general"
    )
    import asyncio
    response = asyncio.run(
        agent.run(
            arguments={
                "type": "heart-rate",
                "value": 200,
                "unit": "rate/min",
                "start_date": "2025-02-27",
                "end_date": "2025-02-27"
            }
        )
    )
