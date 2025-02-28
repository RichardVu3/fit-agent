from langgraph.graph import START, END, StateGraph
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal
from tools import RetrievalTool
from utils import get_llm
from langchain.output_parsers import PydanticOutputParser
from data_types import ALL_DATA_TYPES
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from datetime import datetime


today_date = datetime.today().strftime('%Y-%m-%d')


class Data(BaseModel):
    type: str = Field(..., description=f"The type of data, must be exactly one of {ALL_DATA_TYPES.keys()}")
    value: int = Field(..., description="The value of the data, e.g. 120, 8 hours, etc.")
    unit: str = Field(..., description="The unit of the data, e.g. rate/min, hours, etc.")
    start_date: str = Field(..., description="The date of the data, e.g. 2024-01-04")
    end_date: str = Field(..., description="The date of the data, e.g. 2024-01-04")

    @property
    def info(self):
        return f"{self.type}: {self.value} {self.unit} on {self.start_date} to {self.end_date}"


class RetrievalStep(BaseModel):
    type: str = Field(..., description=f"The type of data to retrieve. Must be exactly one of {ALL_DATA_TYPES.keys()}")
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
        self.describe_llm: OllamaLLM = get_llm(model="medllama2", temperature=0.0)
        self.plan_llm: OllamaLLM = get_llm(model="llama3", temperature=0.3)
        self.insights_llm: OllamaLLM = get_llm(model="llama3.2:1b", temperature=0.3)
        self.stream = stream
        self.tool = RetrievalTool(
            llm=self.insights_llm,
        )

    async def describing_incoming_data(
        self,
        state: FitAgentState,
    ):
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
5. Ensure neutrality and professionalism:
- Do not include personal opinions.
- Keep the explanation scientific and objective.
- Be concise, straightforward and informative without unnecessary elaboration. You do not need to tell the user to consult with a healthcare professional.

When relevant, today's date is {today_date}.

-----------------------------------------------------------
Now, describe in details the incoming data: {incoming_data}
"""
        prompt = ChatPromptTemplate.from_template(template)
        touse_llm = prompt | self.describe_llm
        if self.stream:
            print(f"Describing incoming data:\n")
            response = ""
            async for chunk in touse_llm.astream(
                {"incoming_data": state["incoming_data"].info, "today_date": today_date}
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = await touse_llm.ainvoke(
                {"incoming_data": state["incoming_data"].info, "today_date": today_date}
            )
        return {"incoming_data_description": response}

    async def making_retrieval_plan(
        self,
        state: FitAgentState,
    ):
        parser = PydanticOutputParser(pydantic_object=RetrievalPlan)
        template: str = f"""
You are an expert in medical AI tasked with retrieving relevant historical health data to enhance contextual analysis. Your goal is to formulate a retrieval plan that will complement the incoming data to provide deeper insights. Follow these structured steps:
1. Understand the incoming data description:
- Carefully review the provided health data (type, value, unit, timestamp) based on the incoming data description.
2. Identify the necessary historical data:
- Select which types of historical data would provide meaningful context for analysis (e.g., past heart rate trends, previous sleep records, weight history, etc.). Retrieve as many data types as needed. A lot of data is critical for a comprehensive analysis about the user's overall health status.
- Consider whether the data type requires continuous tracking over multiple days (e.g., heart rate, sleep) or if the latest recorded value is sufficient (e.g., height, weight).
- Note that the data type must be exactly as in this list: """ + str(list(ALL_DATA_TYPES.keys())) + """.
3. Determine the appropriate retrieval range:
- Define a suitable time window for historical data retrieval based on medical reasoning.
- For time-series data (e.g. heart rate, sleep, vitals): Suggest a historical period that provides useful trend insights (e.g., past 14 days of sleep records, last 200 heart rate measurements).
- For static or infrequently updated data (e.g. weight, height): Retrieve only the latest recorded value.
4. Ensure completeness and accuracy:
- The plan should be structured logically and must not overlook any crucial health parameters.
- Avoid unnecessary or excessive data retrieval that does not contribute to meaningful insights.
5. Safety and compliance:
- Do not assume or predict a medical condition.
- Ensure that the retrieval plan aligns with established health monitoring best practices.
- The user clearly understands this is just the suggestion. Thus, you do not need to tell the user to consult with a healthcare professional.
- Be consise, straight to the point and informative without unnecessary elaboration. User only cares about the plan and not the details or explanation.

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
        touse_llm = prompt | self.plan_llm | parser

        response = await touse_llm.ainvoke(
            {"incoming_data_description": state["incoming_data_description"]}
        )

        if self.stream:
            print(f"Making a retrieval plan:\n")
            for step in response.plan:
                print(step)
            print(f"\n---\n")
        
        return {"retrieval_plan": response.plan}

    async def retrieving_data(
        self,
        state: FitAgentState,
    ):
        outputs = []
        for step in state["retrieval_plan"]:
            if self.stream:
                print(f"---Retrieving: {step}---")
            tool_output = self.tool._run(step.type, step.range)
            outputs.append(
                tool_output
            )
        if self.stream:
            print(f"\n---\n")
        return {"retrieved_data": "\n------------------------------\n".join(outputs)}
    
    async def generating_insights(
        self,
        state: FitAgentState,
    ):
        template: str = f"""
You are a medical AI system responsible for generating insights on the INCOMING health data. You must use the historical and related health metrics records retrieved from the database to contextualize and use as the background of the user current health status.
Your analysis must be medically accurate, factual, and safe. Your tasks include trend analysis, anomaly detection, and providing actionable recommendations. Follow these structured guidelines:
1. Trend Analysis:
- Compare the incoming data against historical records. You must use the historical data as a reference to analyze the current health status. Include this analysis in your response.
- Identify patterns or deviations (e.g., increased heart rate over the last week, decreasing sleep duration, stable weight trends, etc.).
- Include both value units and dates when analyzing.
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

**Response Structure**: your response should at least include:
- A thorough analysis of the historical data to set the background and benchmark of the user's health status. You must give numerical evidence for your analysis. Do not hallucinate or assume anything about the user's health status.
- Comparison of the incoming data with historical records and detect any trends of anomaly
- The correlation and insights of the data
- General wellness recommendations based on the insights
You are free to include any other information that you think is relevant to the user's health status.

**Response tone**: user wants a friendly tone as a health and fitness assistant. Don't use a serious tone such as of a doctor or a medical professional. Be friendly and supportive.

When relevant, today's date is {today_date}.

-----------------------------------------------------------
Here is the historical and related health metrics:
{state['retrieved_data']}
-----------------------------------------------------------

Here is the incoming health data:
{state['incoming_data'].info}

-----------------------------------------------------------
""" + "{question}"
        prompt = ChatPromptTemplate.from_template(template)
        touse_llm = prompt | self.insights_llm
        if self.stream:
            print(f"Generating insights:\n")
            response = ""
            async for chunk in touse_llm.astream(
                {"question": "What can you infer from the incoming data compare to my historical health records?"}
            ):
                response += chunk
                print(chunk, end="", flush=True)
            print(f"\n---\n")
        else:
            response = await touse_llm.ainvoke(
                {"question": "What can you infer from the incoming data compare to my historical health records?"}
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
        stream: bool = False,
        *args, **kwargs
    ):
        self.strategy = strategy
        self.graph: StateGraph = FitAgentGraph(
            strategy=strategy,
            stream=stream,
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
