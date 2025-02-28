from agent import FitAgent
from utils import parse_arguments
import asyncio
from langchain_ollama.llms import OllamaLLM
from config import ENV

VERBOSE = ENV == "dev"

def main():
    args = parse_arguments()
    if VERBOSE:
        print(f"Type: {args.type}")
        print(f"Value: {args.value}")
        print(f"Unit: {args.unit}")
        print(f"Start Date: {args.startdate}")
        print(f"End Date: {args.enddate}")
        print()

    agent = FitAgent(
        strategy="general",
        stream=(ENV == "dev")
    )
    response = asyncio.run(
        agent.run(
            arguments={
                "type": args.type,
                "value": args.value,
                "unit": args.unit,
                "start_date": args.startdate,
                "end_date": args.enddate
            }
        )
    )
    if ENV == "prod":
        print(response)

def judge_response(judge_name, response_from_agent):
    llm_judge = OllamaLLM(model=judge_name)
    prompt_input = f"You are an LLM judge responsible for evaluating the output of a health agent. The health agent generates health suggestions or concern alerts based on the provided input data. \
    Your task is to fairly and objectively assess the quality of the agent's response by considering its accuracy, relevance, completeness, and appropriateness given the input data. Provide a single numerical score from 0 to 10, where:\
    0 = Completely incorrect or misleading response and 10 = Fully accurate, relevant, and appropriate response. Here is the agent's response for evaluation: " + response_from_agent
    response = llm_judge.invoke(prompt_input)
    return response



if __name__ == "__main__":
    main()


# python run.py --type heart-rate --value 200 --unit rate/min --startdate 2025-02-28 --enddate 2025-02-28 --judge_name deepseek-r1