# Make this standard
# Build an automatic run for evaluation by LLM-as-a-judge

from agent import FitAgent
from utils import parse_arguments
import asyncio
from langchain_community.llms import Ollama

VERBOSE = True

async def main():
    args = parse_arguments()
    if VERBOSE:
        print(f"Type: {args.type}")
        print(f"Value: {args.value}")
        print(f"Unit: {args.unit}")
        print(f"Start Date: {args.startdate}")
        print(f"End Date: {args.enddate}")

    agent = FitAgent(
        strategy="general"
    )
    response = agent.run(
        arguments={
            "type": args.type,
            "value": args.value,
            "unit": args.unit,
            "startdate": args.startdate,
            "enddate": args.enddate
        }
    )
    llm_judge = Ollama(model=args.judge_name)
    prompt_input = f"You are a LLM judge for our agent and you are able to judge the output of the health agent. The health agent should generate good response based on our data input. \
    Please fairly evaluate its agent's output based on the data I will give to you and rate it from 0 to 10. 0 means the agent gives a totally wrong response and 10 means the agent gives a totally correct response."
    response = llm.invoke(prompt_input)
    print(response)



if __name__ == "__main__":
    asyncio.run(main())


# python run.py --type heart-rate --value 178 --unit rate/min --startdate 2024-01-04 --enddate 2024-01-04