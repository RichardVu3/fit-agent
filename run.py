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


if __name__ == "__main__":
    main()
