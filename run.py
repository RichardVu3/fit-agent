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



# python run.py --type heart-rate --value 200 --unit rate/min --startdate 2025-02-28 --enddate 2025-02-28
# python run.py --type oxygen-saturation --value 92 --unit percent --startdate 2025-02-28 --enddate 2025-02-28
# python run.py --type respiratory-rate --value 25 --unit breaths/min --startdate 2025-02-28 --enddate 2025-02-28
# python run.py --type heart-rate-variability-sdnn --value 20 --unit ms --startdate 2025-02-28 --enddate 2025-02-28
# python run.py --type apple-sleeping-breathing-disturbances --value 5 --unit events/hour --startdate 2025-02-28 --enddate 2025-02-28
