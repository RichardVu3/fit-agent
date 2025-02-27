# Make this standard
# Build an automatic run for evaluation by LLM-as-a-judge

from agent import FitAgent
from utils import parse_arguments
import asyncio

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
    await agent.run(
        arguments={
            "type": args.type,
            "value": args.value,
            "unit": args.unit,
            "startdate": args.startdate,
            "enddate": args.enddate
        }
    )


if __name__ == "__main__":
    asyncio.run(main())


# python run.py --type heart-rate --value 178 --unit rate/min --startdate 2024-01-04 --enddate 2024-01-04