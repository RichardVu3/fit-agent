from dotenv import load_dotenv
import os

load_dotenv()

ENV = os.getenv("ENV", "dev")
