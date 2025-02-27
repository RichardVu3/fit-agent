from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import ollama
from langchain_community.llms import Ollama
from time import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pytz

class RetrievalTool(BaseTool):
    name: str = ""
    description: str = ""
    llm: OllamaLLM

    def __init__(
        self,
        llm_name,
        file_path
    ):

        # Define llm and preprocess the element tree here
        self.llm = Ollama(model=llm_name)
        tree = ET.parse(file_path)
        self.root = tree.getroot()


    def _run(
        self,
        data_type: str,
        most_recent_value: int,
        describe: bool = False
    ) -> str:
        pass
        # Retrieve the data from the element tree
        # Call the LLM to generate a description on the data (don't do summarization, we need as much data as possible)
        # Return the description
        data_input = []
        for record in self.root.findall("Record"):
            if  record.get("type") == data_type :
                date = record.get("startDate")
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S %z") 
                now = datetime.now(pytz.utc)
                time_days_ago = now - timedelta(days=most_recent_value)
                if date >= time_days_ago:
                    data_input += f"Data Type: {record.get("type")}, Date of data: {record.get("startDate")}, Data value: {record.get("value")}. "
        prompt_input = f"Please give me a description on the data type {data_type}. I will give you the data of most recent {most_recent_value} days. The data is: " + data_input
        response = self.llm.invoke(prompt_input)
        return prompt_input


                
                
