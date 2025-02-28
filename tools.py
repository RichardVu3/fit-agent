from langchain_ollama.llms import OllamaLLM
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pytz
import pandas as pd


def read_xml_data(
    file_path: str = "./export.xml"
) -> pd.DataFrame:
    # record_data = pd.read_csv("record_data.csv")
    tree = ET.parse(file_path)
    root = tree.getroot()
    record_list = [x.attrib for x in root.iter('Record')]
    record_data = pd.DataFrame(record_list)
    record_data = record_data.drop(columns=["sourceName", "sourceVersion", "creationDate", "device"])
    record_data.dropna(subset=["unit"], inplace=True)
    record_data['type'] = record_data['type'].str.replace('HKQuantityTypeIdentifier', '').replace('HKDataType', '').replace('HKCategoryTypeIdentifier', '')
    record_data.to_csv("record_data.csv", index=False)
    return record_data

class RetrievalTool:
    name: str = "retrieva_tool"
    description: str = "A tool to retrieve data from an XML file and generate a description of the data using an LLM."
    llm: OllamaLLM

    def __init__(
        self,
        llm: OllamaLLM,
        file_path: str = "./export.xml"
    ):
        self.llm = llm
        self.record_data = read_xml_data(file_path)
        
    def get_all_data_types(self) -> list:
        return self.record_data['type'].unique().tolist()

    def _run(
        self,
        data_type: str,
        most_recent_value: int,
        describe: bool = False,
        stream: bool = True
    ) -> str:
        records = self.record_data[self.record_data["type"] == data_type].sort_values(by=["endDate"], ascending=[False]).head(most_recent_value).to_markdown()

        # if describe:
        #     prompt_input = (f"Analyze the following dataset for the data type '{data_type}'. "
        #         f"The data spans the last {most_recent_value} days. "
        #         "Provide a concise summary, key trends, and any notable patterns. "
        #         "Data entries are formatted as 'Type, Date, Value'. "
        #         f"Here is the data: {data_input}")
        
        #     if stream:
        #         response = ""
        #         for chunk in self.llm.stream(prompt_input):
        #             response += chunk
        #             print(chunk, end="", flush=True)
        #         print()
        #     else:
        #         response = self.llm.invoke(prompt_input)
        #     return response
        
        return records


if __name__ == "__main__":
    from utils import get_llm
    tool = RetrievalTool(
        llm=get_llm(),
        file_path="./export.xml"
    )

    data = tool._run(
        data_type="SleepDuration",
        most_recent_value=14,
        describe=False,
        stream=False
    )

    print(data)