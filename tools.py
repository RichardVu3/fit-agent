from langchain_ollama.llms import OllamaLLM
import xml.etree.ElementTree as ET
import os
import pandas as pd


def read_xml_data(
    file_path: str = "./export.xml"
) -> pd.DataFrame:
    if os.getenv("ENV", "dev").lower() == "prod":
        return pd.read_csv("record_data.csv")
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
    ) -> str:
        return self.record_data[self.record_data["type"] == data_type].sort_values(by=["endDate"], ascending=[False]).head(most_recent_value).to_markdown()
