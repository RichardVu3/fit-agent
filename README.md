# FitAgent: AI-Powered Wearable Health Data Interpreter

## Overview

FitAgent is an AI-powered assistant designed to interpret wearable health data, providing personalized insights, health recommendations, and anomaly detection. Unlike conventional wearable applications that merely display raw health metrics or send simple alerts, FitAgent utilizes a multi-step agentic framework to provide medically contextualized interpretations and meaningful feedback. Key innovations in this project include:

- **Agentic Workflow**: The system decomposes the interpretation process into structured steps: medical description of incoming data, retrieval of relevant historical data, and insightful response generation.
- **Dynamic Data Retrieval**: Instead of processing all historical data at once, FitAgent intelligently selects and retrieves only the most relevant data for context-aware analysis.
- **Hybrid Model Utilization**: Different LLMs are used at various stages—medical LLMs handle data description and retrieval, while general LLMs focus on generating user-friendly insights.

## Running the Project

### Prerequisites

- Ensure you have Python 3.8+ installed.
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Export your Apple Health data from the Apple Health app and place the `export.xml` file in the root directory of this repository.

### Running FitAgent

1. Navigate to the project directory:
   ```bash
   cd fit-agent
   ```
2. Run the agent:
   ```bash
   python run.py --type heart-rate --value 200 --unit rate/min --startdate 2025-02-28 --enddate 2025-02-28
   ```
3. The system will parse the Apple Health data, process it using the AI framework, and provide insights based on detected trends and anomalies.

## Project Documentation

For a detailed explanation of the methodology, experimental results, and findings, refer to the full project report: [🔗 Project Report](./project_report.pdf)

Additionally, a summary of the key insights and findings is available in the presentation: [📊 Project Presentation](./project_presentation.pdf)

## Evaluation and Results

FitAgent was rigorously evaluated using multiple test cases and a structured evaluation framework. Some key findings include:

- **Medical LLMs are superior for data retrieval and description** but tend to provide blunt, less user-friendly responses.
- **General-purpose LLMs generate more engaging and nuanced explanations** but may lack precision in medical interpretation.
- **Fine-tuned medical models (e.g., HealthMateAI) demonstrate significant performance improvements** in both insight generation and user engagement.
- **Agentic workflows enhance response completeness and depth**, allowing for structured and adaptable health analysis.

## Contact

For inquiries, contributions, or collaborations, feel free to reach out or submit a pull request!
