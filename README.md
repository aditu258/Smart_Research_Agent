# Smart Research Assistant

Smart Research Assistant is an AI-powered research tool that helps you conduct comprehensive research on any topic. It combines the power of LangChain, LangGraph, and LangSmith to create a structured research workflow that breaks down complex queries into manageable steps.

## Features

- **Structured Research Process**: Automatically breaks down research questions into logical steps
- **LangGraph Workflow**: Orchestrates the entire research process with a directed graph
- **LangSmith Integration**: Tracks and analyzes performance with detailed tracing
- **Streamlit Web Interface**: Easy-to-use interface for conducting research and summarizing text
- **Multi-Model Support**: Configurable to work with different LLM providers

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-research-assistant.git
cd smart-research-assistant
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

This will start the web interface where you can:
- Enter research queries and get structured results
- Summarize documents
- View and download research results

### Using the Library Programmatically

```python
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph
from smart_research_assistant.langsmith_module import ResearchAssistantLangSmith

# Initialize the research assistant
assistant = ResearchAssistantLangSmith()

# Execute research with tracing
result = assistant.execute_with_tracing("What are the environmental impacts of electric vehicles?")

# Access the results
research_plan = result["result"]["research_plan"]
summary = result["result"]["summary"]
follow_up_questions = result["result"]["follow_up_questions"]
```

## Project Structure

```
smart_research_assistant/
├── __init__.py
├── langchain_module.py    # LangChain integration for basic operations
├── langgraph_module.py    # LangGraph workflow for research orchestration
├── langsmith_module.py    # LangSmith tracing and performance analysis
```

## How It Works

1. **Research Planning**: The assistant creates a detailed 5-step research plan for your query
2. **Step Execution**: Each step is systematically executed by generating search queries
3. **Information Analysis**: Retrieved information is analyzed to identify key insights
4. **Summary Generation**: A comprehensive summary is created along with follow-up questions
5. **Performance Tracking**: All operations are traced in LangSmith for analysis (if enabled)

## Configuration

The assistant can be configured to use different LLM models:

- Currently configured to use Groq's `llama-3.1-8b-instant` model
- Can be easily modified to use other models by changing the LLM initialization

## Development

### Adding New Capabilities

To extend the assistant with new capabilities:

1. Add new methods to the appropriate module
2. Update the LangGraph workflow if needed
3. Add new UI elements to the Streamlit app

### Running Tests

```bash
pytest tests/
```

## License

[MIT License](LICENSE)

## Acknowledgements

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Graph-based workflow powered by [LangGraph](https://github.com/langchain-ai/langgraph)
- Performance tracking with [LangSmith](https://smith.langchain.com/)
- Frontend built with [Streamlit](https://streamlit.io/)
