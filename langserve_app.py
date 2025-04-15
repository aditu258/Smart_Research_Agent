import os
from fastapi import FastAPI, HTTPException
from langserve import add_routes
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field, validator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import your modules
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph
from smart_research_assistant.langsmith_module import ResearchAssistantLangSmith

# Initialize the app
app = FastAPI(
    title="Smart Research Assistant API",
    version="1.0",
    description="API for a Smart Research Assistant that can conduct comprehensive research and summarize text."
)

# Initialize components with error handling
try:
    langchain_module = ResearchAssistantLangChain()
    langgraph_module = ResearchAssistantLangGraph()
    langsmith_module = ResearchAssistantLangSmith()
except Exception as e:
    logger.error(f"Failed to initialize modules: {str(e)}")
    raise RuntimeError("Failed to initialize research assistant modules") from e

# Define Pydantic models with validation
class ResearchInput(BaseModel):
    query: str = Field(..., min_length=3, description="Research question to investigate")
    enable_tracing: bool = Field(False, description="Whether to enable LangSmith tracing")

    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v.strip()

class ResearchOutput(BaseModel):
    summary: str = Field(..., description="Research summary")
    research_plan: List[str] = Field(default_factory=list, description="Steps in the research plan")
    follow_up_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    analysis: str = Field(default="", description="Detailed analysis of findings")
    errors: List[str] = Field(default_factory=list, description="Any errors that occurred")

class SummaryInput(BaseModel):
    text: str = Field(..., min_length=10, description="Text to be summarized")

    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        return v.strip()

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="Generated summary")
    error: Optional[str] = Field(None, description="Error message if any")

class EchoInput(BaseModel):
    text: str = Field(..., description="Text to echo")

class EchoOutput(BaseModel):
    message: str = Field(..., description="Echo response")

# Enhanced research function with better error handling
async def research_function(input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        query = input_data.get("query", "")
        enable_tracing = input_data.get("enable_tracing", False)
        
        logger.info(f"Starting research for query: {query}")
        
        try:
            if enable_tracing:
                result = await langsmith_module.execute_with_tracing(query)
                data = result.get("result", result)
            else:
                data = await langgraph_module.execute_research(query)
                
            return {
                "summary": data.get("summary", "No summary generated"),
                "research_plan": data.get("research_plan", []),
                "follow_up_questions": data.get("follow_up_questions", []),
                "analysis": data.get("analysis", ""),
                "errors": data.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Research execution failed: {str(e)}")
            return {
                "summary": f"Research failed: {str(e)}",
                "research_plan": [],
                "follow_up_questions": [],
                "analysis": "",
                "errors": [str(e)]
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in research function: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced summarization function
async def summarize_function(input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        text = input_data.get("text", "")
        logger.info(f"Summarizing text of length: {len(text)}")
        
        summary = await langchain_module.summarize_document(text)
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return {"summary": f"Summarization failed: {str(e)}", "error": str(e)}

# Simple echo function for testing
async def echo_function(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": f"Received: {input_data['text']}"}

# Add routes with enhanced configuration
add_routes(
    app,
    research_function,
    path="/research",
    input_type=ResearchInput,
    output_type=ResearchOutput,
    config_keys=["configurable"],
    enabled_endpoints=["invoke", "stream"]
)

add_routes(
    app,
    summarize_function,
    path="/summarize",
    input_type=SummaryInput,
    output_type=SummaryOutput,
    config_keys=["configurable"],
    enabled_endpoints=["invoke", "stream"]
)

add_routes(
    app,
    echo_function,
    path="/echo",
    input_type=EchoInput,
    output_type=EchoOutput
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0",
        "services": {
            "langchain": True,
            "langgraph": True,
            "langsmith": hasattr(langsmith_module, 'tracing_enabled')
        }
    }

# Run with: uvicorn langserve_app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        timeout_keep_alive=60
    )