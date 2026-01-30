from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union

Message     = Dict[str, Any]
ResponseFmt = Dict[str, Any]
Tool        = Dict[str, Any]
ToolChoice  = Dict[str, Any]
MemoryConfig = Dict[str, Any]
ThreadConfig = Dict[str, Any]
AgentConfig = Dict[str, Any]

class PipePreset(BaseModel):
    # identification
    name: str = Field(..., description="Unique pipe id")
    description: str = ""
    status: Literal["public", "private"] = "private"
    upsert: bool = False

    # model + generation params
    model: str = "openai:gpt-4o"
    stream: bool = False
    json_output: bool = False
    store: bool = True
    moderate: bool = False
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None

    # tools
    tools: List[Tool] = []
    tool_choice: Optional[Literal["auto", "required"] | ToolChoice] = None
    parallel_tool_calls: bool = False

    # prompt + vars
    messages: List[Message] = []
    variables: Dict[str, str] = {}

    # memory / extras
    memory: Optional[MemoryConfig] = None
    response_format: Optional[ResponseFmt] = None
    few_shot: Optional[List[Message]] = None
    safety_prompt: Optional[str] = None

    # Integration configurations
    thread: Optional[ThreadConfig] = None
    agent: Optional[AgentConfig] = None

    # Error handling and retry
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    # Advanced features
    json_output: bool = False
    enable_tool_execution: bool = True
    enable_memory_integration: bool = True
    enable_thread_integration: bool = True
    enable_agent_integration: bool = True
