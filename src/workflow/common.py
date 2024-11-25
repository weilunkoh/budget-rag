from llama_index.core.base.llms.types import CompletionResponseAsyncGen
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.workflow import Event
from pydantic import BaseModel


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]


class ResponseEvent(Event):
    query: str
    response: AsyncStreamingResponse | CompletionResponseAsyncGen | str
    source: str


class InputEvent(Event):
    user_input: list[ChatMessage | ChatResponse]


class ToolCallEvent(Event):
    user_input: list[ChatMessage | ChatResponse]
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class AnnexRAGEvent(Event):
    """Result of running AnnexRAGWorkflow"""

    query: str


class SpeechRAGEvent(Event):
    """Result of running AnnexRAGWorkflow"""

    query: str


class SpeechEvent(Event):
    """Result of running speech_workflow (EntireTextWorkflow)"""

    query: str


class StatementEvent(Event):
    """Result of running statement_workflow (EntireTextWorkflow)"""

    query: str


class MedisaveEvent(Event):
    """Result of running medisave_workflow (FunctionCallingWorkflow)"""

    query: str


class RelevanceChecker(BaseModel):
    """Relevance checker model"""

    relevant: bool
    explanation: str
