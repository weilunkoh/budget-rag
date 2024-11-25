import time
from typing import AsyncGenerator

from llama_index.core.base.response.schema import AsyncStreamingResponse

from src.workflow.orchestrator import get_default_workflow

orchestrator_workflow = get_default_workflow()


async def get_response_generator(query: str, orchestrator_workflow):
    """
    This file handles the response generation for the chatbot
    by taking in outputs from the orchestration and streaming
    the response to the user.

    Args:
        query (str): This is query and chat history from the user.
        orchestrator_workflow: This is the workflow stored in the user's session state.

    """
    response, source = await orchestrator_workflow.run(query=query)

    if isinstance(response, AsyncStreamingResponse):
        # source == "annex_rag_agent" or source == "speech_rag_agent"
        async for chunk in response.async_response_gen():
            print(chunk, end="", flush=True)
            # print(chunk, end="", flush=True)
            yield chunk
    # elif (
    #     source == "speech_agent"
    #     or source == "statement_agent"
    #     or source == "no_response_streaming"
    # ):
    elif isinstance(response, AsyncGenerator):
        # source == "speech_agent" or source == "statement_agent" or source == "no_response_streaming"
        async for chunk in response:
            # print(chunk.delta, end="", flush=True)
            yield chunk.delta
    else:
        # isinstance(response, str)
        # source == "medisave_agent" or source == "no_response"
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.05)
