from typing import Any, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai import OpenAI

from src.workflow.common import FunctionOutputEvent, InputEvent, ToolCallEvent


class FunctionCallingWorkflow(Workflow):
    """
    This is the function calling tool used by the Medisave Bonus agent.
    """

    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or OpenAI(model="gpt-4o-mini", temperature=0)
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(self, ev: StartEvent) -> InputEvent:
        query = ev.get("query")
        if not query:
            return None
        user_msg = ChatMessage(role="user", content=query)
        return InputEvent(user_input=[user_msg])

    @step
    async def handle_llm_input(self, ev: InputEvent) -> ToolCallEvent | StopEvent:
        user_input = ev.user_input

        response = await self.llm.achat_with_tools(self.tools, chat_history=user_input)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            print(f"\n\n{response}\n\n")
            return StopEvent(result=str(response).replace("assistant:", ""))
        else:
            return ToolCallEvent(user_input=user_input, tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ev: ToolCallEvent) -> InputEvent:
        user_input = ev.user_input
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )
        tool_msgs_str = " ".join([msg.content for msg in tool_msgs])
        return StopEvent(result=tool_msgs_str)


def find_medisave_bonus(
    birth_year: int,
    num_property_own: Optional[int] = None,
    av_of_residence: Optional[int] = None,
) -> str:
    """Function to determine the medisave bonus for a person"""
    if birth_year <= 1959:
        return "You will get $750 worth of MediSave Bonus under the [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf)."
    elif birth_year <= 1973:
        if num_property_own is not None and num_property_own > 1:
            return "You will get $750 worth of MediSave Bonus under the [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf)."
        elif av_of_residence is not None and av_of_residence > 25000:
            return "You will get $750 worth of MediSave Bonus under the [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf)."
        elif (
            num_property_own is not None
            and num_property_own <= 1
            and av_of_residence is not None
            and av_of_residence <= 25000
        ):
            return "You will get $1,500 worth of MediSave Bonus under the [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf)."
        else:
            return f"For your birth year of {birth_year}, you will get between $750 to $1,500 worth of MediSave Bonus under the [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf). For a more exact value, please provide the number of properties you own and the annual value of your residence."

    elif birth_year <= 1983:
        if num_property_own is not None and num_property_own > 1:
            return "You will get $200 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        elif av_of_residence is not None and av_of_residence > 25000:
            return "You will get $200 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        elif (
            num_property_own is not None
            and num_property_own <= 1
            and av_of_residence is not None
            and av_of_residence <= 25000
        ):
            return "You will get $300 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        else:
            return f"For your birth year of {birth_year}, you will get between $200 to $300 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf). For a more exact value, please provide the number of properties you own and the annual value of your residence."

    elif birth_year <= 2003:
        if num_property_own is not None and num_property_own > 1:
            return "You will get $100 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        elif av_of_residence is not None and av_of_residence > 25000:
            return "You will get $100 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        elif (
            num_property_own is not None
            and num_property_own <= 1
            and av_of_residence is not None
            and av_of_residence <= 25000
        ):
            return "You will get $200 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf)."
        else:
            return f"For your birth year of {birth_year}, you will get between $100 to $200 worth of [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf). For a more exact value, please provide the number of properties you own and the annual value of your residence."
    else:
        return "You are not eligible for any MediSave Bonus based on both [one-time MediSave Bonus](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf) and [Majulah Package](https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf)."


def get_default_workflow():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    tools = [FunctionTool.from_defaults(find_medisave_bonus)]
    return FunctionCallingWorkflow(llm=llm, tools=tools, timeout=120, verbose=True)


async def run_standalone():
    workflow = get_default_workflow()

    # sample_query = "What is the MediSave Bonus for a person born in 1990 with 2 properties and an annual value of residence of $30,000?"
    sample_query = "how much medisave bonus can I get? I am born in 1955."

    result = await workflow.run(query=sample_query)
    print(result)


if __name__ == "__main__":
    # Run standalone module for ingesting data
    # python -m src.workflow.function_calling

    import asyncio

    asyncio.run(run_standalone())
