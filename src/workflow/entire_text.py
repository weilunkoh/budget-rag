from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai import OpenAI

from src.prompts import ENTIRE_TEXT_PROMPT_TMPL


class EntireTextWorkflow(Workflow):
    """
    This is the sub-workflow used by the Budget 2024 statement agent and debate round up speech agent.
    """

    def __init__(self, text_file: str, llm):
        super().__init__()
        self.text_file = text_file
        with open(text_file, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.llm = llm
        self.url_mapping = {
            "data/entire_text/fy2024_budget_statement.txt": "https://www.mof.gov.sg/singaporebudget/budget-2024/budget-statement",
            "data/entire_text/fy2024_budget_debate_round_up_speech.txt": "https://www.mof.gov.sg/singaporebudget/budget-2024/budget-debate-round-up-speech",
        }

    @step
    async def ask_text_file(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        query = ev.get("query")
        if not query:
            return None
        url_str = self.url_mapping.get(
            self.text_file, "https://www.mof.gov.sg/singaporebudget"
        )
        full_query = PromptTemplate(ENTIRE_TEXT_PROMPT_TMPL).format(
            context=self.text, query=query, url_str=url_str
        )
        response = await self.llm.astream_complete(full_query)
        return StopEvent(result=response)


def get_speech_workflow():
    speech_workflow = EntireTextWorkflow(
        text_file="data/entire_text/fy2024_budget_debate_round_up_speech.txt",
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
    )
    return speech_workflow


def get_statement_workflow():
    statement_workflow = EntireTextWorkflow(
        text_file="data/entire_text/fy2024_budget_statement.txt",
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
    )
    return statement_workflow


async def run_standalone():
    speech_workflow = get_speech_workflow()
    statement_workflow = get_statement_workflow()

    speech_result = await speech_workflow.run(
        query="What is the budget debate round up speech about?"
    )
    async for chunk in speech_result:
        print(chunk.delta, end="", flush=True)

    statement_result = await statement_workflow.run(
        query="What is the budget statement about?"
    )
    async for chunk in statement_result:
        print(chunk.delta, end="", flush=True)


if __name__ == "__main__":
    # Run standalone module for testing querying on workflow
    # python -m src.workflow.entire_text

    import asyncio

    import tiktoken
    from llama_index.core import Settings

    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini").encode
    asyncio.run(run_standalone())
