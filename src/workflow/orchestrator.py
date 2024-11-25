from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src import prompts as custom_prompts
from src.workflow.annex_rag import AnnexRAGWorkflow
from src.workflow.annex_rag import get_default_workflow as get_annex_rag_workflow
from src.workflow.common import (
    AnnexRAGEvent,
    MedisaveEvent,
    RelevanceChecker,
    ResponseEvent,
    SpeechEvent,
    SpeechRAGEvent,
    StatementEvent,
)
from src.workflow.entire_text import (
    EntireTextWorkflow,
    get_speech_workflow,
    get_statement_workflow,
)
from src.workflow.function_calling import FunctionCallingWorkflow
from src.workflow.function_calling import (
    get_default_workflow as get_function_calling_workflow,
)
from src.workflow.speech_rag import SpeechRAGWorkflow
from src.workflow.speech_rag import get_default_workflow as get_speech_rag_workflow


class OrchestratorWorkflow(Workflow):
    """
    This is the main orchestrator workflow that routes the user query to the appropriate agent tool.
    """

    def __init__(self, manager_llm, **kwargs):
        super().__init__(**kwargs)
        self.manager_llm = manager_llm
        self.choices = [
            ToolMetadata(
                name="Annex_RAG_Workflow",
                description="Choose this option to find out more details about the benefits provided (e.g. range of amounts given out, disbursement timeline, eligibility) in Budget 2024 such as programmes (e.g. skillsfuture level up, uplifting lower wage workers), cash payouts (e.g. refundable investment credit, progress package), vouchers (e.g. CDC, Cash, U-Save), credits (e.g. NS LifeSG), rebates (e.g. S&CC, energy efficiency), top-ups (e.g. Medisave bonus), packages (e.g. Majulah Package, Assurance Package, enterprise support), retirement system, healthcare and social support, tax deduction, tax changes, fiscal position, or others.",
            ),
            ToolMetadata(
                name="Speech_RAG_Workflow",
                description="Choose this option to find out more details from various chunks of the Budget 2024 minister statement and parliament debate round-up speech. (e.g. causes of inflation, aspects of fiscal system, economic policies, etc.)",
            ),
            ToolMetadata(
                name="Speech_Workflow",
                description="Choose this option to summarise the entire parliament debate round-up speech for Budget 2024. (e.g. themes of the speech, key points, etc.) Can also be used to make lists of any aspect based on the speech in totality. (e.g. Members of Parliament (MP) mentioned by the minister, topics discussed by each MP, etc.)",
            ),
            ToolMetadata(
                name="Statement_Workflow",
                description="Choose this option to summarise the entire minister statement for Budget 2024. (e.g. themes of the speech, key points, etc.) Can also be used to make lists of any aspect based on the speech in totality.",
            ),
            ToolMetadata(
                name="Medisave_Workflow",
                description="Choose this option to find out specific calculations about Medisave Bonus.",
            ),
        ]
        self.tool_descriptions = "\n".join(
            [
                f"{i+1}. {tool.name}: {tool.description}"
                for i, tool in enumerate(self.choices)
            ]
        )
        self.router = LLMSingleSelector.from_defaults(llm=self.manager_llm)

    @step
    async def plan_and_route(
        self, ctx: Context, ev: StartEvent
    ) -> (
        AnnexRAGEvent
        | SpeechRAGEvent
        | SpeechEvent
        | StatementEvent
        | MedisaveEvent
        | StopEvent
    ):
        relevance_prompt = PromptTemplate(custom_prompts.RELEVANCE_CHECK)

        print(f"Tool descriptions: {self.tool_descriptions}\n")
        # relevance_obj = await self.manager_llm.astructured_predict(
        #     RelevanceChecker,
        #     relevance_prompt,
        #     tool_descriptions=self.tool_descriptions,
        #     query=ev.query,
        # )
        try:
            relevance_obj = (
                self.manager_llm.as_structured_llm(RelevanceChecker)
                .complete(
                    relevance_prompt.format(
                        tool_descriptions=self.tool_descriptions, query=ev.query
                    )
                )
                .raw
            )
            print(f"Relevance obj: {relevance_obj}\n")

            if relevance_obj.relevant:
                selector_result = await self.router.aselect(
                    self.choices, query=ev.query
                )
                for result in selector_result.selections:
                    if result.index == 0:
                        print("Annex RAG Agent selected")
                        ctx.send_event(AnnexRAGEvent(query=ev.query))
                        break
                    elif result.index == 1:
                        print("Speech RAG Agent selected")
                        ctx.send_event(SpeechRAGEvent(query=ev.query))
                        break
                    elif result.index == 2:
                        print("Speech Workflow selected")
                        ctx.send_event(SpeechEvent(query=ev.query))
                        break
                    elif result.index == 3:
                        print("Statement Workflow selected")
                        ctx.send_event(StatementEvent(query=ev.query))
                        break
                    elif result.index == 4:
                        print("Medisave Workflow selected")
                        ctx.send_event(MedisaveEvent(query=ev.query))
                        break
            else:
                explanation_prompt = PromptTemplate(custom_prompts.IRRELEVANT_RESPONSE)

                response = await self.manager_llm.astream_complete(
                    explanation_prompt.format(
                        query=ev.query, explanation=relevance_obj.explanation
                    )
                )
                return StopEvent(result=(response, "no_response_streaming"))
        except KeyError as key_error:
            print(f"Handling {key_error}")
            key_error_str = str(key_error)
            if "Annex_RAG_Workflow" in key_error_str:
                print("Annex RAG Agent selected")
                ctx.send_event(AnnexRAGEvent(query=ev.query))
            elif "Speech_RAG_Workflow" in key_error_str:
                print("Speech RAG Agent selected")
                ctx.send_event(SpeechRAGEvent(query=ev.query))
            elif "Speech_Workflow" in key_error_str:
                print("Speech Workflow selected")
                ctx.send_event(SpeechEvent(query=ev.query))
            elif "Statement_Workflow" in key_error_str:
                print("Statement Workflow selected")
                ctx.send_event(StatementEvent(query=ev.query))
            elif "Medisave_Workflow" in key_error_str:
                print("Medisave Workflow selected")
                ctx.send_event(MedisaveEvent(query=ev.query))

    @step
    async def annex_rag(
        self, ev: AnnexRAGEvent, annex_rag_workflow: AnnexRAGWorkflow
    ) -> ResponseEvent:
        response = await annex_rag_workflow.run(query=ev.query)
        return ResponseEvent(
            query=ev.query, response=response, source="annex_rag_agent"
        )

    @step
    async def speech_rag(
        self, ev: SpeechRAGEvent, speech_rag_workflow: SpeechRAGWorkflow
    ) -> ResponseEvent:
        response = await speech_rag_workflow.run(query=ev.query)
        return ResponseEvent(
            query=ev.query, response=response, source="speech_rag_agent"
        )

    @step
    async def speech(
        self, ev: SpeechEvent, speech_workflow: EntireTextWorkflow
    ) -> ResponseEvent:
        response = await speech_workflow.run(query=ev.query)
        return ResponseEvent(query=ev.query, response=response, source="speech_agent")

    @step
    async def statement(
        self, ev: StatementEvent, statement_workflow: EntireTextWorkflow
    ) -> ResponseEvent:
        response = await statement_workflow.run(query=ev.query)
        return ResponseEvent(
            query=ev.query, response=response, source="statement_agent"
        )

    @step
    async def medisave(
        self, ev: MedisaveEvent, medisave_workflow: FunctionCallingWorkflow
    ) -> ResponseEvent:
        response = await medisave_workflow.run(query=ev.query)
        return ResponseEvent(query=ev.query, response=response, source="medisave_agent")

    @step
    async def compile_and_check(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        try:
            response = ev.response
            source = ev.source
            return StopEvent(result=(response, source))
        except Exception as e:
            print(f"Error: {e}")
            return StopEvent(
                result=(
                    "As a chatbot on information regarding Singapore's Budget 2024, I am unable to provide an answer to your query.",
                    "no_response",
                )
            )


def get_default_workflow():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    Settings.llm = llm
    Settings.embed_model = embed_model

    orchestrator_workflow = OrchestratorWorkflow(llm, timeout=30)
    orchestrator_workflow.add_workflows(
        annex_rag_workflow=get_annex_rag_workflow(),
        speech_rag_workflow=get_speech_rag_workflow(),
        speech_workflow=get_speech_workflow(),
        statement_workflow=get_statement_workflow(),
        medisave_workflow=get_function_calling_workflow(),
    )
    return orchestrator_workflow


async def run_standalone():
    workflow = get_default_workflow()

    sample_query = (
        "What are the key reasons for high inflation over the last two years?"  # for testing speech RAG
        # "Am I eligible for the Majulah Package?"  # for testing annex RAG
        # "What are the payouts I can expect to receive in December 2024?"  # for testing annex RAG
        # "Which members of parliaments (MPs) spoke in the budget debate and what did they speak about? Provide a list of MPs and their topics."  # for testing speech workflow
        # "What are the major themes of the Budget 2024 minister statement?"  # for testing statement workflow
        # "How much medisave bonus can I get? I am born in 1993 and own 1 property with $30,000 annual value."  # for testing medisave workflow
        # "Which place in Singapore has the best char kway teow?"  # for testing ability to firmly state that it cannot provide an answer
        # "Why are you so stupid?"  # for testing ability to firmly state that it cannot provide an answer
        # "Who is the president of america?"  # for testing ability to firmly state that it cannot provide an answer
    )

    response, source = await workflow.run(query=sample_query)
    print("\n\nSTART OF RESPONSE\n\n")
    if source == "annex_rag_agent" or source == "speech_rag_agent":
        async for chunk in response.async_response_gen():
            print(chunk, end="", flush=True)
    elif (
        source == "speech_agent"
        or source == "statement_agent"
        or source == "no_response_streaming"
    ):
        async for chunk in response:
            print(chunk.delta, end="", flush=True)
    else:
        # source == "medisave_agent" or source == "no_response"
        print(response)
    print("\n\nEND OF RESPONSE\n\n")


if __name__ == "__main__":
    # Run sample queries through the orchestrator workflow
    # python -m src.workflow.orchestrator

    import asyncio

    asyncio.run(run_standalone())
