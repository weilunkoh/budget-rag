from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.prompts import SPEECH_RAG_PROMPT_TMPL
from src.workflow.common import RerankEvent, RetrieverEvent

RETRIEVER_TOP_N = 5
RERANKER_TOP_N = 3


class SpeechRAGWorkflow(Workflow):
    """
    This is the RAG with reranking agent for the Budget 2024 statement and debate round up speech.
    """

    def __init__(self, llm, embed_model, reranker_model, index_path=None):
        super().__init__()
        self.index_id = "vector_index_for_speech"
        self.llm = llm
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.index_path = index_path
        if self.index_path is None:
            self.index = None
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = load_index_from_storage(
                storage_context, index_id=self.index_id
            )

        self.url_mapping = {
            "fy2024_budget_statement.pdf": "https://www.mof.gov.sg/singaporebudget/budget-2024/budget-statement",
            "fy2024_budget_debate_round_up_speech.pdf": "https://www.mof.gov.sg/singaporebudget/budget-2024/budget-debate-round-up-speech",
        }

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dir_name`."""
        dir_name = ev.get("dir_name")
        save_dir = ev.get("save_dir")
        if not dir_name or not save_dir:
            return None

        # Make index
        documents = SimpleDirectoryReader(dir_name).load_data()
        for document in documents:
            document.metadata["url"] = self.url_mapping.get(
                document.metadata["file_name"]
            )
        for document in documents:
            print(document.metadata)

        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=self.embed_model,
        )

        # Save index to disk
        self.index.set_index_id(self.index_id)
        self.index.storage_context.persist(save_dir)

        # Set attributes and return results
        self.index_path = save_dir
        return StopEvent(result=(self.index, self.index_path))

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if self.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = self.index.as_retriever(
            similarity_top_k=RETRIEVER_TOP_N,
            embed_model=self.embed_model,
        )
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = self.reranker_model
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        summarizer = CompactAndRefine(llm=self.llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        if len(ev.nodes) >= 1:
            url_set = set()
            for node in ev.nodes:
                url = node.metadata.get("url")
                if url:
                    url_set.add(url)

            if len(url_set) == 0:
                url_str = "https://www.mof.gov.sg/singaporebudget"
            else:
                url_str = "\n".join(url_set)

            full_query = PromptTemplate(SPEECH_RAG_PROMPT_TMPL).format(
                query=query, urls=url_str
            )

            response = await summarizer.asynthesize(full_query, nodes=ev.nodes)
        else:
            response = "I'm sorry, I couldn't find any relevant Budget 2024 information. You may consider clearing your chat history and trying again so that I can focus more on your latest query."

        return StopEvent(result=response)


def get_default_workflow():
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    reranker_model = LLMRerank(
        choice_batch_size=5, top_n=3, llm=OpenAI(model="gpt-4o-mini", temperature=0)
    )

    workflow = SpeechRAGWorkflow(
        llm=llm,
        embed_model=embed_model,
        reranker_model=reranker_model,
        index_path="data/index_storage_for_speech",
    )


def get_default_workflow():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    reranker_model = LLMRerank(
        choice_batch_size=5,
        top_n=RERANKER_TOP_N,
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
    )

    workflow = SpeechRAGWorkflow(
        llm=llm,
        embed_model=embed_model,
        reranker_model=reranker_model,
        index_path="data/index_storage_for_speech",
    )
    return workflow


# for ingesting data
async def run_standalone():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    reranker_model = LLMRerank(
        choice_batch_size=5,
        top_n=RERANKER_TOP_N,
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
    )

    workflow = SpeechRAGWorkflow(
        llm=llm, embed_model=embed_model, reranker_model=reranker_model
    )

    index, index_path = await workflow.run(
        dir_name="data/budget_statement_and_speech",
        save_dir="data/index_storage_for_speech",
    )

    # Check number of nodes in the index
    retriever = index.as_retriever(similarity_top_k=RETRIEVER_TOP_N)
    print(len(retriever._node_ids))


if __name__ == "__main__":
    # Run standalone module for ingesting data
    # python -m src.workflow.speech_rag

    import asyncio

    asyncio.run(run_standalone())
