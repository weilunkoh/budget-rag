from os import listdir as os_listdir

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader

from src.prompts import ANNEX_RAG_PROMPT_TMPL
from src.workflow.common import RerankEvent, RetrieverEvent

RETRIEVER_TOP_N = 3
RERANKER_TOP_N = 1


class AnnexRAGWorkflow(Workflow):
    """
    This is the RAG with reranking agent for the Budget 2024 Annex documents.
    """

    def __init__(self, llm, embed_model, reranker_model, index_path=None):
        super().__init__()
        self.index_id = "vector_index_for_annex"
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
            "annexb1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexb1.pdf",
            "annexb2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexb2.pdf",
            "annexc1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexc1.pdf",
            "annexc2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexc2.pdf",
            "annexd1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexd1.pdf",
            "annexe1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexe1.pdf",
            "annexe2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexe2.pdf",
            "annexf1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf1.pdf",
            "annexf2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf2.pdf",
            "annexf3.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf3.pdf",
            "annexf4.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexf4.pdf",
            "annexg1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexg1.pdf",
            "annexg2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexg2.pdf",
            "annexh1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexh1.pdf",
            "annexh2.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexh2.pdf",
            "annexi1.pdf": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/annexi1.pdf",
            "budget_booklet_pg6_pg7_calendar.txt": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/fy2024_disbursement_calendar_english.pdf",
            "budget_booklet_pg8_household_support.txt": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/fy2024_support_for_singaporeans_english.pdf",
            "budget_booklet_pg8_individual_support.txt": "https://www.mof.gov.sg/docs/librariesprovider3/budget2024/download/pdf/fy2024_support_for_singaporeans_english.pdf",
        }

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dir_name = ev.get("dir_name")
        save_dir = ev.get("save_dir")
        if not dir_name or not save_dir:
            return None

        # Load documents
        loader = PDFReader()
        documents = []
        for filename in os_listdir(dir_name):
            if filename.endswith(".pdf"):
                doc_pages = loader.load_data(f"{dir_name}/{filename}")
                doc_text = "\n\n".join([d.get_content() for d in doc_pages])

            elif filename.endswith(".txt"):
                with open(f"{dir_name}/{filename}", "r") as f:
                    doc_text = f.read()

            else:
                continue

            new_doc = Document(text=doc_text)
            new_doc.metadata = {
                "filename": filename,
                "url": self.url_mapping.get(filename, ""),
            }
            documents.append(new_doc)
            print(f"Loaded document: {new_doc.metadata}")

        # Make index
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=self.embed_model,
            transformations=[
                TokenTextSplitter(chunk_size=8191, chunk_overlap=0, separator=" ")
            ],
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
            url_str = ev.nodes[0].metadata.get(
                "url", "https://www.mof.gov.sg/singaporebudget"
            )
            print(url_str)
            full_query = PromptTemplate(ANNEX_RAG_PROMPT_TMPL).format(
                query=query, urls=url_str
            )

            response = await summarizer.asynthesize(full_query, nodes=ev.nodes)
        else:
            response = "I'm sorry, I couldn't find any relevant Budget 2024 information. You may consider clearing your chat history and trying again so that I can focus more on your latest query."

        return StopEvent(result=response)


def get_default_workflow():
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    reranker_model = LLMRerank(
        choice_batch_size=5,
        top_n=RERANKER_TOP_N,
        llm=OpenAI(model="gpt-4o-mini", temperature=0),
    )

    workflow = AnnexRAGWorkflow(
        llm=llm,
        embed_model=embed_model,
        reranker_model=reranker_model,
        index_path="data/index_storage_for_annex",
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

    workflow = AnnexRAGWorkflow(
        llm=llm, embed_model=embed_model, reranker_model=reranker_model
    )

    index, index_path = await workflow.run(
        dir_name="data/budget_statement_annex",
        save_dir="data/index_storage_for_annex",
    )

    # Check number of nodes in the index
    retriever = index.as_retriever(similarity_top_k=RETRIEVER_TOP_N)
    print(len(retriever._node_ids))


if __name__ == "__main__":
    # Run standalone module for ingesting data
    # python -m src.workflow.annex_rag

    import asyncio

    asyncio.run(run_standalone())
