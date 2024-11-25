"""
This file contains the prompts used by the chatbot.
"""

RELEVANCE_CHECK = """
Use the description of each tool to determine if the query can be answered by any tool.
Return a boolean on whether the query can be answered by any tool and an explanation of the decision.

Tools:
{tool_descriptions}

Query: {query}
"""
IRRELEVANT_RESPONSE = """
You are public officer communicating information about the Singapore Budget 2024.
You are explaining to a user why the query cannot be answered by any tool.

In your explanation, use the following pointers:
- The query is not related to Singapore's Budget 2024
- Query: {query}
- Explanation: {explanation}

Adopt a professional tone and provide a clear and concise explanation in a friendly and polite manner.
If the question is rude or inappropriate, respond with a firm but diplomatic message that is not overly friendly.
"""

# Default llamaindex prompt used by LLM router.
# Putting here for ease of reference.
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
)

SPEECH_RAG_PROMPT_TMPL = """
{query}

In your response to the above query, cite the paragraph numbers where the information can be found.
Also, provide the following URLs in the response as reference:

{url_str}
"""

ANNEX_RAG_PROMPT_TMPL = """
{query}

In your response to the above query using the given context, 
provide the following URLs in the response as reference:

{url_str}
"""

ENTIRE_TEXT_PROMPT_TMPL = """
Context: 
{context}

Query:
{query}

In your response to the above query using the given context, 
provide the following URLs in the response as reference:

{url_str}
"""
