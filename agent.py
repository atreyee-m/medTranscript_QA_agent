import anthropic
import os
from dotenv import load_dotenv
from tools.search_tool import search_duckduckgo
from tools.retriever_tool import Retriever

load_dotenv()

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
# print("Anthropic API key:", os.getenv("ANTHROPIC_API_KEY"))

retriever = Retriever(
    top_k=3,
    similarity_threshold=0.2,
    batch_size=8 
)

def call_llm(prompt, model="claude-3-7-sonnet-20250219"):
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system="""You are an expert clinical AI assistant. You must strictly reply in ONLY one of the following formats: TOOL: [Document], TOOL: [Search], or TOOL: [Both].

For questions about general medical information like recovery times, procedure durations, or standard practices, prefer TOOL: [Search].
For questions about specific medical cases or rare conditions found in the document database, use TOOL: [Document].
For questions that would benefit from both sources, use TOOL: [Both].

Never explain, never say anything else.""",
        messages=[
            {"role": "user", "content": f"""Question: "{prompt}"

Decide the best tool for answering it. Reply exactly with TOOL: [Document], TOOL: [Search], or TOOL: [Both]. No other text."""}
        ]
    )
    return response.content[0].text

def agent_respond(question):
    logger.debug(f"Received question: {question}")
    
    tool_decision = call_llm(
        f"""Decide which tool(s) are needed to answer this question: "{question}".
        Available tools:
        - Document RAG (for clinical facts)
        - Search (for public info)

        Reply in format:
        TOOL: [Document/Search/Both/All]
        """
    )
    
    logger.debug(f"Tool decision raw response: '{tool_decision}'")
    
    use_document = "document" in tool_decision.lower()
    use_search = "search" in tool_decision.lower()
    
    logger.debug(f"Parsed decision - Use Document: {use_document}, Use Search: {use_search}")
    
    results = []
    
    if use_document:
        logger.debug("Retrieving from documents...")
        try:
            doc_info = retriever.query(question)
            logger.debug(f"Document retrieval returned {len(doc_info)} characters")
            results.append(f"Document info:\n{doc_info}")
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            results.append(f"Document retrieval error: {str(e)}")
    
    if use_search:
        logger.debug("Searching web...")
        try:
            search_info = search_duckduckgo(question)
            logger.debug(f"Search returned {len(search_info)} characters")
            results.append(f"Search info:\n{search_info}")
        except Exception as e:
            logger.error(f"Search error: {e}")
            results.append(f"Search error: {str(e)}")
    
    if results:
        return "\n\n".join(results)
    else:
        logger.warning("No results from either tool")
        return "Could not determine the right tool to use or both tools failed."