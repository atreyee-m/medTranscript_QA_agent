import os
from typing import Dict, List, Any, Optional
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict, List, Optional, Union
import copy

from tools.retriever_tool import DocumentRetriever
from tools.search_tool import WebSearchTool
from tools.pdf_tool import PDFProcessor

class AgentState(TypedDict):
    """State schema for the agent."""
    messages: List[Union[HumanMessage, AIMessage]]
    query: str
    csv_results: Optional[str]
    web_results: Optional[str]
    pdf_results: Optional[str]
    response: Optional[str]

class MedTranscriptAgent:
    def __init__(self, anthropic_api_key: Optional[str] = None, debug: bool = False):
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            anthropic_api_key=self.api_key,
            temperature=0.1
        )
        
        self.doc_retriever = DocumentRetriever()
        self.web_search = WebSearchTool(debug=debug)
        self.pdf_processor = PDFProcessor()
        self.debug = debug
        
        self.memory_store = MemorySaver()
        
        self.conversation_threads = {}
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for conversational QA"""
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("query_router", self._route_query)
        workflow.add_node("document_search", self._perform_doc_search)
        workflow.add_node("web_search", self._perform_web_search)
        workflow.add_node("pdf_search", self._perform_pdf_search)
        workflow.add_node("combine_results", self._generate_response)
        
        workflow.add_edge(START, "query_router")
        workflow.add_edge("query_router", "document_search")
        workflow.add_edge("query_router", "web_search")
        workflow.add_edge("query_router", "pdf_search")
        workflow.add_edge("document_search", "combine_results")
        workflow.add_edge("web_search", "combine_results")
        workflow.add_edge("pdf_search", "combine_results")
        workflow.add_edge("combine_results", END)
        
        return workflow.compile(checkpointer=self.memory_store)
    
    def _route_query(self, state: AgentState) -> Dict[str, Any]:
        """Determine which tool(s) to use for the query"""
        
        query = state["query"]
        messages = state.get("messages", [])
        
        if self.debug:
            print(f"[Router] Processing query with {len(messages)} existing messages")
        
        conversation_history = self._format_conversation_history(messages)
        
        routing_prompt = f"""
        You are a medical query router. Your job is to determine whether a query about medical topics should be:
        1. Answered using document search (for specific patient data or medical transcript information)
        2. Answered using web search (for general medical knowledge)
        3. Answered using PDF search (for detailed medical protocol or research documents)
        
        Consider the conversation history and the current query when making your decision.
        
        Conversation history:
        {conversation_history}
        
        Current query: {query}
        
        Respond with one or more of: "document", "web", "pdf"
        """
        
        route = self.llm.invoke(routing_prompt).content.strip().lower()
        
        if self.debug:
            print(f"[Router] Decision: {route}")
        
        next_steps = []
        if "document" in route:
            next_steps.append("document_search")
        if "web" in route:
            next_steps.append("web_search")
        if "pdf" in route:
            next_steps.append("pdf_search")
        
        if not next_steps:  
            next_steps = ["document_search", "web_search", "pdf_search"]
            
        return {"next": next_steps}
    
    def _perform_doc_search(self, state: AgentState) -> Dict[str, Any]:
        """Perform document search and return results"""
        query = state["query"]
        if self.debug:
            print(f"[Document Search] Searching for: {query}")
        results = self.doc_retriever.query(query)
        
        return {"csv_results": results}
    
    def _perform_web_search(self, state: AgentState) -> Dict[str, Any]:
        """Perform web search and return results"""
        query = state["query"]
        if self.debug:
            print(f"[Web Search] Searching for: {query}")
        results = self.web_search.search(query)
        
        return {"web_results": results}
    
    def _perform_pdf_search(self, state: AgentState) -> Dict[str, Any]:
        """Perform PDF search and return results"""
        query = state["query"]
        if self.debug:
            print(f"[PDF Search] Searching for: {query}")
        results = self.pdf_processor.search(query)
        
        return {"pdf_results": results}
    
    def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate a response based on search results and conversation history"""
        query = state["query"]
        messages = state.get("messages", [])
        
        if self.debug:
            print(f"[Generate Response] Processing with {len(messages)} messages in history")
        
        csv_results = state.get("csv_results", "No document results available")
        web_results = state.get("web_results", "No web results available")
        pdf_results = state.get("pdf_results", "No PDF results available")
        
        conversation_history = self._format_conversation_history(messages)
        
        response_prompt = f"""
        You are a helpful medical assistant answering questions about medical transcripts and general medical knowledge.
        You have access to three types of information sources: medical transcripts (CSV), web search results, and PDF documents.
        
        Conversation history:
        {conversation_history}
        
        Current query: {query}
        
        Document search results: {csv_results}
        
        Web search results: {web_results}
        
        PDF search results: {pdf_results}
        
        Based on all available information and your medical knowledge, provide a helpful, accurate, and compassionate response to the query.
        Make sure to consider the conversation history for context and continuity.
        When citing information, clearly indicate the source (Document, Web, or PDF).
        """
        
        response = self.llm.invoke(response_prompt).content
        
        updated_messages = messages + [
            HumanMessage(content=query),
            AIMessage(content=response)
        ]
        
        if self.debug:
            print(f"[Generate Response] History now has {len(updated_messages)} messages")
        
        return {
            "response": response,
            "messages": updated_messages
        }
    
    def _format_conversation_history(self, messages: List) -> str:
        """Format conversation history for inclusion in prompts"""
        if not messages:
            return "No previous conversation"
        
        formatted = []
        for i in range(0, len(messages), 2):
            if i < len(messages):
                user_msg = messages[i].content if i < len(messages) else ""
                ai_msg = messages[i+1].content if i+1 < len(messages) else ""
                formatted.append(f"Human: {user_msg}\nAI: {ai_msg}")
        
        return "\n\n".join(formatted)
    
    def load_pdf(self, file_path: str) -> str:
        """Load a PDF document into the agent"""
        return self.pdf_processor.load_pdf(file_path)
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """Process a message in a conversation thread"""
        if thread_id in self.conversation_threads:
            messages = self.conversation_threads[thread_id]
            if self.debug:
                print(f"[Chat] Retrieved {len(messages)} messages for thread {thread_id}")
        else:
            messages = []
            if self.debug:
                print(f"[Chat] Started new conversation thread {thread_id}")
        
        state = {
            "query": message,
            "messages": copy.deepcopy(messages)  
        }
        
        if self.debug:
            print(f"[Chat] Processing query with initial state containing {len(state['messages'])} messages")
        
        try:
            result = self.graph.invoke(
                state, 
                config={"configurable": {"thread_id": thread_id}}
            )
            
            updated_messages = result.get("messages", [])
            
            self.conversation_threads[thread_id] = copy.deepcopy(updated_messages)
            
            if self.debug:
                print(f"[Chat] Updated thread {thread_id} with {len(updated_messages)} messages")
            
            return result["response"]
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg