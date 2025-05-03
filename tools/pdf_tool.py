import os
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class PDFProcessor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.pdf_docs = {}  
        self.vector_stores = {} 
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def load_pdf(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        
        doc_id = os.path.basename(file_path).split('.')[0]
        
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        chunks = text_splitter.create_documents([text])
        
        for i, chunk in enumerate(chunks):
            page_num = i // 3  
            chunk.metadata["source"] = f"{doc_id}_page_{page_num}"
        
        self.pdf_docs[doc_id] = chunks
        
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_stores[doc_id] = vector_store
        
        if self.debug:
            print(f"Loaded PDF {doc_id} with {len(chunks)} chunks")
        
        return doc_id
    
    def search(self, query: str, doc_id: Optional[str] = None, k: int = 4) -> str:
        if not self.pdf_docs:
            return "No PDF documents have been loaded yet."
            
        if doc_id and doc_id not in self.pdf_docs:
            return f"Document with ID {doc_id} not found."
            
        stores_to_search = [self.vector_stores[doc_id]] if doc_id else list(self.vector_stores.values())
        
        all_docs = []
        for store in stores_to_search:
            docs = store.similarity_search(query, k=min(k, len(store.index_to_docstore_id)))
            all_docs.extend(docs)
        
        if len(stores_to_search) > 1:
            all_docs = all_docs[:k]
        
        if not all_docs:
            return "No relevant information found in the PDF documents."
        
        results = []
        for i, doc in enumerate(all_docs):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            results.append(f"[PDF-{i+1}] {source}:\n{content}\n")
        
        formatted_results = "\n".join(results)
        
        if self.debug:
            print(f"PDF search results for query '{query}':")
            print(formatted_results)
        
        return formatted_results