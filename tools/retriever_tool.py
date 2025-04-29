import re
import faiss
import numpy as np
import pandas as pd
import gc
import os
import time
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, csv_path="data/mtsamples_surgery.csv", top_k=3, similarity_threshold=0.2, batch_size=8):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadata = []
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self._build_index(csv_path)

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,?!:;()\[\]{}\-\'"]+', ' ', text)
        return text

    def _build_index(self, path):
        gc.collect()
        
        print(f"Loading CSV from {path}...")
        df = pd.read_csv(path)
        
        print(f"Loaded {len(df)} rows")
        
        print("Filtering and preprocessing texts...")
        df = df.dropna(subset=['transcription'])
        
        self.metadata = df[['medical_specialty', 'sample_name']].to_dict('records')
        
        self.texts = []
        for i in range(0, len(df), self.batch_size):
            batch = df['transcription'].iloc[i:i+self.batch_size].tolist()
            self.texts.extend([self._preprocess_text(text) for text in batch])
            gc.collect()
        
        print(f"Preprocessing complete. Starting encoding {len(self.texts)} documents...")
        
        for i in range(0, len(self.texts), self.batch_size):
            end_idx = min(i + self.batch_size, len(self.texts))
            batch = self.texts[i:end_idx]
            
            print(f"Encoding batch {i//self.batch_size + 1}/{(len(self.texts) + self.batch_size - 1)//self.batch_size}...")
            
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            
            faiss.normalize_L2(batch_embeddings)
            
            self.index.add(np.array(batch_embeddings))
            
            del batch_embeddings
            gc.collect()
            
            time.sleep(0.1)
        
        print(f"Index built with {len(self.texts)} documents")

    def add_documents(self, new_texts, new_metadata=None):
        if not new_texts:
            return
            
        processed_texts = [self._preprocess_text(text) for text in new_texts]
        
        # Add to existing texts and metadata
        self.texts.extend(processed_texts)
        if new_metadata:
            self.metadata.extend(new_metadata)
        
        # Encode and add to index
        for i in range(0, len(processed_texts), self.batch_size):
            batch = processed_texts[i:i+min(self.batch_size, len(processed_texts)-i)]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            faiss.normalize_L2(batch_embeddings)
            self.index.add(np.array(batch_embeddings))

    def query(self, question, include_metadata=True):
        try:
            q_embedding = self.model.encode([question])
            faiss.normalize_L2(q_embedding)
            
            k = min(self.top_k * 2, len(self.texts))
            scores, indices = self.index.search(np.array(q_embedding), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score >= self.similarity_threshold and i < self.top_k:
                    doc_text = self.texts[idx]
                    
                    if include_metadata and idx < len(self.metadata):
                        meta = self.metadata[idx]
                        doc_info = f"[Document {i+1}] (Score: {score:.2f}, Specialty: {meta.get('medical_specialty', 'Unknown')}, Sample: {meta.get('sample_name', 'Unknown')})\n\n{doc_text}"
                    else:
                        doc_info = f"[Document {i+1}] (Score: {score:.2f})\n\n{doc_text}"
                    
                    results.append(doc_info)
            
            gc.collect()
            
            if not results:
                return "No relevant documents found for this query."
            
            return "\n\n" + "-"*80 + "\n\n".join(results)
        except Exception as e:
            return f"Error during retrieval: {str(e)}"