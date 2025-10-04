import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict

class InformationRetriever:
    def __init__(self, openai_model: str, excel_path: str, embeddings_path: str, embedding_model_name: str):
        print("Initializing the simplified retriever system...")
        self.excel_path = excel_path
        self.embeddings_path = embeddings_path
        
        print(f"Loading embedding model from Hugging Face Hub: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.openai_model = openai_model
        
        api_key = os.getenv('OPENAI_API_KEY') or "DUMMY_KEY" # Get key from env
        self.openai_client = OpenAI(api_key=api_key)
        
        self.knowledge_base_df, self.faiss_index = self._load_or_build_knowledge_base()
        print("✅ Simplified retriever initialized successfully.")

    def _load_or_build_knowledge_base(self):
        """
        Loads the knowledge base from files or builds it if necessary.
        This function is designed to be run once at startup.
        """
        print("\nLoading/Building knowledge base...")
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Knowledge base file not found at {self.excel_path}. App cannot start.")

        df = pd.read_excel(self.excel_path)
        
        required_cols = {'Text', 'Chunk ID', 'Source URL'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Excel file must contain the columns: {required_cols}")

        # Standardize column names
        df.rename(columns={'Text': 'text', 'Chunk ID': 'id', 'Source URL': 'source_url'}, inplace=True)
        df['text'] = df['text'].astype(str).fillna('')
        print(f"Loaded {len(df)} text chunks from '{self.excel_path}'.")

        # Generate new embeddings if cache is missing or mismatched
        if os.path.exists(self.embeddings_path):
            print("Found existing embeddings file. Checking for consistency...")
            knowledge_base_embeddings = np.load(self.embeddings_path)
            if knowledge_base_embeddings.shape[0] != len(df):
                print("⚠️ Mismatch detected. Regenerating embeddings...")
                knowledge_base_embeddings = self._generate_embeddings(df)
            else:
                print("✅ Embeddings match Excel file. Loading from cache.")
        else:
            print("⚠️ No embeddings file found. Generating new embeddings...")
            knowledge_base_embeddings = self._generate_embeddings(df)

        # Build FAISS index
        embedding_dimension = knowledge_base_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(embedding_dimension)
        # Normalize embeddings for cosine similarity search
        faiss.normalize_L2(knowledge_base_embeddings)
        faiss_index.add(knowledge_base_embeddings)
        print(f"✅ FAISS index built successfully with {faiss_index.ntotal} vectors.")
        
        return df, faiss_index

    def _generate_embeddings(self, df):
        """Generates and saves embeddings for the text chunks."""
        texts_to_embed = df['text'].tolist()
        embeddings = self.embedding_model.encode(
            texts_to_embed, show_progress_bar=True, batch_size=128
        )
        np.save(self.embeddings_path, embeddings)
        print(f"✅ New embeddings saved to '{self.embeddings_path}'.")
        return embeddings

    def search_and_answer(self, query: str, history: List[Dict[str, str]], top_k: int = 5) -> Dict:
        """
        Searches for relevant information and generates an answer.
        """
        # Create a condensed query if history is available
        search_query = self._create_history_aware_query(query, history)
        print(f"Original Query: '{query}' | Search Query: '{search_query}'")
        
        # Search the FAISS index
        query_embedding = self.embedding_model.encode([search_query])
        faiss.normalize_L2(query_embedding)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        
        retrieved_chunks = [self.knowledge_base_df.iloc[idx].to_dict() for idx in indices[0] if idx != -1]
        if not retrieved_chunks:
            return {"answer": "لم أجد أي معلومات ذات صلة بسؤالك في قاعدة البيانات.", "sources": []}

        # Prepare context for the language model
        source_texts = [chunk.get('text', 'N/A') for chunk in retrieved_chunks]
        context_text_for_prompt = "\n\n".join([f"Source [{i}]:\n{text}" for i, text in enumerate(source_texts)])
        
        # Prepare history for the language model
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        
        # Create the final prompt
        prompt = (
            "You are an intelligent assistant for King Saud University. Your task is to answer the user's question based ONLY on the provided sources below. "
            "If the answer is not explicitly found in the provided sources, you MUST state: 'لا أجد إجابة واضحة في المصادر المتوفرة لدي.' Do not use any external knowledge. "
            "When you form your answer, you MUST cite the sources you used by referencing their number in brackets, like `[0]`, `[1]`, etc.\n\n"
            f"## Conversation History:\n{history_str}\n\n"
            f"## Provided Sources:\n{context_text_for_prompt}\n\n"
            f"## User's Latest Question:\n{query}\n\n"
            "## Your Answer (in Arabic, citing sources):"
        )

        completion = self.openai_client.chat.completions.create(model=self.openai_model, messages=[{"role": "user", "content": prompt}])
        answer = completion.choices[0].message.content.strip()
        
        return {"answer": answer, "sources": retrieved_chunks}
    
    def _create_history_aware_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Creates a self-contained query based on the conversation history.
        """
        if not history or len(history) <= 1:
            return query
            
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        
        prompt = (
            f"Based on the following conversation history, generate a self-contained search query that can be understood without the context of the chat. The query should be about the user's latest question.\n\n"
            f"## Conversation History:\n{history_str}\n\n"
            f"## User's Latest Question: {query}\n\n"
            f"## Search Query:"
        )
        try:
            # Using a powerful model for this task is recommended
            completion = self.openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=60)
            rewritten_query = completion.choices[0].message.content.strip()
            return rewritten_query if rewritten_query else query
        except Exception as e:
            print(f"Error creating history-aware query, falling back to original query. Error: {e}")
            return query


