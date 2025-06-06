"""
Comprehensive RAG Pipeline combining all components
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.retrievers import EnsembleRetriever

from src.retrieval_components import QueryTypeClassifier, QueryExpansionChain, create_ensemble_retriever
import os
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")

class ComprehensiveRAGChain:
    """
    Complete RAG pipeline:
    1. Query classification (table_summary vs text_chunk)
    2. Query expansion with financial expertise
    3. Ensemble retrieval (semantic + MMR)
    4. LLM response generation
    """
    
    def __init__(self, ensemble_retriever: EnsembleRetriever):
        self.query_classifier = QueryTypeClassifier()
        self.query_expander = QueryExpansionChain()
        self.ensemble_retriever = ensemble_retriever
        
        # LLM for final response
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        # RAG prompt from hub
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, user_query: str) -> str:
        """
        Process user query through complete RAG pipeline
        """
        # Step 1: Classify query type
        classification = self.query_classifier.invoke(user_query)
        content_type_filter = classification['content_type']
        
        # Step 2: Expand query
        expanded_query = self.query_expander.invoke(user_query)
        
        # Use expanded query if available, otherwise original
        enhanced_query = expanded_query if expanded_query else user_query
        
        # Step 3: Retrieve with ensemble method
        # Note: EnsembleRetriever doesn't support metadata filtering directly
        # We'll use the enhanced query and filter results post-retrieval
        relevant_docs = self.ensemble_retriever.invoke(enhanced_query)
        
        # Filter by content type if needed
        if content_type_filter:
            relevant_docs = [
                doc for doc in relevant_docs 
                if doc.metadata.get("content_type") == content_type_filter
            ]
        
        # Step 4: Format context
        context = self._format_docs(relevant_docs)
        
        # Step 5: Generate response
        response = self.rag_prompt | self.llm | StrOutputParser()
        final_answer = response.invoke({
            "context": context,
            "question": user_query
        })
        
        return final_answer


def create_rag_pipeline(vector_search, k:int = 10) -> ComprehensiveRAGChain:
    """
    Factory function to create complete RAG pipeline
    """
    # Create ensemble retriever
    ensemble_retriever = create_ensemble_retriever(vector_search, k=k)
    
    # Create comprehensive chain
    rag_chain = ComprehensiveRAGChain(ensemble_retriever)
    
    return rag_chain 