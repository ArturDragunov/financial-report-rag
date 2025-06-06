"""
Retrieval components for RAG system with Ensemble retrieval
"""

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")

class QueryTypeClassification(BaseModel):
    content_type: str = Field(description="Metadata filter type for vector search. Allowed values: 'table_summary' or 'text_chunk'")
  
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        if v not in ('table_summary', 'text_chunk'):
            raise ValueError("content_type must be either 'table_summary' or 'text_chunk'")
        return v


class QueryTypeClassifier:
    def __init__(self):
        classification_prompt = ChatPromptTemplate.from_template(
            """You are a classification assistant specialized in annual report analysis.

Given this user query:
"{user_query}"

Determine if the information requested primarily relates to numeric data typically found in tables 
(e.g., revenue, financial metrics, years, numbers), or textual narrative/analysis.

Return JSON with a single field "content_type" set to either:
- "table_summary" if numeric/tabular data is required
- "text_chunk" if textual data is more relevant

Respond with only the JSON.

Example:
User query: "What was Adidas revenue growth in 2023?"
{{"content_type": "table_summary"}}

User query: "Describe Adidas brand strategy."
{{"content_type": "text_chunk"}}
"""
        )
        
        self.classify_llm = classification_prompt | ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.output_parser = JsonOutputParser(pydantic_class=QueryTypeClassification)

    def invoke(self, user_query: str) -> QueryTypeClassification:
        output = self.classify_llm.invoke({"user_query": user_query})
        return self.output_parser.parse(output.content)


class QueryExpansionChain:
    def __init__(self):
        query_expansion_template = ChatPromptTemplate.from_template(
            """You are a financial expert specialized in investments, annual reports, and investment banking. 
Your goal is to help improve a user's search query related to Adidas annual reports by rephrasings
it to better capture the intent and help with retrieval. 

Input query:
{user_query}

Provide one high-quality, diverse variant of this query for use in downstream semantic retrieval."""
        )
        
        self.llm_expander = query_expansion_template | ChatOpenAI(model=LLM_MODEL, temperature=0.3) | StrOutputParser()
    
    def invoke(self, user_query: str) -> str:
        return self.llm_expander.invoke({"user_query": user_query})


def create_ensemble_retriever(vector_search: MongoDBAtlasVectorSearch, k: int = 10) -> EnsembleRetriever:
    """
    Create an ensemble retriever combining:
    - Vanilla semantic search (vector similarity)
    - MMR search (maximal marginal relevance) 
    """
    
    # 1. Vanilla semantic retriever
    retriever_vanilla = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # 2. MMR retriever
    retriever_mmr = vector_search.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k*2}
    )
    
    # Create ensemble retriever with equal weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr], 
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever 