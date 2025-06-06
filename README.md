# Adidas Annual Report RAG System - Assignment

## Overview
This is a comprehensive RAG (Retrieval Augmented Generation) system for analyzing the Adidas 2024 Annual Report (492 pages) with advanced multi-modal data extraction and intelligent querying capabilities.

## Key Improvements Made

### 1. **Fixed Metadata Issues**
- ✅ **Added `content_type` metadata to text chunks**: Previously only table summaries had this metadata
- ✅ **Consistent metadata structure**: Both text chunks and table summaries now have proper content classification

### 2. **Implemented Comprehensive RAG Pipeline**
The system now follows this complete flow:
1. **Query Classification** → Determines if query needs table_summary or text_chunk data
2. **Query Expansion** → Financial expert LLM enhances query with domain knowledge  
3. **Ensemble Retrieval** → Combines semantic search and MMR
4. **Filtered Retrieval** → Uses classified metadata to filter relevant document types
5. **LLM Response Generation** → Uses RAG prompt from LangChain hub

### 3. **Advanced Retrieval with LangChain EnsembleRetriever**
- ✅ **LangChain EnsembleRetriever**: Uses built-in ensemble method with weighted combination
- ✅ **Two retrieval strategies**: 
  - Semantic search (vector similarity)
  - MMR (maximal marginal relevance)
- ✅ **Metadata filtering**: Post-retrieval filtering based on query classification
- ✅ **Equal weights**: 50% weight for each retrieval method

### 4. **Modularized Code Structure**
- ✅ **`retrieval_components.py`**: Contains query classification, expansion, and ensemble retriever
- ✅ **`rag_pipeline.py`**: Comprehensive RAG chain that ties everything together
- ✅ **`app.py`**: Streamlit web interface for interactive querying
- ✅ **`utils.py`**: Utility functions for DOCX export and query processing

### 5. **Enhanced User Interface**
- ✅ **Streamlit Web App**: Interactive interface with dialogue history
- ✅ **DOCX export functionality**: Save analysis results to Word documents
- ✅ **Error handling**: Proper error messages for failed operations

## Architecture Components

### Query Classification
```python
class QueryTypeClassifier:
    # Classifies queries as requiring 'table_summary' or 'text_chunk' data
    # Uses zero-shot prompting with financial domain expertise
```

### Query Expansion  
```python
class QueryExpansionChain:
    # Enhances user queries with financial expert knowledge
    # Generates variant rephrasings for better retrieval
```

### Ensemble Retrieval
```python
def create_ensemble_retriever(vector_search, k=10):
    # Combines semantic search and MMR with equal weights
    # Uses LangChain's built-in EnsembleRetriever for optimal performance
    # Post-retrieval metadata filtering for content type precision
```

### Comprehensive RAG Chain
```python
class ComprehensiveRAGChain:
    # End-to-end pipeline: classify → expand → retrieve → generate
    # Integrates all components into single interface
```

## Usage

### Running the Streamlit App
```bash
streamlit run app.py
```

### Running the Jupyter Notebook
```bash
# For Jupyter-style development
python rag-assignment-jupyter.py
```

### Direct API Usage
```python
from src.rag_pipeline import create_rag_pipeline

# Initialize with your vector store
rag_chain = create_rag_pipeline(vector_search, k=10)

# Query the system
answer = rag_chain.invoke("What are Adidas key financial metrics for 2024?")
```

## Performance Features

### Ensemble Retrieval Benefits
- **Better recall**: Combines strengths of semantic and lexical search
- **Improved diversity**: MMR reduces redundant results
- **Metadata awareness**: Filters by content type for precision

## Files Structure
```
├── app.py                     # Streamlit web interface
├── notebooks/
├────── rag-assignment-jupyter.py  # Jupyter notebook version
├── src/
├────── retrieval_components.py    # Modular retrieval classes
├────── rag_pipeline.py            # Comprehensive RAG pipeline
├────── utils.py                   # Utility functions (DOCX export, etc.)
├── README.md                      # Main documentation
├── README_APP.md                  # Streamlit app documentation
├── requirements.txt               # Dependencies
└── annual-report-adidas-ar24.pdf  # Source document
```

## Key Fixes Applied
1. **Metadata consistency**: All documents now have `content_type` field
2. **LangChain EnsembleRetriever**: Uses proven LangChain ensemble method (Semantic + MMR)
3. **Modular architecture**: Separated concerns into logical components
4. **Streamlit interface**: Clean web app with dialogue history and DOCX export
5. **Consolidated utilities**: Single DOCX function that handles both single queries and dialogue history
6. **Simplified deployment**: All components work together seamlessly

This system now provides a production-ready RAG pipeline with advanced retrieval capabilities and proper architectural separation.
