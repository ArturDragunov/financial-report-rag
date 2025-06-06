# %%[markdown]
"""
# second assisgnment is: take a multiple pdf with text,image,table
1. fetch the data from pdf
2. at lesat there should be 200 pages
3. if chunking(use the sementic chunking technique) required do chunking and then embedding
4. store it inside the vector database(use any of them 1. mongodb 2. astradb 3. opensearch 4.milvus) ## i have not discuss then you need to explore
5. create a index with all three index machnism(Flat, HNSW, IVF) ## i have not discuss then you need to explore
6. create a retriever pipeline
7. check the retriever time(which one is fastet)
8. print the accuray score of every similarity search
9. perform the reranking either using BM25 or MMR ## i have not discuss then you need to explore
10. then write a prompt template
11. generte a oputput through llm
12. render that output over the DOCx ## i have not discuss then you need to explore
as a additional tip: you can follow rag playlist from my youtube

after completing it keep it on your github and share that link on my  mail id:
snshrivas3365@gmail.com

and share the assignment in your community chat as well by tagging krish and sunny

deadline is: till firday 9PM
"""

# %%[markdown]
"""
# Automated Annual Report Analyzer - RAG System

## Project Overview

The goal is to build a pipeline that can extract, process, and intelligently query multi-modal content from PDF documents containing text, images, tables, and charts.

Our target document: **Adidas 2024 Annual Report (492 pages)**

## Architecture Goals
- Multi-modal data extraction and processing
- Semantic chunking for optimal context preservation
- Vector database storage with multiple indexing strategies
- Performance comparison of different retrieval methods
- Advanced reranking techniques
- LLM-powered analysis with structured output generation
"""

# %%[markdown]
"""
## Step 1: PDF Data Extraction Strategy
"""

# %%
# ALL IMPORTS - Consolidated
import os
import json
import re
import time
import asyncio
import warnings
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

# External libraries
import camelot
from pymongo import MongoClient
from dotenv import load_dotenv
from docx import Document as DocxDocument
from docx.shared import Pt

# Local imports
from src.rag_pipeline import create_rag_pipeline
from src.utils import render_to_docx, query_rag_system_streamlit

#%%
# Configuration
warnings.filterwarnings('ignore')
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")

# %%
# setup
pdf_path = "annual-report-adidas-ar24.pdf"
output_dir = Path("extracted")
output_dir.mkdir(exist_ok=True)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

table_prompt = ChatPromptTemplate.from_template(
  """Analyze this table and provide a concise summary:
  Table data: {table_data}
  Shape: {rows} rows, {columns} columns
  Page: {page_number}""")

summarize_chain = table_prompt | llm | StrOutputParser()

# %%
# extract text
loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_chunks = text_splitter.split_documents(documents)

# %%[markdown]
"""
Table extraction
"""

# %%
# extract tables
table_data = []
for flavor in ['lattice', 'stream']:
  tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
  for i, table in enumerate(tables):
    if table.parsing_report.get('accuracy', 0) > 50: # camelot is at least 50% confident this is a properly structured table
      table_data.append({
        "id": f"{flavor}_{i}",
        "page": table.parsing_report['page'],
        "df": table.df,
        "shape": table.shape
      })

# %%[markdown]
"""
Table summary
"""

# %%
async def summarize_table(semaphore, t):
  if t['shape'][0] < 2: # drop tables with less than 2 rows - no additional information
    return None
  async with semaphore:
    summary = await summarize_chain.ainvoke({
      "table_data": t['df'].to_string(index=False),
      "rows": t['shape'][0], 
      "columns": t['shape'][1],
      "page_number": t['page']
    })
    return {"id": t['id'], "page": t['page'], "summary": summary}

semaphore = asyncio.Semaphore(5)
table_summaries = [s for s in await asyncio.gather(*[summarize_table(semaphore, t) for t in table_data]) if s is not None]

# %%[markdown]
"""
Text cleaning - after visual overview, a few strings were found to be repeated
"""

# %%
# text cleaning - repeated strings
def clean_text(text):
  unwanted_strings = [
    "ANNUAL REPORT 2024",
    "1 2 3 4 5 6 \nT O  O U R  \nSHA REHO L D ERS  \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nO U R CO MPA N Y \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nF I N A N CI AL  REVI EW \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nSU ST A I N A B IL I T Y  STA T EMEN T \nCO N SO L I DA T ED   \nF I N A N CI AL  ST AT EMEN T S \nA D D I T I ON A L  \nI N F O RMA T IO N",
    "1 2 3 4 5 6 \nT O  O U R  \nSHA REHO L D ERS  \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nO U R CO MPA N Y \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nF I N A N CI AL  REVI EW \nGRO U P  MAN A GEMEN T  REP O RT \u2013 \nSU ST A I N A B IL I T Y  STA T EMEN T",
  ]

  cleaned = text
  for unwanted in unwanted_strings:
    cleaned = re.sub(re.escape(unwanted), '', cleaned, flags=re.IGNORECASE)

  # remove extra whitespace
  cleaned = re.sub(r'\s+', ' ', cleaned).strip()
  return cleaned

for chunk in text_chunks:
  chunk.page_content = clean_text(chunk.page_content)

# %%
with open(output_dir / "text_chunks.json", 'w') as f:
  json.dump([{"content": c.page_content, "metadata": c.metadata} for c in text_chunks], f)

with open(output_dir / "table_summaries.json", 'w') as f:
  json.dump(table_summaries, f)

# %%[markdown]
"""
## Step 2: Vector embedding and saving to MongoDB
"""

# %%
# Load the extracted text chunks
output_dir = Path("extracted")
with open(output_dir / "text_chunks.json", 'r') as f:
  chunks_data = json.load(f)

# Convert back to langchain documents with content_type metadata
text_chunks = []
for chunk in chunks_data:
  metadata = chunk["metadata"].copy()
  metadata["content_type"] = "text_chunk"  # Add content_type for text chunks
  text_chunks.append(Document(page_content=chunk["content"], metadata=metadata))

print(f"Loaded {len(text_chunks)} text chunks with content_type metadata")

# %%[markdown]
"""
MongoDB connection
"""

# %%
# MongoDB connection setup
MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
# Database and collection names
DB_NAME = "adidas_annual_report"
COLLECTION_NAME = "document_chunks"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Initialize MongoDB client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

print(f"Connected to MongoDB. Database: {DB_NAME}, Collection: {COLLECTION_NAME}")

# %%
# Initialize OpenAI embeddings with the new model
embeddings = OpenAIEmbeddings(
  model="text-embedding-3-small",  # New OpenAI embedding model
  dimensions=1536  # Default dimension for text-embedding-3-small
)

print("Initialized OpenAI embeddings with text-embedding-3-small")

# %%[markdown]
"""
Add documents to MongoDB
"""

# %%
# Create vector store and add documents
print("Creating embeddings and storing in MongoDB...")
vector_search = MongoDBAtlasVectorSearch(
  embedding=embeddings,
  collection=collection,
  index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)
uuids = [str(uuid4()) for _ in range(len(text_chunks))]

# Add to existing vector store
vector_search.add_documents(text_chunks, ids = uuids)
print(f"✅ Successfully stored {len(text_chunks)} documents with embeddings")

# %%[markdown]
"""
Adding table summaries to vector DB
"""

# %%
# Load and vectorize table summaries
with open(output_dir / "table_summaries.json", 'r') as f:
  table_summaries = json.load(f)

# Convert to documents  
table_docs = [
  Document(
    page_content=f"Table Summary: {summary['summary']}",
    metadata={
      "source": "annual-report-adidas-ar24.pdf",
      "page": summary['page'],
      "table_id": summary['id'], 
      "content_type": "table_summary"
    }
  ) for summary in table_summaries
]

uuids = [str(uuid4()) for _ in range(len(table_docs))]

# Add to existing vector store
vector_search.add_documents(table_docs, ids = uuids)

# Verify
total_docs = collection.count_documents({})
table_count = collection.count_documents({"content_type": "table_summary"})
print(f"✅ Total: {total_docs} docs ({table_count} table summaries)")

# %%[markdown]
"""
## Step 3: Vector Search Index in MongoDB Atlas, Speed/Accuracy tests, RAG pipeline

**Important**: You need to create a vector search index in MongoDB Atlas UI:

1. Go to your MongoDB Atlas cluster
2. Navigate to "Search" tab
3. Click "Create Search Index"
4. Choose "Atlas Vector Search"
5. Use this JSON configuration:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

6. Name the index: `vector_index`
7. Select your database and collection: `adidas_annual_report.document_chunks`

You can't create indexes programmatically with M0 cluster
"""

# %%[markdown]
"""
Test Vector Search
"""

# %%
# Test vector search (run after creating the index in Atlas UI)
# query = "What are Adidas key financial metrics for 2024?"
query = "What are Adidas environmental impact?"

# Perform similarity search
docs = vector_search.similarity_search(query, k=2) # index_name="vector_index" was already initialized in the vector_search object

print(f"Query: {query}")
print(f"\nFound {len(docs)} relevant documents:")

for i, doc in enumerate(docs, 1):
  print(f"\n--- Document {i} ---")
  print(f"Content: {doc.page_content[:200]}...")
  print(f"Page: {doc.metadata.get('page', 'N/A')}")
  print(f"Source: {doc.metadata.get('source', 'N/A')}")
  print(f"Content Type: {doc.metadata.get('content_type', 'N/A')}")

# %%[markdown]
"""
Comparison of two index methods
"""

# %%
# Compare speed and accuracy of different indexes
# Test queries
queries = [
  "What are Adidas key financial metrics for 2024?",
  "Sustainability initiatives and environmental impact", 
  "Brand strategy and marketing performance",
  "Revenue growth in different regions",
  "Digital transformation investments"
]
# Test ANN (Approximate Nearest Neighbor - default implementation in MongoDB Atlas)
print("ANN Search:")
ann_times = []
for query in queries:
  start = time.perf_counter()
  docs_scores = vector_search.similarity_search_with_score(query, k=5)
  ann_times.append((time.perf_counter() - start) * 1000)
  avg_score = sum(score for _, score in docs_scores) / len(docs_scores)
  print(f"ANN - {ann_times[-1]:.1f}ms, Score: {avg_score:.3f}")

# Test ENN (Exact Nearest Neighbor)  
print("\nENN Search:")
enn_times = []
for query in queries:
  start = time.perf_counter()
  docs_scores = vector_search.similarity_search_with_score(query, k=5, search_type="exact")
  enn_times.append((time.perf_counter() - start) * 1000)
  avg_score = sum(score for _, score in docs_scores) / len(docs_scores)
  print(f"ENN - {enn_times[-1]:.1f}ms, Score: {avg_score:.3f}")

# %%[markdown]
"""
| Query | ANN Time (ms) | ANN Score | ENN Time (ms) | ENN Score |
|-------|---------------|-----------|---------------|-----------|
| Query 1 | 317.4 | 0.858 | 1738.0 | 0.858 |
| Query 2 | 274.0 | 0.802 | 235.0 | 0.802 |
| Query 3 | 329.6 | 0.757 | 277.8 | 0.757 |
| Query 4 | 299.5 | 0.796 | 260.6 | 0.796 |
| Query 5 | 286.2 | 0.706 | 280.2 | 0.706 |
"""

# %%[markdown]
"""
## Step 4: RAG pipeline with ensemble retrieval
"""

# %%[markdown]
"""
### How `create_rag_pipeline()` Works

The `create_rag_pipeline()` function creates a comprehensive RAG system with multiple retrieval strategies:

**Pipeline Components:**
1. **Query Classification** - LLM determines if query needs `table_summary` or `text_chunk` data
2. **Query Expansion** - Financial expert LLM rephrases query for better retrieval
3. **Ensemble Retrieval** - Combines 3 retrieval methods:
   - **Semantic Search** - Vector similarity (cosine distance)
   - **MMR (Maximal Marginal Relevance)** - Reduces redundancy in results  
4. **Metadata Filtering** - Uses classification to filter by content type
5. **Result Fusion** - Combines and deduplicates results from all retrievers
6. **LLM Synthesis** - Generates final answer using retrieved context

**Key Benefits:**
- **Hybrid retrieval** captures both semantic and lexical matches
- **Smart filtering** routes queries to relevant content types
- **Reduced redundancy** through MMR and deduplication
- **Better recall** by combining multiple search strategies
"""

# %%
# Create comprehensive RAG pipeline
comprehensive_rag_chain = create_rag_pipeline(vector_search, k=10)

# Test queries
test_queries = [
  "What are Adidas key financial metrics for 2024?",
  "Sustainability initiatives and environmental impact", 
  "Brand strategy and marketing performance",
  "Revenue growth in different regions",
  "Digital transformation investments"
]

print("Testing Comprehensive RAG Pipeline:")
for query in test_queries:
  print(f"\nQuery: {query}")
  answer = comprehensive_rag_chain.invoke(query)
  print(f"Answer: {answer[:200]}...")
  print("-" * 50)

# %%[markdown]
"""
## Step 5: DOCX Report Generation
"""

# %%
# Test DOCX generation
sample_output = comprehensive_rag_chain.invoke("What are Adidas key financial metrics for 2024?")
docx_file = render_to_docx("What are Adidas key financial metrics for 2024?", sample_output)
print(f"Sample report saved to {docx_file}")

# %%
# Manual query interface for Jupyter notebook
def query_rag_system(query_text: str):
  """Process a query through the RAG system and optionally save to DOCX."""
  print(f"Processing query: {query_text}")
  
  try:
    answer = query_rag_system_streamlit(comprehensive_rag_chain, query_text)
    print(f"\nAnswer:\n{answer}")
    
    # Optionally save to DOCX
    save_to_docx = input("\nSave to DOCX? (y/n): ").lower().strip() == 'y'
    if save_to_docx:
      docx_path = render_to_docx(
        query_text,
        answer,
        f"adidas_report_{query_text.replace(' ', '_')[:20]}.docx"
      )
      print(f"✅ Report saved to {docx_path}")
      
    return answer
    
  except Exception as e:
    error_msg = f"Error processing query: {str(e)}"
    print(f"❌ {error_msg}")
    return error_msg

# Example usage:
# query_rag_system("What are Adidas key financial metrics for 2024?")
