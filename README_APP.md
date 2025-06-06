# Adidas Annual Report RAG System - Streamlit Frontend

## Overview
Interactive web interface for querying the Adidas 2024 Annual Report using advanced RAG pipeline with ensemble retrieval (Semantic + MMR).

## Features
- 🔍 **Query Interface**: Ask questions about the Adidas Annual Report
- 📜 **Dialogue History**: Track all Q&A sessions in sidebar and main panel
- 📄 **DOCX Export**: Export complete dialogue history to downloadable Word document
- 🚀 **Ensemble Retrieval**: Semantic search + MMR for better results
- 👟 **User-Friendly**: Clean, modern interface with automatic session tracking

## Setup

### 1. Install Dependencies
```bash
# From the repository root
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
MONGODB_ATLAS_CLUSTER_URI=your_mongodb_connection_string
LLM_MODEL=gpt-4.1-mini-2025-04-14
```

### 3. Prerequisites
- MongoDB Atlas cluster with vector search index named `vector_index`
- Database: `adidas_annual_report`
- Collection: `document_chunks`
- Data must already be loaded in MongoDB (document chunks with embeddings)
- OpenAI API key in `.env` file

## Running the App

```bash
# Navigate to the assignment directory
cd "2-Langchain Basics/2.4-VectorDatabase/assignment"

# Run the Streamlit app
streamlit run app.py
```

## Usage

1. **Initialize System**: Click "🚀 Initialize RAG System" in sidebar
2. **Ask Questions**: Type queries in the main text area
3. **View History**: See all previous Q&A in the right panel and sidebar stats
4. **Export**: Click "Export Dialogue to DOCX" to download complete conversation history
5. **Clear**: Reset dialogue history when needed

## Architecture

```
User Query → Query Classification → Query Expansion → Ensemble Retrieval → Metadata Filtering → LLM → Answer + DOCX Export
              ↓                      ↓                  ↓                     ↓                    
        [table_summary vs       [Financial Expert    [Semantic +          [Content Type       
         text_chunk]             Query Rephrasing]    MMR]                 Filtering]          
                                                            ↓
                                                   MongoDB Vector Database
                                                   (document_chunks collection)
```

## Sample Questions
- "What are Adidas key financial metrics for 2024?"
- "Tell me about sustainability initiatives and environmental impact"
- "What are the main revenue drivers?"
- "How is Adidas performing in different regions?"
- "What are the digital transformation investments?" 