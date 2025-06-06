"""
Streamlit frontend for the Adidas Annual Report RAG System.
"""
import os
import streamlit as st
from datetime import datetime
from typing import List, Tuple

# Import our utilities
from src.utils import render_to_docx, query_rag_system_streamlit

# Import the RAG components
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv
from src.rag_pipeline import create_rag_pipeline

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
  page_title="Adidas Annual Report Analyzer",
  page_icon="👟",
  layout="wide",
  initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_system():
  """Initialize the RAG system with caching."""
  MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
  
  if not MONGODB_ATLAS_CLUSTER_URI:
    st.error("MongoDB connection string not found in environment variables!")
    st.stop()

  # Initialize MongoDB client
  client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
  collection = client["adidas_annual_report"]["document_chunks"]

  # Initialize embeddings and vector search
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
  vector_search = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index"
  )

  return create_rag_pipeline(vector_search, k=10)

def initialize_session_state():
  """Initialize Streamlit session state."""
  if 'dialogue_history' not in st.session_state:
    st.session_state.dialogue_history: List[Tuple[str, str]] = []
  if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

def main():
  """Main Streamlit app."""
  initialize_session_state()
  
  # Header
  st.title("👟 Adidas Annual Report Analyzer")
  st.markdown("**AI-powered analysis of the Adidas 2024 Annual Report (492 pages)**")
  
  # Sidebar
  with st.sidebar:
    st.header("📊 Dashboard")
    
    # Initialize system
    if st.button("🚀 Initialize RAG System", type="primary"):
      with st.spinner("Initializing RAG system..."):
        try:
          st.session_state.rag_system = initialize_rag_system()
          st.success("✅ RAG system initialized!")
        except Exception as e:
          st.error(f"❌ Failed to initialize: {e}")
    
    # System status
    status_color = "🟢" if st.session_state.rag_system else "🔴"
    status_text = "Ready" if st.session_state.rag_system else "Not initialized"
    st.markdown(f"**System Status:** {status_color} {status_text}")
    
    # Dialogue stats
    st.markdown(f"**Total Questions:** {len(st.session_state.dialogue_history)}")
    
    # Export dialogue
    if st.session_state.dialogue_history:
      if st.button("📄 Export Dialogue to DOCX"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        docx_path = f"adidas_dialogue_{timestamp}.docx"
        
        with st.spinner("Generating DOCX..."):
          try:
            file_path = render_to_docx(st.session_state.dialogue_history, docx_path)
            
            with open(file_path, "rb") as file:
              st.download_button(
                label="⬇️ Download DOCX",
                data=file.read(),
                file_name=docx_path,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              )
            
            os.remove(file_path)
            st.success("✅ Dialogue ready for download!")
            
          except Exception as e:
            st.error(f"❌ Export failed: {e}")
    
    # Clear history
    if st.session_state.dialogue_history:
      if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.dialogue_history = []
        st.rerun()

  # Main content area
  col1, col2 = st.columns([2, 1])
  
  with col1:
    st.header("💬 Ask Questions")
    
    query = st.text_area(
      "Enter your question about the Adidas Annual Report:",
      placeholder="e.g., What are Adidas key financial metrics for 2024?",
      height=100
    )
    
    if st.button("🔍 Submit Query", disabled=not st.session_state.rag_system or not query.strip()):
      with st.spinner("Processing your query..."):
        try:
          answer = query_rag_system_streamlit(st.session_state.rag_system, query)
          st.session_state.dialogue_history.append((query, answer))
          
          st.success("✅ Query processed!")
          st.markdown("### 📋 Answer:")
          st.markdown(answer)
                  
        except Exception as e:
          st.error(f"❌ Query failed: {e}")

  with col2:
    st.header("📜 Dialogue History")
    
    if st.session_state.dialogue_history:
      for i, (question, answer) in enumerate(reversed(st.session_state.dialogue_history), 1):
        with st.expander(f"Q{len(st.session_state.dialogue_history) + 1 - i}: {question[:50]}..."):
          st.markdown(f"**Question:** {question}")
          st.markdown(f"**Answer:** {answer[:200]}..." if len(answer) > 200 else answer)
    else:
      st.info("💭 No questions asked yet. Start by asking a question about the Adidas Annual Report!")

  # Footer
  st.markdown("---")
  st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🏃‍♂️ Powered by LangChain, OpenAI, MongoDB Atlas, and Streamlit<br>
    📊 Analyzing Adidas 2024 Annual Report with Advanced RAG Pipeline
    </div>
    """, 
    unsafe_allow_html=True
  )

if __name__ == "__main__":
  main() 